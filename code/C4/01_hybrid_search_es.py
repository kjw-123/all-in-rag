"""用阿里云百炼生成稀疏和密集向量，再用 Elasticsearch 做混合检索。

运行前在环境变量中设置 DASHSCOPE_API_KEY 和 DASHSCOPE_HTTP_BASE_URL。
后者是阿里云百炼 DashScope API 地址，例如北京地域：
https://dashscope.aliyuncs.com/api/v1
云端会返回向量，本脚本不会下载模型权重。
"""

import json  # JSON 解析库，用于读取 dragon.json 数据文件
import os  # 操作系统接口，用于读取环境变量
from pathlib import Path  # 路径处理工具，用于定位项目内的数据文件

import numpy as np  # NumPy 数值计算库，用于向量范数校验等数值运算
import requests  # 调用阿里云百炼的 DashScope HTTP 接口；不会在本地下载模型
from dotenv import load_dotenv  # 读取仓库根目录的 .env，让 PyCharm 直接运行时也能拿到配置
from elasticsearch import Elasticsearch, helpers  # Elasticsearch 官方客户端，helpers 提供批量写入助手


# 整体流程：读取龙的数据 → 阿里云百炼返回两种向量 → 写入 ES → 两路搜索 → Python 端 RRF 融合。
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录（本文件位于 code/C4/ 下，向上两级即仓库根）
load_dotenv(PROJECT_ROOT / ".env")  # 只读取该项目的 .env，不依赖运行时工作目录
DATA_PATH = PROJECT_ROOT / "data" / "C4" / "metadata" / "dragon.json"  # 龙类数据文件路径
ES_URI = os.getenv("ES_URL", "http://localhost:9201")  # ES 服务地址，默认指向本机 Docker 的 elastic-labs-es8
MODEL_NAME = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding")  # 用户选定的阿里云云端向量模型
INDEX_NAME = f"dragon_hybrid_{MODEL_NAME.replace('-', '_').replace('.', '_')}_es"  # 不同模型使用不同索引，避免向量混用
DENSE_DIM = 1024  # 百炼这两种模型都支持 1024 维
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 百炼密钥只从环境变量读取
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_HTTP_BASE_URL", "").rstrip("/")  # 业务空间的 DashScope API 地址
QUERY = "悬崖上的巨龙"  # 演示用的查询语句
ALLOWED_CATEGORIES = ["western_dragon", "chinese_dragon", "movie_character"]  # 检索时的类别过滤白名单
TOP_K = 5  # 每路检索与融合后保留的结果条数
RRF_K = 60  # RRF（倒数排名融合）公式中的平滑常数 k
DROP_INDEX_AT_END = False  # 运行结束是否删除索引：False 保留（可重复运行、可在 Kibana 查看），True 则清理

# 只返回展示用字段，避免把上千维的向量打印到终端。
DISPLAY_FIELDS = [  # 需要在检索结果中展示的字段列表
    "img_id", "path", "title", "description", "category", "location", "environment"  # 数据中的可见字段
]  # 展示字段列表定义结束


def load_data() -> list[dict]:  # 定义读取数据函数
    """读取原 Milvus 示例使用的同一份龙类数据。"""  # 函数用途说明
    if not DATA_PATH.is_file():  # 检查数据文件是否存在
        raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")  # 不存在时给出明确报错
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))  # 以 UTF-8 读取并解析 JSON 数据
    if not data or len({item["img_id"] for item in data}) != len(data):  # 校验数据非空且 img_id 不重复
        raise ValueError("数据不能为空，且每条数据的 img_id 必须唯一。")  # 校验失败时中止
    return data  # 返回原始数据列表


def document_text(item: dict) -> str:  # 定义待编码文本拼接函数
    """与原脚本一致：用标题、描述、地点和环境生成待编码文本。"""  # 函数用途说明
    return " ".join(  # 用空格把各字段拼接成一段文本
        str(item.get(field, ""))  # 取出字段值并转为字符串（缺失则用空串）
        for field in ("title", "description", "location", "environment")  # 参与拼接的字段名
        if item.get(field)  # 只保留非空字段
    )  # 文本拼接结束


def validate_cloud_config() -> None:
    """在写入 ES 前检查配置，避免缺少密钥时创建空索引。"""
    if MODEL_NAME not in {"text-embedding-v4", "qwen3.7-text-embedding"}:
        raise ValueError("DASHSCOPE_EMBEDDING_MODEL 仅支持 text-embedding-v4 或 qwen3.7-text-embedding。")
    if not DASHSCOPE_API_KEY:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请先在终端环境变量中配置阿里云百炼 API Key。")
    if not DASHSCOPE_BASE_URL.startswith("https://") or not DASHSCOPE_BASE_URL.endswith("/api/v1"):
        raise ValueError("缺少或无效的 DASHSCOPE_HTTP_BASE_URL；北京地域可用 https://dashscope.aliyuncs.com/api/v1")


def sparse_to_dict(items: list[dict]) -> dict[str, float]:
    """把百炼返回的 [{index, value, token}, ...] 转成 ES 的 {词元 ID: 权重}。"""
    result = {}
    for item in items:
        token_id, weight = str(item["index"]), float(item["value"])
        if np.isfinite(weight) and weight > 0:
            result[token_id] = weight
    if not result:
        raise ValueError("百炼没有返回可用的稀疏词元权重，请检查 output_type='dense&sparse'。")
    return result


def embed_texts(texts: list[str], text_type: str) -> list[dict]:
    """批量请求阿里云百炼；每条返回 dense 和 sparse，两类向量来自同一个模型。"""
    if not texts or text_type not in {"document", "query"}:
        raise ValueError("texts 不能为空，text_type 必须为 document 或 query。")
    endpoint = DASHSCOPE_BASE_URL + "/services/embeddings/text-embedding/text-embedding"
    output = []
    for start in range(0, len(texts), 10):  # v4 一次最多 10 条；qwen3.7 也兼容此批量大小
        batch = texts[start:start + 10]
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            json={"model": MODEL_NAME, "input": {"texts": batch}, "parameters": {
                "dimension": DENSE_DIM, "output_type": "dense&sparse", "text_type": text_type,
            }},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code") or not body.get("output", {}).get("embeddings"):
            raise RuntimeError(f"百炼向量接口失败：{body.get('code')} {body.get('message')}；请求 ID：{body.get('request_id')}")
        embeddings = body["output"]["embeddings"]
        if len(embeddings) != len(batch):
            raise ValueError("百炼返回的向量数量与输入文本数量不一致。")
        by_index = {item["text_index"]: item for item in embeddings}
        if set(by_index) != set(range(len(batch))):
            raise ValueError("百炼返回的 text_index 不完整或重复。")
        for index in range(len(batch)):
            item = by_index[index]
            dense = np.asarray(item["embedding"], dtype=float)
            if dense.shape != (DENSE_DIM,) or not np.all(np.isfinite(dense)):
                raise ValueError(f"百炼返回的密集向量无效，预期 {DENSE_DIM} 维。")
            norm = np.linalg.norm(dense)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("百炼返回了零向量，无法用于点积检索。")
            if not item.get("sparse_embedding"):
                raise ValueError("百炼响应中缺少 sparse_embedding，请检查模型及 output_type。")
            output.append({
                "dense": (dense / norm).tolist(),  # ES dot_product 需要单位长度向量
                "sparse": sparse_to_dict(item["sparse_embedding"]),
            })
    return output


def ensure_index(client: Elasticsearch, dense_dim: int) -> None:  # 定义索引检查/创建函数
    """只在索引不存在时创建它；已有索引不删除、不重建。"""  # 函数用途说明
    if client.indices.exists(index=INDEX_NAME):  # 索引已存在时校验其 Mapping
        properties = client.indices.get_mapping(index=INDEX_NAME)[INDEX_NAME]["mappings"]["properties"]  # 取出现有字段定义
        if (  # 校验向量字段的类型、维度与相似度是否与预期一致
            properties.get("sparse_vector", {}).get("type") != "sparse_vector"  # 稀疏字段必须是 sparse_vector 类型
            or properties.get("dense_vector", {}).get("type") != "dense_vector"  # 密集字段必须是 dense_vector 类型
            or properties["dense_vector"].get("dims") != dense_dim  # 密集向量维度必须与模型输出一致
            or properties["dense_vector"].get("similarity") != "dot_product"  # 相似度必须是 dot_product（对应 Milvus 的 IP）
        ):  # 校验条件结束
            raise ValueError(f"已有索引 {INDEX_NAME} 的向量 Mapping 不匹配，请检查后使用新的索引名。")  # 不匹配则中止，避免写入脏数据
        print(f"--> 使用已有 ES 索引：{INDEX_NAME}")  # 提示复用已有索引
        return  # 复用已有索引后直接返回，不再重复创建

    client.indices.create(  # 创建 ES 索引
        index=INDEX_NAME,  # 指定索引名称
        settings={"number_of_shards": 1},  # 单分片设置，便于本地演示
        mappings={"properties": {  # 字段映射定义
            "img_id": {"type": "keyword"},  # 图像 ID，关键字类型（精确匹配，同时用作文档 ID）
            "path": {"type": "keyword"},  # 图像路径，关键字类型
            "title": {"type": "text"},  # 标题，全文类型
            "description": {"type": "text"},  # 描述，全文类型
            "category": {"type": "keyword"},  # 类别，关键字类型（供 terms 过滤使用）
            "location": {"type": "keyword"},  # 地点，关键字类型
            "environment": {"type": "keyword"},  # 环境，关键字类型
            "sparse_vector": {"type": "sparse_vector"},  # 百炼稀疏向量字段（词元 ID → 权重）
            "dense_vector": {  # 百炼密集向量字段
                "type": "dense_vector", "dims": dense_dim,  # 向量类型与维度（= 模型输出维度）
                "index": True, "similarity": "dot_product",  # 建立 kNN 索引，相似度用内积（要求向量为单位长度）
            },  # 密集向量字段定义结束
        }},  # 字段映射定义结束
    )  # 创建索引调用结束
    print(f"--> 已创建 ES 索引：{INDEX_NAME}")  # 提示索引创建成功


def index_missing_documents(client: Elasticsearch, data: list[dict]) -> None:  # 定义幂等写入函数
    """仅写入尚不存在的 img_id；重复运行不会重复插入或覆盖已有文档。"""  # 函数用途说明
    missing = [item for item in data if not client.exists(index=INDEX_NAME, id=item["img_id"])]  # 找出尚未入库的数据
    if not missing:  # 若全部数据都已存在
        print(f"--> {len(data)} 条文档已存在，跳过写入。")  # 提示跳过写入
        return  # 无需继续写入

    print(f"--> 请求阿里云百炼，为 {len(missing)} 条新文档生成稀疏和密集向量...")
    embeddings = embed_texts([document_text(item) for item in missing], "document")  # 批量调用云端模型
    actions = []  # 初始化批量写入动作列表
    for row, item in enumerate(missing):  # 逐条构造待写入文档
        source = {field: item[field] for field in DISPLAY_FIELDS}  # 复制展示字段作为文档内容
        source["sparse_vector"] = embeddings[row]["sparse"]  # 写入该条的稀疏向量
        source["dense_vector"] = embeddings[row]["dense"]  # 写入该条的密集向量
        actions.append({  # 组装一条 bulk 动作
            "_op_type": "create", "_index": INDEX_NAME,  # 使用 create 语义，保证不会覆盖已有文档
            "_id": item["img_id"], "_source": source,  # 用 img_id 作为稳定文档 ID，防重复
        })  # 单条动作组装结束
    success, _ = helpers.bulk(client, actions, refresh="wait_for")  # 批量写入并等待刷新，保证随后可立即检索
    print(f"--> 已写入 {success} 条文档；稳定 ID 可防止重复。")  # 提示实际写入条数


def sparse_search(client: Elasticsearch, query_vector: dict[str, float]) -> list[dict]:  # 定义稀疏检索函数
    """在 ES 的 sparse_vector 字段上做百炼词元权重检索。"""  # 函数用途说明
    response = client.search(  # 调用 ES 检索接口
        index=INDEX_NAME,  # 指定检索的索引
        size=TOP_K,  # 返回条数
        source_includes=DISPLAY_FIELDS,  # 只返回展示字段，避免携带向量
        query={"bool": {  # 布尔查询：必须命中稀疏向量 + 必须满足类别过滤
            "must": [{"sparse_vector": {  # 稀疏向量查询子句
                "field": "sparse_vector", "query_vector": query_vector, "prune": False,  # 字段名、查询向量、关闭剪枝以保精度
            }}],  # 稀疏查询子句结束
            "filter": [{"terms": {"category": ALLOWED_CATEGORIES}}],  # 类别过滤（对结果起筛选作用，不参与打分）
        }},  # 布尔查询结束
    )  # 检索调用结束
    return response["hits"]["hits"]  # 返回命中文档列表


def dense_search(client: Elasticsearch, query_vector: list[float]) -> list[dict]:  # 定义密集 kNN 检索函数
    """在 ES 的 dense_vector 字段上做 kNN，并在检索时过滤类别。"""  # 函数用途说明
    response = client.search(  # 调用 ES 检索接口
        index=INDEX_NAME,  # 指定检索的索引
        size=TOP_K,  # 返回条数
        source_includes=DISPLAY_FIELDS,  # 只返回展示字段
        knn={  # kNN 近似近邻参数
            "field": "dense_vector", "query_vector": query_vector,  # 参与检索的向量字段与查询向量
            "k": TOP_K, "num_candidates": 128,  # 返回近邻数与 HNSW 候选数量（越大越精确、越慢）
            "filter": {"terms": {"category": ALLOWED_CATEGORIES}},  # 在 kNN 阶段就按类别过滤
        },  # kNN 参数结束
    )  # 检索调用结束
    return response["hits"]["hits"]  # 返回命中文档列表


def rrf_fuse(sparse_hits: list[dict], dense_hits: list[dict]) -> list[dict]:  # 定义 RRF 融合函数
    """按原脚本的 k=60 融合排名；只使用名次，不直接相加不同检索器的分数。"""  # 函数用途说明
    fused = {}  # 以文档 ID 为键累积融合分数
    for branch, hits in (("稀疏", sparse_hits), ("密集", dense_hits)):  # 依次处理两路检索结果
        for rank, hit in enumerate(hits, start=1):  # 遍历某一路的排名（从 1 开始）
            doc_id = hit["_id"]  # 取出文档 ID
            entry = fused.setdefault(doc_id, {"hit": hit, "score": 0.0, "ranks": {}})  # 首次出现则初始化融合条目
            entry["score"] += 1.0 / (RRF_K + rank)  # 累加 RRF 分数 1/(k+rank)
            entry["ranks"][branch] = rank  # 记录该文档在该路中的名次
    return sorted(fused.values(), key=lambda item: (-item["score"], item["hit"]["_id"]))[:TOP_K]  # 按分数降序（同分按 ID）取前 K 条


def print_hits(label: str, hits: list[dict]) -> None:  # 定义结果打印函数
    """展示某一路搜索的原始排名和 ES 分数。"""  # 函数用途说明
    print(f"\n--- {label} ---")  # 打印该路检索的标题
    for rank, hit in enumerate(hits, start=1):  # 遍历命中文档
        source = hit["_source"]  # 取出文档的展示字段
        print(f"{rank}. {source['title']} (ES _score: {hit['_score']:.4f})")  # 打印名次、标题与 ES 分数
        print(f"   路径: {source['path']}")  # 打印图像路径
        print(f"   描述: {source['description'][:100]}...")  # 打印描述前 100 字
    if not hits:  # 若该路没有命中
        print("无命中结果。")  # 打印空结果提示


def main() -> None:  # 定义主流程函数
    data = load_data()  # 读取龙类数据
    validate_cloud_config()  # 密钥和云端地址缺失时提前报错，不创建空索引
    client = Elasticsearch(ES_URI, request_timeout=60)  # 创建 ES 客户端（超时设为 60 秒）
    if not client.ping():  # 测试 ES 连通性
        raise ConnectionError(f"无法连接 Elasticsearch：{ES_URI}，请检查 Docker 容器与端口。")  # 连接失败则给出明确报错
    print(f"--> 已连接 Elasticsearch：{ES_URI}")  # 提示连接成功

    print(f"--> 使用阿里云百炼云端模型：{MODEL_NAME}（无需下载模型）")
    query_embeddings = embed_texts([QUERY], "query")[0]  # 用 query 模式生成查询的两种向量
    dense_vector = query_embeddings["dense"]  # 取出查询的密集向量
    sparse_vector = query_embeddings["sparse"]  # 取出查询的稀疏向量
    ensure_index(client, DENSE_DIM)  # 云端调用成功后再创建索引，避免密钥失效时留下空索引
    index_missing_documents(client, data)  # 幂等写入尚未入库的文档
    print(f"\n查询：{QUERY}；类别过滤：{ALLOWED_CATEGORIES}")  # 打印查询语句与过滤条件
    print(f"密集向量维度：{len(dense_vector)}；L2 范数：{np.linalg.norm(dense_vector):.4f}")  # 打印密集向量维度与范数
    print(f"稀疏向量非零词元：{len(sparse_vector)}")  # 打印稀疏向量的非零词元数量

    sparse_hits = sparse_search(client, sparse_vector)  # 单独执行稀疏向量检索
    dense_hits = dense_search(client, dense_vector)  # 单独执行密集向量 kNN 检索
    print_hits("单独：阿里云百炼稀疏向量", sparse_hits)  # 打印稀疏检索的结果
    print_hits("单独：阿里云百炼密集向量 kNN", dense_hits)  # 打印密集检索的结果

    print("\n--- 混合：Python RRF（k=60）---")  # 打印 RRF 融合结果标题
    for rank, item in enumerate(rrf_fuse(sparse_hits, dense_hits), start=1):  # 遍历融合后的最终结果
        source = item["hit"]["_source"]  # 取出文档的展示字段
        print(f"{rank}. {source['title']} (RRF: {item['score']:.4f}；分路名次：{item['ranks']})")  # 打印名次、RRF 分数与各路名次
        print(f"   路径: {source['path']}")  # 打印图像路径
        print(f"   描述: {source['description'][:100]}...")  # 打印描述前 100 字

    if DROP_INDEX_AT_END:  # 按开关决定是否清理索引
        client.indices.delete(index=INDEX_NAME)  # 删除本示例的索引
        print(f"--> 已删除索引：{INDEX_NAME}")  # 提示索引已删除
    else:  # 默认保留索引
        print(f"--> 已保留索引：{INDEX_NAME}（可在 Kibana 中查看；如需清理请将 DROP_INDEX_AT_END 设为 True）")  # 提示索引已保留


if __name__ == "__main__":  # 当本文件被直接运行时
    main()  # 执行主流程
