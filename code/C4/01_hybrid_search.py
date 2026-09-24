import json  # 导入 json 模块，用于读取 JSON 格式的数据文件
import os  # 导入 os 模块，用于处理文件路径与判断文件是否存在
import numpy as np  # 导入 numpy 库，用于数值计算（如计算向量范数）
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker  # 从 pymilvus 导入连接、客户端、字段/集合模式、数据类型、集合、ANN 搜索请求及 RRF 融合器等核心类
from pymilvus.model.hybrid import BGEM3EmbeddingFunction  # 导入 BGE-M3 混合嵌入函数，用于同时生成稠密与稀疏向量

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"  # 定义要使用的 Milvus 集合名称
MILVUS_URI = "http://localhost:19530"  # 服务器模式
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 相对路径
BATCH_SIZE = 50  # 定义批量插入的批大小（预留配置项）

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")  # 打印正在连接 Milvus 的提示信息
connections.connect(uri=MILVUS_URI)  # 使用指定 URI 建立与 Milvus 服务的连接

print("--> 正在初始化 BGE-M3 嵌入模型...")  # 打印正在初始化嵌入模型的提示信息
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")  # 创建 BGE-M3 嵌入模型实例，关闭 FP16 并使用 CPU 运行
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")  # 打印模型初始化完成信息及稠密向量的维度

# 3. 创建 Collection
milvus_client = MilvusClient(uri=MILVUS_URI)  # 创建 MilvusClient 客户端实例，用于集合管理等操作
if milvus_client.has_collection(COLLECTION_NAME):  # 判断目标集合是否已存在
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")  # 打印正在删除已存在集合的提示信息
    milvus_client.drop_collection(COLLECTION_NAME)  # 删除已存在的同名集合，确保从干净状态开始

fields = [  # 定义集合的字段列表（即表结构）
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),  # 主键字段，字符串类型，自动生成 ID
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),  # 图片 ID 字段，字符串类型
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),  # 图片路径字段，字符串类型
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),  # 标题字段，字符串类型
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),  # 描述字段，字符串类型，长度较大
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),  # 分类字段，字符串类型
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),  # 位置字段，字符串类型
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),  # 环境字段，字符串类型
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # 稀疏向量字段，用于稀疏检索
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])  # 稠密向量字段，维度取自嵌入模型
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):  # 判断目标集合是否不存在
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")  # 打印正在创建集合的提示信息
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")  # 使用字段列表构建集合模式，并附加描述
    # 创建集合
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")  # 按模式创建集合，一致性级别设为强一致
    print("--> Collection 创建成功。")  # 打印集合创建成功信息

    # 4. 创建索引
    print("--> 正在为新集合创建索引...")  # 打印正在创建索引的提示信息
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}  # 定义稀疏向量的倒排索引参数，使用内积度量
    collection.create_index("sparse_vector", sparse_index)  # 为稀疏向量字段创建索引
    print("稀疏向量索引创建成功。")  # 打印稀疏向量索引创建成功信息

    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}  # 定义稠密向量的自动索引参数，使用内积度量
    collection.create_index("dense_vector", dense_index)  # 为稠密向量字段创建索引
    print("密集向量索引创建成功。")  # 打印稠密向量索引创建成功信息

collection = Collection(COLLECTION_NAME)  # 重新获取集合对象引用，用于后续加载与检索操作

# 5. 加载数据并插入
collection.load()  # 将集合数据加载到内存中，以便执行搜索
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")  # 打印集合已加载到内存的提示信息

if collection.is_empty:  # 判断集合是否为空（无任何数据）
    print(f"--> Collection 为空，开始插入数据...")  # 打印集合为空、即将插入数据的提示信息
    if not os.path.exists(DATA_PATH):  # 判断数据文件路径是否存在
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")  # 若文件不存在则抛出文件未找到异常
    with open(DATA_PATH, 'r', encoding='utf-8') as f:  # 以只读、UTF-8 编码方式打开数据文件
        dataset = json.load(f)  # 读取并解析 JSON 文件内容为 Python 对象

    docs, metadata = [], []  # 初始化文档文本列表与元数据列表
    for item in dataset:  # 遍历数据集中的每一条记录
        parts = [  # 收集用于拼接成检索文本的各字段内容
            item.get('title', ''),  # 获取标题字段，缺失时为空字符串
            item.get('description', ''),  # 获取描述字段，缺失时为空字符串
            item.get('location', ''),  # 获取位置字段，缺失时为空字符串
            item.get('environment', ''),  # 获取环境字段，缺失时为空字符串
            # *item.get('combat_details', {}).get('combat_style', []),  # 注释：可选字段——战斗风格
            # *item.get('combat_details', {}).get('abilities_used', []),  # 注释：可选字段——使用的能力
            # item.get('scene_info', {}).get('time_of_day', '')  # 注释：可选字段——场景时间
        ]
        docs.append(' '.join(filter(None, parts)))  # 用空格拼接非空字段，组成文档文本并加入列表
        metadata.append(item)  # 将该条原始记录加入元数据列表
    print(f"--> 数据加载完成，共 {len(docs)} 条。")  # 打印加载完成的文档数量

    print("--> 正在生成向量嵌入...")  # 打印正在生成向量嵌入的提示信息
    embeddings = ef(docs)  # 调用嵌入模型批量生成所有文档的稠密与稀疏向量
    print("--> 向量生成完成。")  # 打印向量生成完成信息

    print("--> 正在分批插入数据...")  # 打印正在插入数据的提示信息
    # 为每个字段准备批量数据
    img_ids = [doc["img_id"] for doc in metadata]  # 提取所有记录的图片 ID 组成列表
    paths = [doc["path"] for doc in metadata]  # 提取所有记录的路径组成列表
    titles = [doc["title"] for doc in metadata]  # 提取所有记录的标题组成列表
    descriptions = [doc["description"] for doc in metadata]  # 提取所有记录的描述组成列表
    categories = [doc["category"] for doc in metadata]  # 提取所有记录的分类组成列表
    locations = [doc["location"] for doc in metadata]  # 提取所有记录的位置组成列表
    environments = [doc["environment"] for doc in metadata]  # 提取所有记录的环境组成列表
    
    # 获取向量
    sparse_vectors = embeddings["sparse"]  # 取出生成的稀疏向量集合
    dense_vectors = embeddings["dense"]  # 取出生成的稠密向量集合
    
    # 插入数据
    collection.insert([  # 将各字段数据按列方式插入集合
        img_ids,  # 图片 ID 列
        paths,  # 路径列
        titles,  # 标题列
        descriptions,  # 描述列
        categories,  # 分类列
        locations,  # 位置列
        environments,  # 环境列
        sparse_vectors,  # 稀疏向量列
        dense_vectors  # 稠密向量列
    ])
    
    collection.flush()  # 刷新集合，确保插入的数据持久化并可见
    print(f"--> 数据插入完成，总数: {collection.num_entities}")  # 打印插入完成后的实体总数
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")  # 集合非空时打印已有数据量并跳过插入

# 6. 执行搜索
search_query = "悬崖上的巨龙"  # 定义用于搜索的查询文本
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'  # 定义过滤表达式，仅检索指定分类的数据
top_k = 5  # 定义返回的 Top-K 结果数量

print(f"\n{'='*20} 开始混合搜索 {'='*20}")  # 打印带分隔符的搜索开始标题
print(f"查询: '{search_query}'")  # 打印当前查询文本
print(f"过滤器: '{search_filter}'")  # 打印当前过滤条件

query_embeddings = ef([search_query])  # 对查询文本生成嵌入向量（返回稠密与稀疏两种）
dense_vec = query_embeddings["dense"][0]  # 取出查询的稠密向量（第一条）
sparse_vec = query_embeddings["sparse"]._getrow(0)  # 取出查询的稀疏向量（第一行）

# 打印向量信息
print("\n=== 向量信息 ===")  # 打印向量信息标题
print(f"密集向量维度: {len(dense_vec)}")  # 打印稠密向量的维度大小
print(f"密集向量前5个元素: {dense_vec[:5]}")  # 打印稠密向量的前 5 个元素值
print(f"密集向量范数: {np.linalg.norm(dense_vec):.4f}")  # 打印稠密向量的 L2 范数（保留 4 位小数）

print(f"\n稀疏向量维度: {sparse_vec.shape[1]}")  # 打印稀疏向量的维度大小
print(f"稀疏向量非零元素数量: {sparse_vec.nnz}")  # 打印稀疏向量的非零元素个数
print("稀疏向量前5个非零元素:")  # 打印提示：即将列出稀疏向量的前 5 个非零元素
for i in range(min(5, sparse_vec.nnz)):  # 遍历至多 5 个非零元素
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")  # 打印每个非零元素的索引与数值
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)  # 计算稀疏向量的非零密度百分比
print(f"\n稀疏向量密度: {density:.8f}%")  # 打印稀疏向量密度（保留 8 位小数）

# 定义搜索参数
search_params = {"metric_type": "IP", "params": {}}  # 定义检索参数：使用内积（IP）度量，参数为空

# 先执行单独的搜索
print("\n--- [单独] 密集向量搜索结果 ---")  # 打印单独稠密向量搜索的结果标题
dense_results = collection.search(  # 在稠密向量字段上执行相似度搜索
    [dense_vec],  # 查询向量列表（稠密向量）
    anns_field="dense_vector",  # 指定进行 ANN 搜索的字段为稠密向量
    param=search_params,  # 传入检索参数
    limit=top_k,  # 限制返回结果数量为 top_k
    expr=search_filter,  # 应用过滤表达式限定检索范围
    output_fields=["title", "path", "description", "category", "location", "environment"]  # 指定需要返回的标量字段
)[0]  # 取第一个查询的结果列表（单个查询）

for i, hit in enumerate(dense_results):  # 遍历稠密检索结果并获取序号与命中项
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")  # 打印排名、标题与相似度分数
    print(f"    路径: {hit.entity.get('path')}")  # 打印命中项的图片路径
    print(f"    描述: {hit.entity.get('description')[:100]}...")  # 打印命中项描述的前 100 个字符

print("\n--- [单独] 稀疏向量搜索结果 ---")  # 打印单独稀疏向量搜索的结果标题
sparse_results = collection.search(  # 在稀疏向量字段上执行相似度搜索
    [sparse_vec],  # 查询向量列表（稀疏向量）
    anns_field="sparse_vector",  # 指定进行 ANN 搜索的字段为稀疏向量
    param=search_params,  # 传入检索参数
    limit=top_k,  # 限制返回结果数量为 top_k
    expr=search_filter,  # 应用过滤表达式限定检索范围
    output_fields=["title", "path", "description", "category", "location", "environment"]  # 指定需要返回的标量字段
)[0]  # 取第一个查询的结果列表（单个查询）

for i, hit in enumerate(sparse_results):  # 遍历稀疏检索结果并获取序号与命中项
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")  # 打印排名、标题与相似度分数
    print(f"    路径: {hit.entity.get('path')}")  # 打印命中项的图片路径
    print(f"    描述: {hit.entity.get('description')[:100]}...")  # 打印命中项描述的前 100 个字符

print("\n--- [混合] 稀疏+密集向量搜索结果 ---")  # 打印混合搜索的结果标题
# 创建 RRF 融合器
rerank = RRFRanker(k=60)  # 创建 RRF 融合器实例，k 参数设为 60 用于融合排序

# 创建搜索请求
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)  # 构造稠密向量的 ANN 搜索请求
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)  # 构造稀疏向量的 ANN 搜索请求

# 执行混合搜索
results = collection.hybrid_search(  # 执行稠密与稀疏的混合检索
    [sparse_req, dense_req],  # 传入多个搜索请求
    rerank=rerank,  # 指定使用 RRF 融合器对多路结果重排
    limit=top_k,  # 限制最终返回结果数量为 top_k
    output_fields=["title", "path", "description", "category", "location", "environment"]  # 指定需要返回的标量字段
)[0]  # 取第一个查询的结果列表（单个查询）

# 打印最终结果
for i, hit in enumerate(results):  # 遍历混合检索结果并获取序号与命中项
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")  # 打印排名、标题与融合后的分数
    print(f"    路径: {hit.entity.get('path')}")  # 打印命中项的图片路径
    print(f"    描述: {hit.entity.get('description')[:100]}...")  # 打印命中项描述的前 100 个字符

# 7. 清理资源
milvus_client.release_collection(collection_name=COLLECTION_NAME)  # 从内存中释放集合，释放相关资源
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")  # 打印集合已释放的提示信息
milvus_client.drop_collection(COLLECTION_NAME)  # 删除集合，清理本次示例产生的数据
print(f"已删除 Collection: '{COLLECTION_NAME}'")  # 打印集合已删除的提示信息
