import os  # 操作系统接口，用于文件路径拼接
from tqdm import tqdm  # 进度条工具，用于显示嵌入生成进度
from glob import glob  # 通配符匹配工具，用于批量查找图像文件
import torch  # PyTorch 深度学习框架，用于模型推理
from visual_bge.visual_bge.modeling import Visualized_BGE  # Visualized BGE 多模态编码模型
from elasticsearch import Elasticsearch  # Elasticsearch 官方 Python 客户端
import numpy as np  # NumPy 数值计算库，用于图像数组处理
import cv2  # OpenCV 图像处理库，用于图像绘制与拼接
from PIL import Image  # Pillow 图像库，用于图像读取与显示

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"  # 底座文本模型名称
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"  # 多模态模型权重文件路径
DATA_DIR = "../../data/C3"  # 数据根目录
INDEX_NAME = "multimodal_demo"  # Elasticsearch 索引名称
# 指向本机 Docker 中运行的 Elasticsearch 容器 elastic-labs-es8（映射到 127.0.0.1:9201）
ES_URI = "http://localhost:9201"  # Elasticsearch 服务地址
KEEP_INDEX = True  # 运行结束后是否保留索引（True=保留，便于在 Kibana 数据视图中查看；False=自动清理）

# 2. 定义工具 (编码器和可视化函数)
class Encoder:  # 编码器类
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):  # 构造方法，接收模型名与权重路径
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)  # 加载多模态模型
        self.model.eval()  # 切换为推理模式

    def encode_query(self, image_path: str, text: str) -> list[float]:  # 图文混合编码，用于构造查询向量
        with torch.no_grad():  # 关闭梯度计算以加速推理
            query_emb = self.model.encode(image=image_path, text=text)  # 将图像与文本联合编码
        return query_emb.tolist()[0]  # 转为 Python 列表并取出首条向量

    def encode_image(self, image_path: str) -> list[float]:  # 仅图像编码，用于生成待入库向量
        with torch.no_grad():  # 关闭梯度计算以加速推理
            query_emb = self.model.encode(image=image_path)  # 对单张图像编码
        return query_emb.tolist()[0]  # 转为 Python 列表并取出首条向量

def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300, row_count: int = 3) -> np.ndarray:  # 可视化函数：拼接查询图与检索结果图
    """从检索到的图像列表创建一个全景图用于可视化。"""
    panoramic_width = img_width * row_count  # 全景图宽度 = 单图宽度 × 每行图数
    panoramic_height = img_height * row_count  # 全景图高度 = 单图高度 × 行数
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)  # 创建白色画布用于摆放检索结果
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)  # 创建左侧查询图展示区域

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")  # 打开查询图像并转为 RGB 格式
    query_cv = np.array(query_pil)[:, :, ::-1]  # 转为 OpenCV 的 BGR 数组格式
    resized_query = cv2.resize(query_cv, (img_width, img_height))  # 缩放到统一展示尺寸
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))  # 添加红色边框以突出查询图
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))  # 将查询图放到展示区底部
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # 在查询图上标注 "Query" 字样

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):  # 遍历检索到的图像列表
        row, col = i // row_count, i % row_count  # 计算当前图在网格中的行列位置
        start_row, start_col = row * img_height, col * img_width  # 计算当前图的像素起始坐标

        retrieved_pil = Image.open(img_path).convert("RGB")  # 打开检索结果图像并转为 RGB
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]  # 转为 OpenCV 的 BGR 数组格式
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))  # 缩放并预留边框空间
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # 添加黑色细边框
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved  # 将结果图贴入画布对应位置

        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 在结果图左上角标注红色序号

    return np.hstack([query_display_area, panoramic_image])  # 横向拼接查询区与结果区并返回

# 3. 初始化客户端
print("--> 正在初始化编码器和Elasticsearch客户端...")  # 打印初始化提示
encoder = Encoder(MODEL_NAME, MODEL_PATH)  # 创建编码器实例并加载模型
es_client = Elasticsearch(hosts=ES_URI, request_timeout=60)  # 创建 ES 客户端连接（超时 60 秒）

# 4. 创建 Elasticsearch 索引
print(f"\n--> 正在创建索引 '{INDEX_NAME}'")  # 打印创建索引提示
if es_client.indices.exists(index=INDEX_NAME):  # 检查索引是否已存在
    es_client.indices.delete(index=INDEX_NAME)  # 若已存在则先删除，保证演示可重复运行
    print(f"已删除已存在的索引: '{INDEX_NAME}'")  # 打印删除提示

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))  # 获取 dragon 目录下全部 PNG 图像路径
if not image_list:  # 若未找到任何图像
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")  # 抛出文件未找到异常
dim = len(encoder.encode_image(image_list[0]))  # 用首张图获取向量维度

# 定义索引 Mapping（对应 Milvus 的 CollectionSchema）
mapping = {  # 索引定义字典
    "mappings": {  # 字段结构定义
        "properties": {  # 各字段属性
            # 向量字段，维度与模型的输出向量维度一致，相似度度量为余弦相似度
            "vector": {"type": "dense_vector", "dims": dim, "similarity": "cosine"},  # 稠密向量字段
            # 存储原图像路径的标量字段
            "image_path": {"type": "keyword"},  # 关键字字段，支持精确匹配与过滤
        }  # properties 定义结束
    }  # mappings 定义结束
}  # mapping 定义结束

# 创建索引（ES 无需单独建索引，kNN 检索内置 HNSW 算法）
es_client.indices.create(index=INDEX_NAME, mappings=mapping["mappings"])  # 按 Mapping 创建索引
print(f"成功创建索引: '{INDEX_NAME}'")  # 打印创建成功提示
print("索引 Mapping:")  # 打印 Mapping 标题
print(es_client.indices.get_mapping(index=INDEX_NAME))  # 查看并打印索引的 Mapping 结构

# 5. 准备并插入数据
print(f"\n--> 正在向 '{INDEX_NAME}' 插入数据")  # 打印插入数据提示
for image_path in tqdm(image_list, desc="生成图像嵌入"):  # 遍历图像并显示进度条
    vector = encoder.encode_image(image_path)  # 生成当前图像的向量
    es_client.index(index=INDEX_NAME, document={"vector": vector, "image_path": image_path})  # 将向量与路径写入 ES 文档

# 刷新索引使文档可被检索
es_client.indices.refresh(index=INDEX_NAME)  # 强制刷新使新文档立即可搜索
insert_count = es_client.count(index=INDEX_NAME)["count"]  # 统计索引中的文档总数
print(f"成功插入 {insert_count} 条数据。")  # 打印插入数量

# 6. 执行多模态检索
print(f"\n--> 正在 '{INDEX_NAME}' 中执行检索")  # 打印检索提示
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")  # 查询图像路径
query_text = "一条龙"  # 查询文本
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)  # 将查询图与文本联合编码为查询向量

search_response = es_client.search(  # 调用 Elasticsearch 检索 API
    index=INDEX_NAME,  # 指定检索的索引
    knn={  # kNN 近邻检索参数
        "field": "vector",  # 参与检索的向量字段
        "query_vector": query_vector,  # 查询向量（图文联合编码结果）
        "k": 5,  # 返回最相似的 5 个结果
        "num_candidates": 128,  # HNSW 候选数量，越大越精确但越慢
    },  # knn 参数结束
    source_includes=["image_path"],  # 结果中仅返回 image_path 字段
)["hits"]["hits"]  # 从响应中提取命中文档列表

retrieved_images = []  # 初始化检索结果图像路径列表
print("检索结果:")  # 打印检索结果标题
for i, hit in enumerate(search_response):  # 遍历命中文档
    print(f"  Top {i+1}: ID={hit['_id']}, 相似度={hit['_score']:.4f}, 路径='{hit['_source']['image_path']}'")  # 打印排名、文档 ID、相似度与路径
    retrieved_images.append(hit["_source"]["image_path"])  # 收集检索到的图像路径用于可视化

# 7. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")  # 打印可视化提示
if not retrieved_images:  # 若没有检索到任何图像
    print("没有检索到任何图像。")  # 打印空结果提示
else:  # 检索到结果时进行可视化
    panoramic_image = visualize_results(query_image_path, retrieved_images)  # 生成查询图与结果图的对比全景图
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")  # 结果图像保存路径
    cv2.imwrite(combined_image_path, panoramic_image)  # 将全景图保存为文件
    print(f"结果图像已保存到: {combined_image_path}")  # 打印保存路径
    Image.open(combined_image_path).show()  # 调用系统看图工具展示结果

if not KEEP_INDEX:  # 仅在未开启保留开关时执行清理
    es_client.indices.delete(index=INDEX_NAME)  # 删除演示索引，清理 ES 资源
    print(f"已删除索引: '{INDEX_NAME}'")  # 打印删除完成提示
else:  # 开启保留时跳过清理
    print(f"已保留索引: '{INDEX_NAME}'，可在 Kibana 数据视图中查看。")  # 打印保留提示
