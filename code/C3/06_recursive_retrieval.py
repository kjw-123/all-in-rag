import os  # 导入自带工具 os，用来读取电脑里的环境变量（比如密钥）
import pandas as pd  # 导入 pandas，用来读取 Excel 表格
from dotenv import load_dotenv  # 导入小工具，用来从 .env 文件里读取密钥
from llama_index.core import VectorStoreIndex  # 导入"按像不像来找内容"的工具，可以理解成一本自动目录
from llama_index.core.schema import IndexNode  # 导入"卡片"工具：每张卡片是一段简介，背后指向真正的数据
from llama_index.experimental.query_engine import PandasQueryEngine  # 导入"会算表格的问答助手"：能用大白话查一张表
from llama_index.core.retrievers import RecursiveRetriever  # 导入"两步找资料"的工具：先找卡片，再顺着卡片找到具体数据
from llama_index.core.query_engine import RetrieverQueryEngine  # 导入"问答机"：把找资料 + 大模型 拼成一台会回答问题的机器
from llama_index.llms.deepseek import DeepSeek  # 导入"语文老师"大模型（DeepSeek 聊天模型）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 导入"把文字变成数字指纹"的工具
from llama_index.core import Settings  # 导入"全局设置"，用来统一指定用哪个大模型、哪个指纹工具

load_dotenv()  # 从 .env 文件里把密钥读进来（比如 DEEPSEEK_API_KEY）

# 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))  # 指定全局的"语文老师"：用 deepseek 聊天模型，密钥从环境变量取
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")  # 指定全局的"数字指纹"工具：用中文小模型

# 1.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'  # Excel 文件的位置（相对本脚本所在文件夹）
xls = pd.ExcelFile(excel_file)  # 打开这个 Excel，方便一年一年地翻里面的表（每个表是一年）

df_query_engines = {}  # 准备一个字典，记录"哪一年 -> 对应哪个表格助手"
all_nodes = []  # 准备一个列表，收集所有年份的"卡片"

for sheet_name in xls.sheet_names:  # 一年一年地翻（每个表名就是一年，如"年份_1994"）
    df = pd.read_excel(xls, sheet_name=sheet_name)  # 把这一年读成一张表
    
    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)  # 给这一年的表配一个"表格助手"，能用大白话问它，它去算
    
    # 为当前工作表创建一个摘要节点（IndexNode）
    year = sheet_name.replace('年份_', '')  # 从表名里把年份数字抠出来（"年份_1994" -> "1994"）
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"  # 给这一年写一句简介
    node = IndexNode(text=summary, index_id=sheet_name)  # 把简介做成一张"卡片"，卡片上写着"我是哪一年"的门牌号
    all_nodes.append(node)  # 把这张卡片收进总目录
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine  # 记下"这一年的门牌号 -> 它的表格助手"，方便后面按门牌号找到它

# 2. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)  # 把所有"卡片"整理成一本会自动按像不像来找的目录

# 3. 创建递归检索器
# 3.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)  # 第一步的"找卡片员"：只挑出最像的那 1 张卡片

# 3.2 创建递归检索器
recursive_retriever = RecursiveRetriever(  # 造一台"两步找资料"的机器
    "vector",  # 给第一步的"找卡片员"起个名字
    retriever_dict={"vector": vector_retriever},  # 名字 -> 找卡片员 的对应关系
    query_engine_dict=df_query_engines,  # 门牌号 -> 表格助手 的对应关系（找到卡片后，凭门牌号跳到对应助手）
    verbose=True,  # 打开日志，打印"找了哪张卡、跳到了哪个表"的过程
)

# 4. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)  # 把两步找资料的机器，包装成一台能直接"回答问题"的问答机

# 5. 执行查询
query = "1994年评分人数最少的电影是哪一部？"  # 要问的问题
print(f"查询: {query}")  # 先把问题打印出来
response = query_engine.query(query)  # 让问答机自己跑：找卡片 -> 跳表格算数 -> 组织成答案
print(f"回答: {response}")  # 打印最终答案
