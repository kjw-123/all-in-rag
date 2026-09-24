import os  # 导入自带工具 os，用来读取电脑里的环境变量（比如密钥）
import pandas as pd  # 导入 pandas，用来读取 Excel 表格
from dotenv import load_dotenv  # 导入小工具，用来从 .env 文件里读取密钥
from llama_index.core import VectorStoreIndex, Document, Settings  # 导入：自动目录（向量索引）、文档对象、全局设置
from llama_index.core.retrievers import VectorIndexRetriever  # 导入"找资料员"：按像不像从目录里挑内容
from llama_index.core.query_engine import RetrieverQueryEngine  # 导入"问答机"：把找资料 + 大模型 拼成会回答问题的机器
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter  # 导入过滤器：按标签精确筛选，比如只选某一年
from llama_index.llms.deepseek import DeepSeek  # 导入"语文老师"大模型（DeepSeek 聊天模型）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 导入"把文字变成数字指纹"的工具

load_dotenv()  # 从 .env 文件里把密钥读进来（比如 DEEPSEEK_API_KEY）

# 配置模型
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))  # 指定全局的"语文老师"：用 deepseek 聊天模型，密钥从环境变量取
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")  # 指定全局的"数字指纹"工具：用中文小模型

# 1. 加载和预处理数据
excel_file = '../../data/C3/excel/movie.xlsx'  # Excel 文件的位置（相对本脚本所在文件夹）
xls = pd.ExcelFile(excel_file)  # 打开这个 Excel，方便一年一年地翻里面的表（每个表是一年）

summary_docs = []  # 装"简介卡片"的列表：每张卡片是一句话，用来做路由
content_docs = []  # 装"整张表内容"的列表：把表格转成文字，用来最后答题

print("开始加载和处理Excel文件...")  # 提示：开始读取数据
for sheet_name in xls.sheet_names:  # 一年一年地翻（每个表名就是一年，如"年份_1994"）
    df = pd.read_excel(xls, sheet_name=sheet_name)  # 把这一年读成一张表
    
    # 数据清洗
    if '评分人数' in df.columns:  # 如果表里有"评分人数"这一列，就先清洗它
        df['评分人数'] = df['评分人数'].astype(str).str.replace('人评价', '').str.strip()  # 去掉"人评价"字样和空格，只留数字
        df['评分人数'] = pd.to_numeric(df['评分人数'], errors='coerce').fillna(0).astype(int)  # 转成数字，转不了的当 0，最后取整数

    # 创建摘要文档 (用于路由)
    year = sheet_name.replace('年份_', '')  # 从表名里把年份数字抠出来（"年份_1994" -> "1994"）
    summary_text = f"这个表格包含了年份为 {year} 的电影信息，包括电影名称、导演、评分、评分人数等。"  # 给这一年写一句简介
    summary_doc = Document(  # 把简介做成一张"卡片"
        text=summary_text,  # 卡片上的文字（简介）
        metadata={"sheet_name": sheet_name}  # 给卡片贴个标签，记下它属于哪一年
    )
    summary_docs.append(summary_doc)  # 把这张卡片收进"简介卡片"列表
    
    # 创建内容文档 (用于最终问答)
    content_text = df.to_string(index=False)  # 把整张表转成一段文字（去掉行号）
    content_doc = Document(  # 把整张表内容做成一份"文档"
        text=content_text,  # 文档内容就是整张表的文字
        metadata={"sheet_name": sheet_name}  # 同样贴标签，记下属于哪一年
    )
    content_docs.append(content_doc)  # 把这份文档收进"内容文档"列表

print("数据加载和处理完成。\n")  # 提示：数据准备完毕

# 2. 构建向量索引
# 使用默认的内存SimpleVectorStore，它支持元数据过滤

# 2.1 为摘要创建索引
summary_index = VectorStoreIndex(summary_docs)  # 把所有"简介卡片"整理成一本会自动按像不像来找的目录（用于路由）

# 2.2 为内容创建索引
content_index = VectorStoreIndex(content_docs)  # 把所有"整张表内容"整理成另一本目录（用于最终答题）

print("摘要索引和内容索引构建完成。\n")  # 提示：两本目录都建好了

# 3. 定义两步式查询逻辑
def query_safe_recursive(query_str):  # 定义一个函数，接收问题，走"两步"返回答案
    print(f"--- 开始执行查询 ---")  # 打印分隔提示
    print(f"查询: {query_str}")  # 打印要问的问题
    
    # 第一步：路由 - 在摘要索引中找到最相关的表格
    print("\n第一步：在摘要索引中进行路由...")  # 提示：进入第一步
    summary_retriever = VectorIndexRetriever(index=summary_index, similarity_top_k=1)  # 在"简介卡片"目录里找，只挑最像的 1 张
    retrieved_nodes = summary_retriever.retrieve(query_str)  # 拿问题去找，返回挑中的卡片
    
    if not retrieved_nodes:  # 如果一张卡片都没找到
        return "抱歉，未能找到相关的电影年份信息。"  # 直接回复：找不到
    
    # 获取匹配到的工作表名称
    matched_sheet_name = retrieved_nodes[0].node.metadata['sheet_name']  # 从挑中的卡片标签里，读出它属于哪一年
    print(f"路由结果：匹配到工作表 -> {matched_sheet_name}")  # 打印：路由到了哪一年
    
    # 第二步：检索 - 在内容索引中根据工作表名称过滤并检索具体内容
    print("\n第二步：在内容索引中检索具体信息...")  # 提示：进入第二步
    content_retriever = VectorIndexRetriever(  # 在"整张表内容"目录里找
        index=content_index,  # 用内容目录来查
        similarity_top_k=1, # 通常只返回最匹配的整个表格即可
        filters=MetadataFilters(  # 加上筛选条件：只在刚路由到的那一年里找
            filters=[ExactMatchFilter(key="sheet_name", value=matched_sheet_name)]  # 条件是标签 sheet_name 等于匹配到的年份
        )
    )
    
    # 创建查询引擎并执行查询
    query_engine = RetrieverQueryEngine.from_args(content_retriever)  # 把"内容找资料员"包装成一台会回答问题的机器
    response = query_engine.query(query_str)  # 让机器根据找到的那张表来回答问题
    
    print("--- 查询执行结束 ---\n")  # 提示：查询结束
    return response  # 返回答案

# 4. 执行查询
query = "1994年评分人数最少的电影是哪一部？"  # 要问的问题
response = query_safe_recursive(query)  # 调用上面定义的函数，走两步得到答案

print(f"最终回答: {response}")  # 打印最终答案
