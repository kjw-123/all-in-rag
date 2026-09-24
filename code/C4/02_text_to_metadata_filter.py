import os  # 导入标准库 os，用于读取环境变量（如 API Key）
from langchain_deepseek import ChatDeepSeek  # 导入 DeepSeek 聊天模型封装类，作为后续自查询的 LLM
from langchain_community.document_loaders import BiliBiliLoader  # 导入 B 站视频加载器，用于抓取视频信息
from langchain.chains.query_constructor.base import AttributeInfo  # 导入元数据字段描述类，用于声明可过滤字段
from langchain.retrievers.self_query.base import SelfQueryRetriever  # 导入自查询检索器，将自然语言转成带过滤条件的查询
from langchain_community.vectorstores import Chroma  # 导入 Chroma 向量数据库，用于存储文档向量
from langchain_huggingface import HuggingFaceEmbeddings  # 导入 HuggingFace 嵌入模型封装，用于文本向量化
import logging  # 导入日志模块，用于控制运行日志输出

logging.basicConfig(level=logging.INFO)  # 配置全局日志级别为 INFO，便于观察加载与检索过程

# 1. 初始化视频数据
video_urls = [  # 定义待加载的 B 站视频链接列表
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",  # 第 1 个视频地址
    "https://www.bilibili.com/video/BV1ug4y157xA",  # 第 2 个视频地址
    "https://www.bilibili.com/video/BV1yh411V7ge",  # 第 3 个视频地址
]  # 列表定义结束

bili = []  # 初始化空列表，用于存放加载并整理后的视频文档
try:  # 用 try 包裹加载过程，避免单个请求失败导致整个脚本中断
    loader = BiliBiliLoader(video_urls=video_urls)  # 创建 B 站加载器实例，指定视频地址列表
    docs = loader.load()  # 执行加载，返回 Document 对象列表（每个视频对应一个文档）
    
    for doc in docs:  # 遍历加载得到的每个文档
        original = doc.metadata  # 取出原始元数据字典，便于后续提取字段
        
        # 提取基本元数据字段
        metadata = {  # 构造精简后的元数据字典，仅保留检索过滤所需的字段
            'title': original.get('title', '未知标题'),  # 视频标题，缺失时回退为“未知标题”
            'author': original.get('owner', {}).get('name', '未知作者'),  # 从 owner.name 取作者名，缺失时回退
            'source': original.get('bvid', '未知ID'),  # 取视频 BVID 作为来源标识，缺失时回退
            'view_count': original.get('stat', {}).get('view', 0),  # 从 stat.view 取观看次数，缺失时默认 0
            'length': original.get('duration', 0),  # 取视频时长（秒），缺失时默认 0
        }  # 元数据字典构造结束
        
        doc.metadata = metadata  # 用精简后的元数据替换文档原有元数据，屏蔽无关字段
        bili.append(doc)  # 将整理好的文档加入结果列表
        
except Exception as e:  # 捕获加载过程中的任意异常
    print(f"加载BiliBili视频失败: {str(e)}")  # 打印错误信息，方便排查失败原因

if not bili:  # 判断是否一个视频都没加载成功
    print("没有成功加载任何视频，程序退出")  # 提示无可用数据
    exit()  # 直接退出程序，避免后续无效操作

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")  # 加载中文小模型 bge-small-zh，用于生成文本向量
vectorstore = Chroma.from_documents(bili, embed_model)  # 将视频文档向量化并写入 Chroma 向量库

# 3. 配置元数据字段信息
metadata_field_info = [  # 声明可被自查询解析的元数据字段（供 LLM 理解字段含义与类型）
    AttributeInfo(  # 描述 title 字段
        name="title",  # 字段名：标题
        description="视频标题（字符串）",  # 字段语义说明，帮助 LLM 生成过滤条件
        type="string",  # 字段类型为字符串
    ),  # title 字段描述结束
    AttributeInfo(  # 描述 author 字段
        name="author",  # 字段名：作者
        description="视频作者（字符串）",  # 字段语义说明
        type="string",  # 字段类型为字符串
    ),  # author 字段描述结束
    AttributeInfo(  # 描述 view_count 字段
        name="view_count",  # 字段名：观看次数
        description="视频观看次数（整数）",  # 字段语义说明
        type="integer",  # 字段类型为整数，支持大小比较
    ),  # view_count 字段描述结束
    AttributeInfo(  # 描述 length 字段
        name="length",  # 字段名：时长
        description="视频长度（整数）",  # 字段语义说明
        type="integer"  # 字段类型为整数，支持比较与排序
    )  # length 字段描述结束
]  # 元数据字段列表结束

# 4. 创建自查询检索器
llm = ChatDeepSeek(  # 初始化 DeepSeek 聊天模型，用于解析自然语言查询
    model="deepseek-chat",  # 指定使用的模型名称
    temperature=0,  # 温度设为 0，保证查询解析结果稳定确定
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量读取 API Key，避免硬编码泄露
    )  # LLM 初始化结束

retriever = SelfQueryRetriever.from_llm(  # 通过 LLM 构建自查询检索器（自然语言 → 结构化过滤查询）
    llm=llm,  # 传入用于解析查询的语言模型
    vectorstore=vectorstore,  # 传入待检索的向量存储
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",  # 文档内容描述，辅助 LLM 理解语料
    metadata_field_info=metadata_field_info,  # 传入可过滤字段定义
    enable_limit=True,  # 允许 LLM 解析“数量限制/取前 N 个”这类请求
    verbose=True  # 打开详细日志，打印生成的查询语句便于调试
)  # 检索器构建结束

# 5. 执行查询示例
queries = [  # 定义待测试的自然语言查询列表
    "时间最短的视频",  # 示例一：按时长排序取最短
    "时长大于600秒的视频"  # 示例二：按时长做数值过滤
]  # 查询列表结束

for query in queries:  # 遍历每条查询
    print(f"\n--- 查询: '{query}' ---")  # 打印当前查询分隔标题
    results = retriever.invoke(query)  # 执行检索，返回匹配的文档列表
    if results:  # 判断是否有检索结果
        for doc in results:  # 遍历每条结果文档
            title = doc.metadata.get('title', '未知标题')  # 取出标题
            author = doc.metadata.get('author', '未知作者')  # 取出作者
            view_count = doc.metadata.get('view_count', '未知')  # 取出观看次数
            length = doc.metadata.get('length', '未知')  # 取出时长
            print(f"标题: {title}")  # 打印标题
            print(f"作者: {author}")  # 打印作者
            print(f"观看次数: {view_count}")  # 打印观看次数
            print(f"时长: {length}秒")  # 打印时长（单位：秒）
            print("="*50)  # 打印分隔线，区隔不同结果
    else:  # 没有检索结果时
        print("未找到匹配的视频")  # 提示未找到匹配项
