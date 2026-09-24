import os  # 导入标准库 os，用于读取环境变量中的 API Key
from langchain_core.prompts import ChatPromptTemplate  # 导入聊天提示模板，用于构造对话式提示
from langchain_core.output_parsers import StrOutputParser  # 导入字符串输出解析器，把模型输出转为纯字符串
from langchain_deepseek import ChatDeepSeek  # 导入 DeepSeek 聊天模型封装，作为底层 LLM
from langchain_core.runnables import RunnableBranch  # 导入条件分支组件，实现根据条件路由到不同链

llm = ChatDeepSeek(  # 初始化 DeepSeek 聊天模型
    model="deepseek-chat",  # 指定使用的模型名称
    temperature=0,  # 温度设为 0，保证分类与回答结果稳定确定
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量读取 API Key，避免硬编码泄露
    )  # LLM 初始化结束

# 1. 设置不同菜系的处理链
sichuan_prompt = ChatPromptTemplate.from_template(  # 定义川菜大厨的提示模板
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"  # 模板文本，{question} 为占位变量
)  # 川菜提示模板结束
sichuan_chain = sichuan_prompt | llm | StrOutputParser()  # 用 LCEL 组装川菜链：提示 → 模型 → 字符串解析

cantonese_prompt = ChatPromptTemplate.from_template(  # 定义粤菜大厨的提示模板
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"  # 模板文本，{question} 为占位变量
)  # 粤菜提示模板结束
cantonese_chain = cantonese_prompt | llm | StrOutputParser()  # 用 LCEL 组装粤菜链：提示 → 模型 → 字符串解析

# 定义备用通用链
general_prompt = ChatPromptTemplate.from_template(  # 定义通用美食助手的提示模板（兜底）
    "你是一个美食助手。请回答关于「{question}」的问题。"  # 通用模板文本，{question} 为占位变量
)  # 通用提示模板结束
general_chain = general_prompt | llm | StrOutputParser()  # 用 LCEL 组装通用链：提示 → 模型 → 字符串解析


# 2. 创建路由链
classifier_prompt = ChatPromptTemplate.from_template(  # 定义分类器的提示模板，用于判断问题属于哪个菜系
    # 提示文本（要求：分类为 川菜/粤菜/其他，只返回一个分类词，不解释理由）
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)  # 分类器提示模板结束
classifier_chain = classifier_prompt | llm | StrOutputParser()  # 组装分类链：提示 → 模型 → 字符串解析

# 定义路由分支
router_branch = RunnableBranch(  # 创建条件分支，按顺序判断条件并选择对应链
    (lambda x: "川菜" in x["topic"], sichuan_chain),  # 条件1：topic 含「川菜」则走川菜链
    (lambda x: "粤菜" in x["topic"], cantonese_chain),  # 条件2：topic 含「粤菜」则走粤菜链
    general_chain  # 默认选项：以上都不满足时走通用链
)  # 路由分支定义结束

# 组合成完整路由链
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch  # 先用分类链生成 topic，再把 topic+question 交给分支路由
print("完整的路由链创建成功。\n")  # 打印提示，确认链已构建完成


# 3. 运行演示查询
demo_questions = [  # 定义演示问题列表
    {"question": "麻婆豆腐怎么做？"},      # 应被分类为川菜，路由到川菜链
    {"question": "白切鸡的正宗做法是什么？"}, # 应被分类为粤菜，路由到粤菜链
    {"question": "番茄炒蛋需要放糖吗？"}      # 应被分类为其他，路由到通用链
]  # 演示问题列表结束

for i, item in enumerate(demo_questions, 1):  # 遍历每个演示问题，序号从 1 开始
    question = item["question"]  # 取出当前问题的文本
    print(f"\n--- 问题 {i}: {question} ---")  # 打印当前问题的分隔标题
    
    try:  # 用 try 包裹执行过程，避免单个问题报错中断整个循环
        # 获取路由决策
        topic = classifier_chain.invoke({"question": question})  # 调用分类链，得到分类结果字符串
        print(f"路由决策: {topic}")  # 打印分类（路由）决策

        # 执行完整链
        result = full_router_chain.invoke(item)  # 执行完整链，内部先分类再路由到对应链求解
        print(f"回答: {result}")  # 打印最终回答
    except Exception as e:  # 捕获执行过程中的任意异常
        print(f"执行错误: {e}")  # 打印错误信息，便于排查

