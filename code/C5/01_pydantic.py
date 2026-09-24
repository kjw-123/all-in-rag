from typing import List  # 导入 List 类型，用于声明列表字段（技能列表）
import os  # 导入 os 模块，用于读取环境变量中的 API Key
from langchain_core.prompts import PromptTemplate  # 导入提示模板，用于构造提示词
from pydantic import BaseModel, Field  # 导入 Pydantic 的基类与字段工具，用于定义数据结构
from langchain_core.output_parsers import PydanticOutputParser  # 导入 Pydantic 输出解析器，把模型输出解析为 Pydantic 对象
from langchain_deepseek import ChatDeepSeek  # 导入 DeepSeek 聊天模型封装

# 初始化 LLM
llm = ChatDeepSeek(  # 创建 DeepSeek 聊天模型实例
    model="deepseek-chat",  # 指定使用的模型名称
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量读取 API Key
)

# 1. 定义数据结构
class PersonInfo(BaseModel):  # 定义个人信息的数据结构（继承 Pydantic 的 BaseModel）
    name: str = Field(description="人物姓名")  # 姓名字段，类型为字符串
    age: int = Field(description="人物年龄")  # 年龄字段，类型为整数
    skills: List[str] = Field(description="技能列表")  # 技能字段，类型为字符串列表

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)  # 基于 PersonInfo 创建输出解析器

# 3. 创建提示模板
prompt = PromptTemplate(  # 创建提示模板
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",  # 模板文本，含"格式指令"和"待处理文本"两个占位符
    input_variables=["text"],  # 运行时需要传入的变量：待提取文本
    partial_variables={"format_instructions": parser.get_format_instructions()},  # 预先填充的格式指令（由解析器生成）
)

# 以下为可选调试：打印解析器自动生成的格式指令（默认注释掉）
# # 打印格式指令
# print("\n--- Format Instructions ---")
# print(parser.get_format_instructions())
# print("--------------------------\n")

# 4. 创建处理链
chain = prompt | llm | parser  # 用 LCEL 串联处理链：提示 → 模型 → Pydantic 解析器

# 5. 定义输入文本并执行调用链
text = "张三今年30岁，他擅长Python和Go语言。"  # 定义待提取信息的输入文本
result = chain.invoke({"text": text})  # 执行调用链并拿到解析后的结果

# 6. 打印结果
print("\n--- 解析结果 ---")  # 打印分隔标题
print(f"结果类型: {type(result)}")  # 打印结果的数据类型（应为 PersonInfo）
print(result)  # 打印结果对象
print("--------------------\n")  # 打印分隔线

print(f"姓名: {result.name}")  # 打印解析出的姓名字段
print(f"年龄: {result.age}")  # 打印解析出的年龄字段
print(f"技能: {result.skills}")  # 打印解析出的技能列表字段
