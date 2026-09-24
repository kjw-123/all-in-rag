from openai import OpenAI  # 导入 OpenAI 兼容客户端（DeepSeek 兼容 OpenAI 接口）
import os  # 导入 os 模块，用于读取环境变量中的 API Key

# 初始化 OpenAI 客户端
client = OpenAI(  # 创建 OpenAI 兼容客户端实例
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取 API Key
    base_url="https://api.deepseek.com",  # 指定服务地址为 DeepSeek 的兼容端点
)

# 定义一个函数，用于发送消息并获取模型的响应
def send_messages(messages, tools=None):  # 定义发送消息函数，tools 为可选工具列表
    response = client.chat.completions.create(  # 发起一次聊天补全请求
        model="deepseek-chat",  # 指定使用的模型
        messages=messages,  # 传入对话消息历史
        tools=tools,  # 传入可用工具列表
        tool_choice="auto",  # 让模型自主决定是否调用工具
    )
    return response.choices[0].message  # 返回模型返回的消息对象

# 1. 定义工具（函数）的 Schema
tools = [  # 工具定义列表
    {  # 一个工具的完整定义
        "type": "function",  # 类型为函数
        "function": {  # 函数的具体描述
            "name": "get_weather",  # 函数名称
            "description": "获取指定地点的天气信息",  # 函数功能描述
            "parameters": {  # 函数参数定义（JSON Schema）
                "type": "object",  # 参数整体为一个对象
                "properties": {  # 各参数属性
                    "location": {  # location 参数
                        "type": "string",  # 参数类型为字符串
                        "description": "城市和省份，例如：杭州市, 浙江省",  # 参数说明
                    }
                },
                "required": ["location"]  # 必填参数为 location
            },
        }
    },
]

# 1. 用户提问，模型决策调用工具
messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]  # 初始化对话历史，放入用户提问
print(f"User> {messages[0]['content']}\n")  # 打印用户问题
message = send_messages(messages, tools=tools)  # 第一次调用：把问题与工具一起发给模型

# 2. 执行工具，并将结果返回模型
if message.tool_calls:  # 判断模型是否发起了工具调用
    print("--- 模型发起了工具调用 ---")  # 打印提示
    tool_call = message.tool_calls[0]  # 取第一个工具调用
    function_info = tool_call.function  # 取出函数信息
    print(f"工具名称: {function_info.name}")  # 打印要调用的工具名
    print(f"工具参数: {function_info.arguments}")  # 打印工具调用参数

    # 将模型的回复（包含工具调用请求）添加到消息历史中
    messages.append(message)  # 把模型的这条回复追加到消息历史

    # 模拟执行工具
    tool_output = "24℃，晴朗"  # 模拟工具执行的结果
    print(f"--- 执行工具并返回结果 ---")  # 打印提示
    print(f"工具执行结果: {tool_output}\n")  # 打印工具执行结果

    # 将工具的执行结果作为一个新的消息添加到历史中
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_output})  # 追加工具结果消息（role=tool）

    # 3. 第二次调用：将工具结果返回给模型，获取最终回答
    print("--- 将工具结果返回给模型，获取最终答案 ---")  # 打印提示
    final_message = send_messages(messages, tools=tools)  # 第二次调用：把工具结果发回模型
    print(f"Model> {final_message.content}")  # 打印模型最终回答
else:  # 模型没有调用工具时
    # 如果模型没有调用工具，直接打印其回答
    print(f"Model> {message.content}")  # 直接打印模型的回答
