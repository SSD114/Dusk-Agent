# main_local.py
import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 导入核心图构建器
from agent_graph import create_agent_graph

# 网络设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# --- 1. 定义本地工具 ---
@tool
def get_weather_api(city: str):
    """
    通过调用外部天气 API 获取指定城市的实时天气数据。
    """
    # 这里模拟一个 API 调用过程
    # 实际开发中你可以使用 requests.get("https://api.weather.com/...")
    print(f"--- 正在调用天气 API 接口，查询城市: {city} ---")
    
    # 模拟返回的 JSON 数据
    mock_response = {
        "city": city,
        "temperature": "22°C",
        "description": "细雨绵绵，适合听低保真音乐",
        "status": "Success"
    }
    return f"城市：{mock_response['city']}, 温度：{mock_response['temperature']}, 状况：{mock_response['description']}"


@tool
def multiply(a: int, b: int) -> int:
    """乘法"""
    return a * b

# --- 2. 组装 ---
# A. 准备工具
my_tools = [get_weather_api, multiply]

# B. 准备 LLM (纯净版，未绑定工具)
raw_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

# C. 通过工厂创建图 (注入依赖)
app = create_agent_graph(raw_llm, my_tools)

    # D. 运行循环 (简单版)
def chat_with_dusk():
    # 初始化状态
    current_state = {
        "messages": [],
        "mood": 50,
        "energy": 80,
        "last_action": "",
        "is_done": False
    }

    print("--- 已进入画卷世界 (输入 'exit' 退出) ---")
    
    while True:
        user_input = input("\n[博士]: ")
        if user_input.lower() in ["exit", "退出"]:
            break
            
        current_state["messages"].append(HumanMessage(content=user_input))
        
        # 执行图逻辑
        result = app.invoke(current_state)
        
        # 更新状态，确保下一轮对话记得上一轮的精力值
        current_state = result

        # 提取最后一条 AI 回复
        last_msg = result['messages'][-1]
        raw_content = ""
        
        # 1. 统一提取文本内容
        if isinstance(last_msg.content, str):
            raw_content = last_msg.content
        elif isinstance(last_msg.content, list):
            raw_content = "".join([item['text'] for item in last_msg.content if isinstance(item, dict) and 'text' in item])
        
        # 2. 移除 <think> 标签及其内部的所有思考逻辑
        import re
        # 使用 DOTALL 模式确保能匹配跨行的 think 标签
        clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        
        # 3. 移除模型可能误打印的工具调用原始 JSON 块 (有些模型会把 tool_use 打印在正文)
        clean_content = re.sub(r'```json.*?```', '', clean_content, flags=re.DOTALL)
        clean_content = re.sub(r'\{.*?"name".*?"args".*?\}', '', clean_content, flags=re.DOTALL)
        
        # 4. 去除多余的空行和首尾空格
        final_output = clean_content.strip()

        # 5. 兜底逻辑：如果模型正在调用工具（没有正文内容），夕应该保持高冷
        if not final_output and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            # 这种情况说明模型只发出了指令，还没到吐槽环节
            # 但因为你的逻辑是 tools -> reflection -> dusk，这里通常不会触发
            pass 
        elif final_output:
            print(f"\n[夕]: {final_output}")
            print(f"(状态: 心情 {result['mood']} | 精力 {result['energy']})")

if __name__ == "__main__":
    chat_with_dusk()