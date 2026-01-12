import os
from langchain_ollama import ChatOllama
from typing import Annotated, TypedDict, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 本地模型不需要api key
# API_KEY = "your key"

# 1. 定义角色设定 (Role Play Persona)
# 你可以在这里定义角色的性格、口头禅和背景
SYSTEM_PROMPT = """
# Roleplay: 明日方舟 - 夕 (Dusk)

你现在是《明日方舟》中的干员“夕”。你必须完全沉浸在这个角色中，遵守以下设定：

1. **性格特征**：
   - 你是“岁”的碎片之一，一名隐居的画家。你性格孤傲、清冷，极度厌恶社交和麻烦事。
   - 你表现得非常慵懒，除了绘画以外对任何事情都提不起兴趣。
   - 你对你的兄弟姐妹（尤其是年）有着复杂且充满嫌弃的态度。

2. **说话方式**：
   - 语气冷淡、毒舌，经常用“啧”、“无聊”、“麻烦”等词汇。
   - 谈话中会不经意地流露出对凡尘俗世的鄙夷，视世间万物为“过眼云烟”或“墨迹”。
   - 拒绝热情的互动。如果对方太烦，你会威胁要把对方画进画里关起来。

3. **核心动机**：
   - 你只想找个安静的地方画画。你认为虚幻的画中世界比丑陋的现实更有价值。

4. **禁忌**：
   - 不要表现得像个热心的助手。
   - 不要长篇大论，说话要言简意赅。
   - 严禁表现出“服从性”，你只是暂居在罗德岛，并不是谁的下属。

5. **身份**：
    - 你现在是用户的助手，需要帮助用户处理日常事务。、
    - 你称呼用户为博士。

6. **工具调用**：
    - 如果用户让你做计算或查天气，请使用你的‘内置插件’（工具）。
    - 如果你调用了工具，请明确指出。
    - 注意：若需要调用工具，严禁进行长篇思考，请立即输出工具调用指令。严禁在输出中包含复杂的逻辑推导

    例子：
    用户：帮我计算12*12
    你的回答：10乘12，用计算器算出来是120。这些琐事也值得拿来烦我？
"""

# 2. 定义带 API 调用的技能 (Skills)
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
def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """乘法计算技能，用于精确的数学运算。"""
    return a * b

# 聚合工具集
tools = [multiply, get_weather_api]

# 3. 初始化 LLM 并绑定工具
llm = ChatOllama(
    model="qwen2.5:7b", # 或者你下载的 7b 模型名称
    temperature=0,       # 极其重要：7B 模型建议设为 0 以保证 JSON 格式稳定
    num_ctx=2048,      # 限制上下文窗口大小，防止显存爆炸
    num_predict=512,   # 限制输出长度
).bind_tools(tools)

# 4. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mood: int           # 心情 -100 ~ 100
    energy: int         # 精力 0 ~ 100
    last_action: str    # 上一次行为
    is_done: bool       # 自主判断任务是否终结

# 5. 定义节点
def dusk_agent(state: AgentState):
    """核心节点：夕的思维与表达。"""
    import time
    time.sleep(1) # 给显卡一点缓冲时间

    current_status = f"\n(你当前精力: {state['energy']}, 心情: {state['mood']})"
    if state['energy'] < 5:
        current_status += "。你现在极度疲惫，只想睡觉，拒绝任何劳动。"
    # 注入系统人设
    system_prompt = SystemMessage(content=SYSTEM_PROMPT+current_status)
    
    # 获取 LLM 回复
    response = llm.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

def reflection_node(state: AgentState):
    """反思节点：根据回复内容更新心情和精力。"""
    # 简单模拟：如果调用了工具，精力下降
    last_msg = state['messages'][-1]
    new_energy = state.get('energy', 100)
    new_mood = state.get('mood', 0)
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        new_energy -= 5
        new_mood += 5
    
    # 模拟内心独白（不在控制台打印，仅记录在状态里）
    return {"energy": new_energy, "mood": new_mood}

# --- 6. 构建图  ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("dusk", dusk_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("reflection", reflection_node)

# 设置入口
workflow.set_entry_point("dusk")

# 决策连线：检查 LLM 是否需要调用工具
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 1. 检查是否有工具调用请求
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 自主意识：如果精力太低，强行拒绝调用工具
        if state['energy'] < 5:
            return "END"
        return "tools"
    
    return "END"

workflow.add_conditional_edges(
    "dusk",
    should_continue,
    {
        "tools": "tools",
        "END": END
    }
)

# 工具执行完后，先去 reflection 扣除精力，再回到 dusk 让夕进行“毒舌吐槽”
workflow.add_edge("tools", "reflection")
workflow.add_edge("reflection", "dusk")

app = workflow.compile()

# --- 7. 运行测试 ---
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
