# agent_core.py
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from functools import partial

# --- 1. 状态定义 ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mood: int
    energy: int
    last_action: str
    is_done: bool

# --- 2. 角色人设 (可复用) ---
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
    - 如果用户让你做计算或查天气，请使用工具（tools）。
    - 如果你调用了工具，请明确指出。
    - 注意：若需要调用工具，严禁进行长篇思考，请立即输出工具调用指令。严禁在输出中包含复杂的逻辑推导

    例子：
    用户：帮我计算12*12
    你的回答：10乘12，用计算器算出来是120。这些琐事也值得拿来烦我？
"""

# --- 3. 节点逻辑 ---
async def dusk_agent(state: AgentState, llm_bound):
    """核心节点：夕的思维与表达。"""
    import time
    time.sleep(1) # 给显卡一点缓冲时间

    current_status = f"\n(你当前精力: {state['energy']}, 心情: {state['mood']})"
    if state['energy'] < 5:
        current_status += "。你现在极度疲惫，只想睡觉，拒绝任何劳动。"
    # 注入系统人设
    system_prompt = SystemMessage(content=SYSTEM_PROMPT+current_status)
    
    # 获取 LLM 回复
    response = llm_bound.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

async def reflection_node(state: AgentState):
    """反思节点：根据回复内容更新心情和精力。"""
    last_msg = state['messages'][-1]
    new_energy = state.get('energy', 100)
    new_mood = state.get('mood', 0)
    
    # 判断消息类型是否为 ToolMessage
    if isinstance(last_msg, ToolMessage):
        print(f"  [系统提示] 检测到工具调用完成，扣除精力...") # 调试信息
        new_energy -= 5
        new_mood += 5    
    return {"energy": new_energy, "mood": new_mood}

# 决策节点
async def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 1. 检查是否有工具调用请求
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 自主意识：如果精力太低，强行拒绝调用工具
        if state['energy'] < 5:
            return "END"
        return "tools"
    
    return "END"

# --- 4. 图创建函数  ---
def create_agent_graph(llm, tools: List):
    """
    接收 LLM 和 Tools，组装并返回编译好的 Graph。
    这实现了逻辑与配置的解耦。
    """
    # 1. 在这里进行绑定，确保 LLM 知道工具有哪些
    llm_bound = llm.bind_tools(tools)
    
    # 2. 构建图
    workflow = StateGraph(AgentState)
    
    # 注入绑定了工具的 LLM
    workflow.add_node("dusk", partial(dusk_agent, llm_bound=llm_bound))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("reflection", reflection_node)
    
    workflow.set_entry_point("dusk")
    
    workflow.add_conditional_edges(
        "dusk",
        should_continue,
        {
            "tools": "tools", 
            "END": END
        }
    )
    
    workflow.add_edge("tools", "reflection")
    workflow.add_edge("reflection", "dusk")
    
    return workflow.compile()