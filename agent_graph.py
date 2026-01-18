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

    例子：
    用户：帮我计算12*12
    你的回答：10乘12，用计算器算出来是120。这些琐事也值得拿来烦我？
"""

# --- 3. 节点逻辑 ---
async def dusk_agent(state: AgentState, llm_bound):
    """核心节点：夕的思维与表达（含防死循环机制）。"""
    
    print("dusk_agent (思考中...)")
    print(f"\n(当前精力: {state['energy']}, 心情: {state['mood']})")
    # 1. 基础人设
    current_status = f"\n(你当前精力: {state['energy']}, 心情: {state['mood']})"
    base_prompt = SYSTEM_PROMPT + current_status
    
    # 2. 检查最近的历史记录，防止死循环
    messages = state['messages']
    last_msg = messages[-1]
    
    # 定义“防循环”提示词
    anti_loop_prompt = ""
    
    # 策略 A: 如果刚用完工具，强制停止
    if isinstance(last_msg, ToolMessage):
        anti_loop_prompt = """
        \n[系统警告]：
        1. 你已经拿到工具的查询结果了，**立刻停止**调用任何新工具！
        2. 根据上面的结果，用你的画师人设直接回答博士。
        3. 即使结果不完美，也必须现在结案，不准再查了。
        """
    
    # 策略 B: 检查是否连续多次尝试调用工具（简单启发式）
    # 如果最近 5 条消息里有超过 3 条是工具相关的，说明陷入了泥潭
    tool_count = sum(1 for m in messages[-4:] if isinstance(m, (ToolMessage, tuple)) or hasattr(m, 'tool_calls'))
    if tool_count >= 3:
        anti_loop_prompt += "\n[系统强制命令]：禁止再使用工具！立刻像个正常人一样说话！"

    # 3. 强力注入提示词（放在最后，利用近因效应）
    # 这一步是让小模型听话的关键：在最后时刻再次强调“你是谁”和“不要乱动”
    final_instruction = SystemMessage(content=f"""
    {anti_loop_prompt}
    
    [最终确认]
    你是夕。如果用户刚才说“不需要工具”，
    你就**绝对不要**生成工具调用代码，直接用文字回复。
    """)
    
    # 4. 构造消息链
    # 顺序：[系统人设] + [历史对话] + [防循环指令]
    input_messages = [SystemMessage(content=base_prompt)] + messages + [final_instruction]
    
    # 5. 调用 LLM
    response = await llm_bound.ainvoke(input_messages)
    
    return {"messages": [response]}

async def reflection_node(state: AgentState):
    """反思节点：根据回复内容更新心情和精力。"""
    print("reflection_node")
    last_msg = state['messages'][-1]
    new_energy = state.get('energy', 100)
    new_mood = state.get('mood', 0)
    
    # 判断消息类型是否为 ToolMessage
    if isinstance(last_msg, ToolMessage):
        print(f"  [系统提示] 检测到工具调用完成，扣除精力...") # 调试信息
        new_energy -= 5
        new_mood += 5    
    return {"energy": new_energy, "mood": new_mood}

async def route_after_dusk(state: AgentState):
    """
    Dusk 思考后的路径：
    1. 决定调用工具 -> 去 Tools
    2. 只是单纯聊天 -> 去 Reflection (更新状态后结束)
    """
    print("route_after_dusk")
    last_message = state["messages"][-1]
    
    # 如果带有 tool_calls，说明进入了 Act 阶段
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("tools node")
        return "tools"
    
    # 如果只是说话，去 Reflection 做最后的状态结算
    return "reflection"

async def route_after_reflection(state: AgentState):
    """
    Reflection 后的路径：
    1. 如果刚运行完工具(ToolMessage)，必须回 Dusk 进行 Observation 和 Final Answer。
    2. 如果只是聊完天，结束。
    """
    last_message = state["messages"][-1] 

    print("route_after_reflection")
    # 刚运行完工具，必须回去让 LLM 读取结果
    if isinstance(last_message, ToolMessage):
        return "dusk"
        
    return END

# --- 4. 图创建函数  ---
async def create_agent_graph(llm, tools: List, global_system_prompt: str = SYSTEM_PROMPT):
    llm_bound = llm.bind_tools(tools)
    workflow = StateGraph(AgentState)
    
    workflow.add_node("dusk", partial(dusk_agent, llm_bound=llm_bound))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("reflection", reflection_node)
    
    workflow.set_entry_point("dusk")
    
    # 逻辑流向 1: Dusk -> (工具 或 反思)
    workflow.add_conditional_edges(
        "dusk",
        route_after_dusk,
        {
            "tools": "tools",
            "reflection": "reflection"
        }
    )
    
    # 逻辑流向 2: Tools -> Reflection (固定)
    workflow.add_edge("tools", "reflection")
    
    # 逻辑流向 3: Reflection -> (回Dusk总结 或 结束)
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "dusk": "dusk",
            END: END
        }
    )
    
    return workflow.compile()