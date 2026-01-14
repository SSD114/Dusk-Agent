# main_local.py
import os
import asyncio
import sys
import shutil
from functools import partial
from typing import Annotated, TypedDict, Any
from pydantic import create_model, Field

# LangChain / LangGraph imports
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------网络配置------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# 1. 定义角色设定 (保持不变)
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

# 2. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mood: int           
    energy: int         
    is_done: bool       

# 3. 核心节点逻辑 (Dusk Agent)
# 注意：现在需要在 global scope 或者传入 app 之前绑定工具，
# 所以我们将 llm 的初始化推迟到 main 函数中工具加载完成后。

async def dusk_agent(state: AgentState, llm_bound):
    """核心节点：夕的思维与表达。"""
    # 模拟思考延迟
    await asyncio.sleep(0.5) 

    current_status = f"\n(你当前精力: {state['energy']}, 心情: {state['mood']})"
    if state['energy'] < 5:
        current_status += "。你现在极度疲惫，只想睡觉，拒绝任何劳动。"
    
    system_prompt = SystemMessage(content=SYSTEM_PROMPT + current_status)
    
    # 异步调用 LLM
    response = await llm_bound.ainvoke([system_prompt] + state['messages'])
    return {"messages": [response]}

def reflection_node(state: AgentState):
    """反思节点"""
    last_msg = state['messages'][-1]
    new_energy = state.get('energy', 100)
    new_mood = state.get('mood', 0)
    
    # 检测是否使用了工具 (ToolMessage 前一个是 AIMessage 且带有 tool_calls)
    # 简化逻辑：如果在工具节点之后，最后一条消息应该是 ToolMessage
    if isinstance(last_msg, ToolMessage):
        # 只要最近有工具执行，就扣除精力
        new_energy -= 5
        new_mood += 5
    
    return {"energy": new_energy, "mood": new_mood}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 检查 LLM 是否发出了工具调用请求
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if state['energy'] < 5:
            return END # 精力不足，拒绝干活
        return "tools"
    return END

# --- MCP 适配器逻辑 ---
async def load_mcp_tools(session: ClientSession):
    mcp_tools_list = await session.list_tools()
    langchain_tools = []

    # 类型映射表：将 JSON Schema 类型映射为 Python 类型
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    for tool in mcp_tools_list.tools:
        # 1. 动态构建 Pydantic 模型 (参数定义)
        # 这步至关重要，它告诉 LLM 这个工具具体需要什么参数 (如 city, a, b)
        schema_fields = {}
        input_schema = tool.inputSchema
        
        if "properties" in input_schema:
            for prop_name, prop_def in input_schema["properties"].items():
                # 获取参数类型，默认为 Any
                prop_type = type_map.get(prop_def.get("type"), Any)
                prop_desc = prop_def.get("description", "")
                
                # 创建 Pydantic 字段定义
                # 注意：为了容错，这里设置 default=None，允许参数选填
                # 严格模式下应该检查 "required" 列表
                schema_fields[prop_name] = (prop_type, Field(default=None, description=prop_desc))
        
        # 动态创建参数模型类
        ToolInputModel = create_model(f"{tool.name}Input", **schema_fields)

        # 2. 定义异步包装器
        # 多嵌套一层定义，修复绑定错误问题
        def make_tool_wrapper(tool_name: str):
            async def _tool_wrapper(**kwargs):
                clean_args = {k: v for k, v in kwargs.items() if v is not None}
                print(f"  [Debug] LLM 调用工具 {tool_name} 参数: {clean_args}")
                
                result = await session.call_tool(tool_name, arguments=clean_args)
                
                if result.content and hasattr(result.content[0], "text"):
                    return result.content[0].text
                return str(result)
            return _tool_wrapper
        
        _tool_wrapper = make_tool_wrapper(tool.name)
        # 3. 创建带有明确 Schema 的 LangChain 工具
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=tool.name,
            description=tool.description or "",
            args_schema=ToolInputModel  # 注入参数模型
        )
        langchain_tools.append(lc_tool)
        
    return langchain_tools

# --- 主程序 ---
async def main():
    # 配置 MCP 服务端参数 (指向 mcp_tools.py)
    # 确保使用当前环境的 python 解释器
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_tools.py"], 
        env=os.environ.copy() # 继承环境变量
    )

    print("--- 正在连接 MCP 工具服务器 (mcp_tools.py) ---")
    
    # 建立 MCP 连接上下文
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化阶段
            await session.initialize()
            
            # 2. 加载并转换工具
            tools = await load_mcp_tools(session)
            print(f"--- 已加载 MCP 工具: {[t.name for t in tools]} ---")
            
            # 3. 初始化 LLM 并绑定工具
            llm = ChatOllama(
                model="qwen2.5:7b",
                temperature=0.7,
                num_ctx=2048,
                num_predict=512,
            ).bind_tools(tools)

            # 4. 构建图
            workflow = StateGraph(AgentState)
            
            # 使用 functools.partial 或 lambda 将 bound llm 注入节点
            workflow.add_node("dusk", partial(dusk_agent, llm_bound=llm))
            workflow.add_node("tools", ToolNode(tools)) # ToolNode 支持异步调用
            workflow.add_node("reflection", reflection_node)

            workflow.set_entry_point("dusk")
            workflow.add_conditional_edges("dusk", should_continue, {"tools": "tools", END: END})
            workflow.add_edge("tools", "reflection")
            workflow.add_edge("reflection", "dusk")

            app = workflow.compile()

            # 5. 聊天循环
            current_state = {
                "messages": [],
                "mood": 50,
                "energy": 80,
                "is_done": False
            }

            print("--- 已进入画卷世界 (输入 'exit' 退出) ---")
            print("(提示：因为使用了 MCP 异步通信，响应可能比纯本地稍慢)")

            while True:
                # 异步环境下的非阻塞输入比较麻烦，这里使用简单的 input() 会阻塞
                # 为了保持代码简洁，暂时允许在 input 处阻塞主线程
                try:
                    user_input = await asyncio.to_thread(input, "\n[博士]: ")
                except EOFError:
                    break
                
                if user_input.lower() in ["exit", "退出"]:
                    break

                current_state["messages"].append(HumanMessage(content=user_input))

                # 执行工作流
                final_state = None
                async for event in app.astream(current_state, stream_mode="values"):
                    # 获取每一步的更新，以便保存状态
                    final_state = event
                
                if final_state:
                    current_state = final_state
                    last_msg = current_state['messages'][-1]
                    
                    # 输出处理 (复用之前的正则清理逻辑)
                    if isinstance(last_msg, ToolMessage):
                         # 如果最后一步是工具结果，通常不会直接打印给用户，
                         # 因为图的流向是 Tools -> Reflection -> Dusk
                         # 所以最后一条消息通常是 Dusk 的 AIMessage
                         pass
                    elif isinstance(last_msg.content, str):
                        import re
                        raw = last_msg.content
                        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
                        clean = re.sub(r'```json.*?```', '', clean, flags=re.DOTALL)
                        clean = clean.strip()
                        if clean:
                            print(f"\n[夕]: {clean}")
                            print(f"(状态: 心情 {current_state['mood']} | 精力 {current_state['energy']})")

if __name__ == "__main__":
    # 运行异步主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n连接已断开，退出画卷。")