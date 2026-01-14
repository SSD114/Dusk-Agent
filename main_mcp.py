import os
import sys
import asyncio
import re
from typing import Any
from pydantic import create_model, Field

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

# 导入图构建工厂 (核心修改点)
from agent_graph import create_agent_graph

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------网络配置------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# --- MCP 适配器逻辑 (保持不变，用于将 MCP 工具转换为 LangChain 工具) ---
async def load_mcp_tools(session: ClientSession):
    mcp_tools_list = await session.list_tools()
    langchain_tools = []

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    for tool in mcp_tools_list.tools:
        # 1. 动态构建 Pydantic 模型
        schema_fields = {}
        input_schema = tool.inputSchema
        
        if "properties" in input_schema:
            for prop_name, prop_def in input_schema["properties"].items():
                prop_type = type_map.get(prop_def.get("type"), Any)
                prop_desc = prop_def.get("description", "")
                schema_fields[prop_name] = (prop_type, Field(default=None, description=prop_desc))
        
        ToolInputModel = create_model(f"{tool.name}Input", **schema_fields)

        # 2. 定义异步包装器
        def make_tool_wrapper(tool_name: str):
            async def _tool_wrapper(**kwargs):
                clean_args = {k: v for k, v in kwargs.items() if v is not None}
                print(f"  [MCP工具] 调用 {tool_name} 参数: {clean_args}")
                
                result = await session.call_tool(tool_name, arguments=clean_args)
                
                if result.content and hasattr(result.content[0], "text"):
                    return result.content[0].text
                return str(result)
            return _tool_wrapper
        
        _tool_wrapper = make_tool_wrapper(tool.name)
        
        # 3. 创建 StructuredTool
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=tool.name,
            description=tool.description or "",
            args_schema=ToolInputModel
        )
        langchain_tools.append(lc_tool)
        
    return langchain_tools

# --- 主程序 ---
async def main():
    # 配置 MCP 服务端参数 (请确保 mcp_tools.py 在同一目录下)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_tools.py"], 
        env=os.environ.copy()
    )

    print("--- 正在连接 MCP 工具服务器 (mcp_tools.py) ---")
    
    # 建立 MCP 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. 初始化
            await session.initialize()
            
            # 2. 加载 MCP 工具
            tools = await load_mcp_tools(session)
            print(f"--- 已加载 MCP 工具: {[t.name for t in tools]} ---")
            
            # 3. 初始化 LLM
            llm = ChatOllama(
                model="qwen2.5:7b",
                temperature=0.7,
                num_ctx=2048,
                num_predict=512,
            )

            # 4. 创建图 (调用 agent_graph 中的工厂函数)
            # 这里复用了 agent_graph.py 中定义的通用逻辑
            app = create_agent_graph(llm, tools)

            # 5. 聊天循环
            current_state = {
                "messages": [],
                "mood": 50,
                "energy": 80,
                "last_action": "",
                "is_done": False
            }

            print("--- 已进入画卷世界 (MCP 增强版) ---")
            print("(输入 'exit' 退出)")

            while True:
                try:
                    user_input = await asyncio.to_thread(input, "\n[博士]: ")
                except EOFError:
                    break
                
                if user_input.lower() in ["exit", "退出"]:
                    break

                current_state["messages"].append(HumanMessage(content=user_input))

                # --- 统一使用 ainvoke (和 main_local 一致) ---
                result = await app.ainvoke(current_state)
                
                current_state = result
                last_msg = result['messages'][-1]
                
                # --- 输出清理逻辑 ---
                raw_content = ""
                if isinstance(last_msg.content, str):
                    raw_content = last_msg.content
                elif isinstance(last_msg.content, list):
                    raw_content = "".join([item['text'] for item in last_msg.content if isinstance(item, dict) and 'text' in item])
                
                clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
                clean_content = re.sub(r'```json.*?```', '', clean_content, flags=re.DOTALL)
                clean_content = re.sub(r'\{.*?"name".*?"args".*?\}', '', clean_content, flags=re.DOTALL)
                final_output = clean_content.strip()

                if final_output:
                    print(f"\n[夕]: {final_output}")
                    print(f"(状态: 心情 {result['mood']} | 精力 {result['energy']})")
                elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print("(夕正在思考工具的使用...)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n连接已断开，退出画卷。")