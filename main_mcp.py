import os
import sys
import asyncio
import re
from typing import List, Dict, Any, Optional
from pydantic import create_model, Field
from contextlib import AsyncExitStack

# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

# 导入图构建函数
from agent_graph import create_agent_graph

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------网络配置------
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict
}

async def _convert_session_to_tools(session: ClientSession, server_name: str = None) -> List[StructuredTool]:
    """
    处理单个 Session：获取工具列表并转换为 LangChain 工具。
    如果提供了 server_name，工具名称将加上前缀以避免冲突。
    """
    try:
        mcp_tools_list = await session.list_tools()
    except Exception as e:
        print(f"Error listing tools from {server_name}: {e}")
        return []

    langchain_tools = []

    for tool in mcp_tools_list.tools:
        # --- 1. 处理工具名称 (增加命名空间) ---
        # 如果有 server_name，工具名变为 "server_tool" (例如: filesystem_read_file)
        tool_name = f"{server_name}_{tool.name}" if server_name else tool.name
        
        # --- 2. 动态构建 Pydantic 模型 ---
        schema_fields = {}
        input_schema = tool.inputSchema
        
        if "properties" in input_schema:
            for prop_name, prop_def in input_schema["properties"].items():
                prop_type = TYPE_MAP.get(prop_def.get("type"), Any)
                prop_desc = prop_def.get("description", "")
                # 处理 required 字段
                is_required = prop_name in input_schema.get("required", [])
                if is_required:
                     schema_fields[prop_name] = (prop_type, Field(description=prop_desc))
                else:
                     schema_fields[prop_name] = (Optional[prop_type], Field(default=None, description=prop_desc))
        
        # 模型名称也需要唯一，避免 Pydantic 冲突
        ToolInputModel = create_model(f"{tool_name}Input", **schema_fields)

        # --- 3. 定义异步包装器 ---
        # 必须利用闭包或参数绑定确保 session 和 tool.name 是正确的
        def make_tool_wrapper(current_session, original_tool_name, debug_name):
            async def _tool_wrapper(**kwargs):
                clean_args = {k: v for k, v in kwargs.items() if v is not None}
                print(f"  [MCP工具::{debug_name}] 调用参数: {clean_args}")
                
                try:
                    result = await current_session.call_tool(original_tool_name, arguments=clean_args)
                    
                    if result.content and hasattr(result.content[0], "text"):
                        return result.content[0].text
                    return str(result)
                except Exception as e:
                    return f"Tool execution error: {str(e)}"
            return _tool_wrapper
        
        # 传入原始的 tool.name 给 mcp 调用，但 LangChain 层使用带前缀的 tool_name
        _tool_wrapper = make_tool_wrapper(session, tool.name, tool_name)
        
        # --- 4. 创建 StructuredTool ---
        lc_tool = StructuredTool.from_function(
            func=None,
            coroutine=_tool_wrapper,
            name=tool_name,  # LangChain 使用带前缀的唯一名称
            description=tool.description or "",
            args_schema=ToolInputModel
        )
        langchain_tools.append(lc_tool)
        
    return langchain_tools

async def load_mcp_tools(sessions: Dict[str, ClientSession]) -> List[StructuredTool]:
    """
    主入口函数：支持从多个 MCP Server 获取工具。
    
    Args:
        sessions: 一个字典，Key是服务器名称(用于前缀)，Value是 ClientSession 对象
                  例如: {"filesystem": session1, "search": session2}
    """
    tasks = []
    for name, session in sessions.items():
        # 为每个 session 创建一个任务
        tasks.append(_convert_session_to_tools(session, server_name=name))
    
    # 并发执行所有 list_tools 操作
    results = await asyncio.gather(*tasks)
    
    # 展平结果列表 (results 是 list of lists)
    all_tools = [tool for sublist in results for tool in sublist]
    
    print(f"成功加载了 {len(all_tools)} 个 MCP 工具，来自 {len(sessions)} 个服务器。")
    return all_tools

# --- 主程序 ---
async def main():
    # --- 1. 定义多服务器配置 ---
    # 这里可以配置多个不同的 MCP 服务器
    # Key 是服务器名称（将成为工具前缀），Value 是启动参数
    # 小模型可能存在注意力淹没问题，暂时只使用本地tools
    servers_config = {
        "local": StdioServerParameters(
            command=sys.executable,
            args=["mcp_tools.py"],
            env=os.environ.copy()
        ),

        # 示例：如果你有第二个服务器 (比如文件系统 server)
        # "fs": StdioServerParameters(
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-filesystem", "C:/test_folder"],
        #     env=os.environ.copy()
        # ),
    }

    print(f"--- 准备连接 {len(servers_config)} 个 MCP 服务器 ---")

    # --- 2. 使用 AsyncExitStack 动态管理多个连接 ---
    async with AsyncExitStack() as stack:
        sessions = {}  # 用于存储所有活跃的 session
        
        for server_name, params in servers_config.items():
            try:
                print(f"正在连接 [{server_name}] ...")
                # 启动进程 (相当于 async with stdio_client...)
                read, write = await stack.enter_async_context(stdio_client(params))
                
                # 建立会话 (相当于 async with ClientSession...)
                session = await stack.enter_async_context(ClientSession(read, write))
                
                # 初始化会话
                await session.initialize()
                
                # 存入字典
                sessions[server_name] = session
                print(f"✅ [{server_name}] 连接成功")
                
            except Exception as e:
                print(f"❌ [{server_name}] 连接失败: {e}")

        if not sessions:
            print("没有成功连接任何服务器，程序退出。")
            return

        # --- 3. 批量加载所有工具 ---
        # 此时 sessions 字典已经是 {"local": session_obj, "fs": session_obj}
        # load_mcp_tools 会并发加载它们
        tools = await load_mcp_tools(sessions)
        
        print(f"--- 所有工具加载完毕，共 {len(tools)} 个 ---")
        print(f"工具列表: {[t.name for t in tools]}")

        # --- 4. 初始化 LLM 和 Agent ---
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.7,
            num_ctx=2048,
            num_predict=512,
        )

        app = await create_agent_graph(llm, tools)

        # --- 5. 聊天循环 ---
        current_state = {
            "messages": [],
            "mood": 50,
            "energy": 100,
            "last_action": "",
            "is_done": False
        }

        print("\n--- 已进入画卷世界  ---")
        print("(输入 'exit' 退出)")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n[博士]: ")
            except EOFError:
                break
            
            if user_input.lower() in ["exit", "退出"]:
                break

            current_state["messages"].append(HumanMessage(content=user_input))
            
            result = await app.ainvoke(current_state)
            current_state = result
            
            # 输出清理逻辑
            last_msg = result['messages'][-1]
            raw_content = last_msg.content if isinstance(last_msg.content, str) else ""
            if isinstance(last_msg.content, list):
                raw_content = "".join([item['text'] for item in last_msg.content if 'text' in item])
            
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            if clean_content:
                print(f"\n[夕]: {clean_content}")
            elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print("(夕正在思考工具的使用...)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n连接已断开，退出画卷。")