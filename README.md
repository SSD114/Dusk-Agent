# Dusk-Agent

# 🖌️ Dusk (夕) - LangGraph Agent with MCP Support

这是一个基于 LangGraph 和 Ollama (Qwen2.5) 构建的角色扮演 AI Agent 项目。该项目复刻了《明日方舟》中干员 夕 (Dusk) 的人格，具备状态管理（心情/精力）、工具调用能力，并支持 MCP (Model Context Protocol) 协议来扩展外部工具。

## ✨ 核心特性
- 沉浸式角色扮演 (RP):

- 深度定制的 System Prompt，还原夕“高冷、毒舌、慵懒”的性格。

- 智能状态管理 (State Machine):

基于 LangGraph 构建的状态机。

精力/心情系统: 每次调用工具会消耗“精力”并提升“心情”（反映她厌恶麻烦但不得不工作的心理）。

反思节点 (Reflection): 在行动结束后更新自身状态。

死循环熔断: 内置防循环逻辑，防止 Agent 在工具调用中卡死。

- 双模式运行:

Local Mode: 使用内置的 Python 函数（天气、计算器）。

MCP Mode: 完整的 Model Context Protocol 客户端实现，支持动态加载和转换 MCP Server 的工具。

# 📂 项目结构
```Plaintext
.
├── agent_graph.py      # 核心逻辑：定义 StateGraph、节点行为、人设提示词、反思逻辑
├── main_local.py       # 入口1：本地简易模式，包含内置的 mock 工具
├── main_mcp.py         # 入口2：MCP 进阶模式，支持连接外部 MCP Server
└── mcp_tools.py        # (需自备) MCP Server 的具体实现脚本
```
# 🛠️ 架构说明
本项目采用了图（Graph）结构来控制 Agent 的思考流：

- Dusk Node (思考核心):

接收历史消息，结合当前心情/精力生成回复。

决定是直接回复（闲聊）还是调用工具（处理麻烦事）。

- Tools Node (工具执行):

执行具体的函数调用（本地函数或 MCP 协议转换后的工具）。

- Reflection Node (状态结算):

工具执行完毕后，扣除精力值，更新心情值，强制 Agent 再次进行观察并输出最终结果。

# 🚀 快速开始
1. 前置要求
Python 3.10+

Ollama: 需安装并运行 Ollama 服务。

模型下载：ollama pull qwen2.5:7b (代码默认配置)

2. 安装依赖
请确保安装了以下核心库：

```Bash
pip install langchain-ollama langgraph langchain-core mcp pydantic
```
3. 运行模式

- 模式 A: 本地简易模式 (main_local.py)
适合测试 Agent 的人设和基础逻辑，内置了模拟的“天气查询”和“乘法计算”工具。

- 模式 B: MCP 协议模式 (main_mcp.py)
适合生产环境或需连接复杂工具集。该模式会启动一个 MCP Client，连接到配置文件中指定的 Server。

注意: 默认配置依赖一个名为 mcp_tools.py 的脚本作为 Server。你需要确保该文件存在，或修改 servers_config 连接到其他 MCP Server（如 Filesystem, Google Drive 等）。

⚙️ 配置说明
1. 修改模型
在 main_local.py 或 main_mcp.py 中，你可以修改 llm 初始化部分来更换模型：

```Python

llm = ChatOllama(
    model="qwen2.5:7b", # 可替换为 "llama3", "mistral" 等
    temperature=0.7,
    # ...
)
```
2. MCP 服务器配置
在 main_mcp.py 中，你可以添加多个 MCP 服务器：

```Python

servers_config = {
    # 本地 Python 脚本 Server
    "local": StdioServerParameters(
        command=sys.executable,
        args=["mcp_tools.py"],
        env=os.environ.copy()
    ),
    # 示例：连接 Node.js 文件系统 Server
    # "filesystem": StdioServerParameters(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-filesystem", "./data"],
    #     env=os.environ.copy()
    # ),
}
```
## ⚠️ 常见问题
1. 连接报错 mcp_tools.py not found:

main_mcp.py 默认试图启动同目录下的 mcp_tools.py。如果你没有这个文件，请创建一个标准的 MCP Server 脚本，或者在代码中注释掉 local 配置。

2. 回复包含 think 标签:

代码中已经内置了正则清洗逻辑 **(re.sub(r'<think>.*?</think>'))**。如果仍然出现，请检查你的 Ollama 模型版本或调整正则表达式。

3. 网络代理问题:

代码默认关闭了系统代理 (os.environ.pop("HTTP_PROXY")) 以确保连接本地 Ollama 顺畅。如果你的 Ollama 在远程服务器，请注释掉相关代码。
