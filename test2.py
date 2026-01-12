import os
import re
import time
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

# 1. 初始化本地模型 (针对 4060 8G 显存优化)
llm = ChatOllama(
    model="qwen2.5:7b", 
    temperature=0,        
    num_ctx=2048,        # 限制上下文，防止 502 报错
    num_predict=512,     # 限制单次输出长度
)

# 2. 定义状态结构
class AgentState(TypedDict):
    # add_messages 会自动处理消息列表的叠加
    messages: Annotated[list, add_messages]

# 3. 定义节点逻辑
def dusk_agent(state: AgentState):
    """核心节点：调用模型并获取回复。"""
    print("--- 夕正在构思... ---")
    time.sleep(1) # 给显卡一点缓冲时间
    
    # 执行调用
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

# 4. 构建 LangGraph 图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", dusk_agent)

# 设置流程：开始 -> agent -> 结束
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# 编译应用
app = workflow.compile()

# 5. 运行测试函数
def run_test():
    # 模拟初始状态
    inputs = {
        "messages": [
            SystemMessage(content="你现在是干员夕，说话要毒舌、简短。"),
            HumanMessage(content="喂，你在干什么？")
        ]
    }

    try:
        # 执行图逻辑
        result = app.invoke(inputs)
        
        # 获取最后一条回复
        last_msg = result['messages'][-1].content
        
        # 过滤 DeepSeek-R1 的 <think> 标签
        clean_content = re.sub(r'<think>.*?</think>', '', last_msg, flags=re.DOTALL).strip()
        
        print(f"\n[夕]: {clean_content}")
        
    except Exception as e:
        print(f"\n[错误]: {e}")
        print("提示: 如果报 502，请尝试彻底重启 Ollama 服务后再运行。")

if __name__ == "__main__":
    run_test()
