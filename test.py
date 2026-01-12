import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. 配置你的 API KEY
# 建议：由于之前的 Key 已泄露，请去 AI Studio 换一个新的填在这里
API_KEY = "your key"

# 2. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. 初始化模型 
# 注意：我改成了 1.5-flash，因为这是目前最稳、报错率最低的免费模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=API_KEY
)

# 4. 定义唯一节点
def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 5. 构建最简单的线性图：开始 -> 调用 -> 结束
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# 6. 运行测试
if __name__ == "__main__":
    input_state = {"messages": [HumanMessage(content="你好，请回复'测试通过'")]}
    try:
        result = app.invoke(input_state)
        print("\n--- 运行结果 ---")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"\n--- 依然报错 ---")
        print(e)