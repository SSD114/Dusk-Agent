from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="qwen2.5:1.5b",
)

resp = llm.invoke([
    HumanMessage(content="你好")
])

print(resp.content)