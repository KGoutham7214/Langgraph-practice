from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os
import certifi

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

llm = ChatOpenAI(model="gpt-5.4-mini")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": response}

graph = StateGraph(AgentState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

agent = graph.compile()
message1 = HumanMessage(content="Hello, how are you?")
turn1_state = agent.invoke({"messages": [message1]})
print(turn1_state)
