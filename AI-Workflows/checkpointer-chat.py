from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import os 
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

llm = ChatOpenAI(model="gpt-5.4-mini")

def chatbot(state: MessagesState):
    response = llm.invoke(state['messages'])

    return{
        "messages": [response]
    }

builder = StateGraph(MessagesState)
builder.add_node(chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()

# graph = builder.compile()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable":{
        "thread_id" : "chat_session_1"
    }
}

message_1 = "Hi My name is Goutham Kundena, I am an AI Engineer"

input_1 = {
    "messages" : [HumanMessage(content= message_1)]
}

result_1 = graph.invoke(input_1, config=config)

print(f"User: {message_1}")
print(f"AI: {result_1['messages'][-1].content}")

message_2 = "What's my name?"

input_2 = {
    "messages" : [HumanMessage(content= message_2)]
}

result_2 = graph.invoke(input_2, config=config)

print(f"User: {message_2}")
print(f"AI: {result_2['messages'][-1].content}")





