from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.embeddings import init_embeddings
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from typing import TypedDict, Annotated
from operator import add
from datetime import datetime
import os 
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

llm = ChatOpenAI(model="gpt-5.4-mini")

class ChatMessagesState(TypedDict):
    messages: Annotated[list,add]
    memory_context : str

def retrive_mem(state: ChatMessagesState,  config : RunnableConfig, store:BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")

    user_mem_namespace = (user_id, "memories")
    memories = store.search(
        user_mem_namespace, 
        query = "what are facts about this user"
    )

    mem_context = ""
    if memories:
        print(f"Found {len(memories)} memories ")
        mem_texts = []
        for i , memory in enumerate(memories,1):
            text = memory.value.get("text","")
            print(f"{i} . {text}")
            mem_texts.append(text)
        mem_context =  "\n".join([f"- {text}" for text in mem_texts])
    else:
        print("No mem found")
    
    return{
        "memory_context" : mem_context
    }


def chatbot(state:ChatMessagesState, config : RunnableConfig):
    user_id = config["configurable"].get("user_id", "default_user")

    mem_context = state.get("memory_context", "")
    if mem_context:
        print("using mem context")

        system_prompt = f""" You're helpful assistant with memory of past conversations

        What you remember about this user : {mem_context}

        use this information to personalize your response. be natureal and conversational 
        """
    else:
        print("no context available ")
        system_prompt = f""" You're helpful assistant. This is ur first convo with this user """

    messages = [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ]

    response = llm.invoke(messages)

    return{
        "messages": [response]
    }

def mem_extraction(state: ChatMessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")

    if len(state["messages"])>=2:
        user_msg = state["messages"][-2].content
        assistant_msg = state["messages"][-1].content
    else:
        return state
    
    print(f"User said: {user_msg[:60]}")
    print(f"Assistant said: {assistant_msg[:60]}")

    extract_prompt = f"""Look at this conversation and extract any facts worth remembering about
    the user.

    User: {user_msg}
    Assistant: {assistant_msg}

    List each fact on a new line starting with a dash (-).
    Only include clear, factual information about the USER (not about the assistant).
    If there are no facts to remember, respond with: NONE

    Examples of good facts:
    - User's name is Alice
    - User works as a teacher
    - User enjoys hiking
    - User is learning Python

    Examples of bad facts (don't include these) :
    - The assistant was helpful
    - We had a conversation
    - The user asked a question"""

    extraction = llm.invoke(extract_prompt).content

    print(f"extraction result: {extraction[:80]}")

    if "NONE" not in extraction.upper():
        
        lines = [line.strip() for line in extraction.split("\n") if line.strip().startswith("-")]

        saved_count = 0
        for line in lines:
            fact = line[1:].strip()

            if fact and len(fact)>5:
                mem_key = f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                store.put(
                    namespace= (user_id, "memories"),
                    key = mem_key,
                    value = {
                        "text": fact,
                        "timestamp": datetime.now().isoformat(),
                        "source": "conversation"
                    }
                )
                print(f"saved fact: {fact}")
                saved_count += 1   

    return state


builder = StateGraph(ChatMessagesState)
builder.add_node(mem_extraction)
builder.add_node(chatbot)
builder.add_node(retrive_mem)

builder.add_edge(START, "retrive_mem")
builder.add_edge("retrive_mem", "chatbot")
builder.add_edge("chatbot", "mem_extraction")
builder.add_edge("mem_extraction", END)

checkpointer = MemorySaver()
store_embed_model = init_embeddings("openai: text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed" : store_embed_model,
        "dims" : 1536,
        "fields" : ["text", "$"]
    }
)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)

config = {
    "configurable":{
        "thread_id" : "chat-001",
        "user_id": "bobby"
    }
}

sarah_msg_1 = "HI my name is Goutham. I'm a Software Engineer"

result  = graph.invoke (
    {
        "messages": [HumanMessage(content = sarah_msg_1)]
    },
    config = config
)


print(f" Last message: {result["messages"][-1].content}")
result2  = graph.invoke (
    {
        "messages": [HumanMessage(content = "what does goutham do?")]
    },
    config = config
)

print(f" Last message: {result2["messages"][-1].content}")
    

