from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class GraphState(MessagesState):
    turn_count: int

def user_node(state: GraphState) -> dict:
   print("exec user_node")
   return {"messages": [HumanMessage(content="what is the weather like today?")]}

def ai_node(state: GraphState) -> dict:
   print("exec ai_node")
   last_message = state["messages"][-1].content
   print(f"last message: {last_message}")
   response_content = f"I got your message '{last_message}' and the weather is sunny!"
   return {"messages": [AIMessage(content=response_content)]}

def counter_node(state: GraphState) -> dict:
   print("exec counter_node")
   turn_count = state["turn_count"] + 1
   print(f"turn count: {turn_count}")
   return {"turn_count": turn_count}

graph = StateGraph(GraphState)
graph.add_node("user_node", user_node)
graph.add_node("ai_node", ai_node)
graph.add_node("counter_node", counter_node)
graph.add_edge(START, "user_node")
graph.add_edge("user_node", "ai_node")
graph.add_edge("ai_node", "counter_node")
graph.add_edge("counter_node", END)

agent = graph.compile()
initial_state = {"turn_count": 0}
final_state = agent.invoke(initial_state)
print(final_state)