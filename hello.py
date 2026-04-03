from langgraph.graph import StateGraph, START, END
from typing import Annotated, Dict, List
from langgraph.graph import add_messages

class TypedStateGraph(StateGraph):
    message: Annotated[List[str], add_messages]

def node_update(state: TypedStateGraph) -> dict:
    print("updated ")
    return {"message": "hehehehe"}

init_state = {
    "message": []
}

def main(TypedStateGraph, node_update, init_state):
    graph = StateGraph(TypedStateGraph)
    graph.add_node("start", node_update)
    graph.add_edge(START, "start")  
    graph.add_edge("start", END)
    finalstate = graph.compile().invoke(init_state)
      
    print(finalstate)

print(init_state)
main(TypedStateGraph, node_update, init_state)