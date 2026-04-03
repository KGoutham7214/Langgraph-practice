from typing import TypedDict, Literal 
from langgraph.graph import StateGraph, START
from langgraph.types import Command

class GraphState(TypedDict):
    temperature : int
    status_message : str
    warning_agent: bool
    final_action: str 

def check_temp_node(state: GraphState) -> Command[Literal["warn_user", "success"]]:
    if state["temperature"] > 90:
        print("Temperature is too high! Warning the user.")
        return Command(
            update={"warning_agent": True, "status_message": "Temperature is too high!"},
            goto="warn_user"
        )
    else:
        print("Temperature is normal. Proceeding with success.")
        return Command(
            update={"status_message": "Temperature is normal.","warning_agent": False},
            goto="success"
        )
    
def warn_user(state: GraphState) -> dict:
    print("Executing warn_user node.")
    return {
        "warning_agent": True,
        "final_action": "warned_user"
    }

def success(state: GraphState) -> dict:
    print("Executing success node.")
    return {
        "warning_agent": False,
        "final_action": "proceeded_successfully"
    }

graph = StateGraph(GraphState)
graph.add_node("check_temp", check_temp_node)
graph.add_node("warn_user", warn_user)
graph.add_node("success", success)
graph.add_edge(START, "check_temp")

agent = graph.compile()
initial_state = {
    "temperature": 75,
    "status_message": "",
    "warning_agent": False,
    "final_action": ""
}
final_state = agent.invoke(initial_state)
print(final_state)