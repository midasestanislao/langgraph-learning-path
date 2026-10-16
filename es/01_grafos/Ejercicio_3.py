from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    count: int
    message: str
    limit: int

def greeting_node(state: AgentState) -> AgentState:
    """This node adds a greeting message to the state and set the count and the limit"""
    state["message"] = f"Hello user, this program will sum a random number from 1 to {state['limit']}"
    state["count"] = 0
    return state

def counter_node(state: AgentState) -> AgentState:
    "This node increments the counter adding the random number to the sum until it reaches the limit"
    next_number = random.randint(1,20)
    state['count'] += next_number
    print(f"The sum is now {state['count']} and the random number added is {next_number}")
    return state

def should_continue(state: AgentState) -> str:
    "This node decides if the counter should continue or not"
    if state['count'] < state['limit']:
        return "loop"
    else:
        print(f"We reach the total sum is: {state['count']}")
        return "exit"
    
        
graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("counter_loop", counter_node)
graph.add_edge("greeting", "counter_loop")

graph.add_conditional_edges(
    "counter_loop",
    should_continue,
    {
        "loop": "counter_loop",
        "exit": END
    }
)

graph.set_entry_point("greeting")

app = graph.compile()

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"limit": 100})
