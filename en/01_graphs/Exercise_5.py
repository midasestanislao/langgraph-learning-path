from typing import TypedDict
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    status: str
    attempts: int
    max_attempts: int

def init_node(state: AgentState) -> AgentState:
    """Initializes the state before starting the attempts"""
    state['status'] = "pending"
    state['attempts'] = 0
    state['max_attempts'] = 5
    return state

def attempt_node(state: AgentState) -> AgentState:
    """Simulates an attempt: 30% chance of success, 70% of failure"""
    state['attempts'] += 1

    if random.random() < 0.3:
        state['status'] = "success"
    else:
        state['status'] = "fail"

    print(f"Attempt #{state['attempts']}: {state['status']}")
    return state

def should_continue(state: AgentState) -> str:
    """Decides whether to retry or exit. Exits if there was a success OR the attempts ran out."""
    if state['status'] == "success":
        print("Exiting: the attempt was successful")
        return "exit"
    elif state['attempts'] >= state['max_attempts']:
        print("Exiting: the attempts ran out")
        return "exit"
    else:
        return "loop"


graph = StateGraph(AgentState)

graph.add_node("init", init_node)
graph.add_node("attempt", attempt_node)
graph.add_edge("init", "attempt")

graph.add_conditional_edges(
    "attempt",
    should_continue,
    {
        "loop": "attempt",
        "exit": END
    }
)

graph.set_entry_point("init")

app = graph.compile()

result = app.invoke({})
print("\n--- Final result ---")
print(f"Status: {result['status']}")
print(f"Attempts used: {result['attempts']}/{result['max_attempts']}")
