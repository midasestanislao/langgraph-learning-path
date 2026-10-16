from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    pending: List[int]
    evens: List[int]
    odds: List[int]

def greeting_node(state: AgentState) -> AgentState:
    """Estado tiene pending: list[int] con 20 números. Cada iteración saca uno, si es par lo mueve a evens, si es impar a odds"""
    state['pending'] = []
    state['evens'] = []
    state['odds'] = []
    for i in range(1,21):
        state['pending'].append(i)
    return state

def extractor_node(state: AgentState) -> AgentState:
    "This node extract all the number inside of pending's array"
    extacted_number = state['pending'].pop()
    if extacted_number % 2 == 0:
        state['evens'].append(extacted_number)
    else:
        state['odds'].append(extacted_number)
    
    print(f"Extracted number: {extacted_number} (Even's list: {state['evens']}, Odd's list: {state['odds']})")
    return state

def should_continue(state: AgentState) -> str:
    "This node decides if the counter should continue or not"
    if len(state['pending']) != 0:
        return "loop"
    else:
        print(f"We extract all numbers pending")
        return "exit"
    
        
graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("extractor_loop", extractor_node)
graph.add_edge("greeting", "extractor_loop")

graph.add_conditional_edges(
    "extractor_loop",
    should_continue,
    {
        "loop": "extractor_loop",
        "exit": END
    }
)

graph.set_entry_point("greeting")

app = graph.compile()

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({})
print(result['pending'])
print(result['evens'])
print(result['odds'])
