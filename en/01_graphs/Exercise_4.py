from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    pending: List[int]
    evens: List[int]
    odds: List[int]

def greeting_node(state: AgentState) -> AgentState:
    """The state has pending: list[int] with 20 numbers. Each iteration takes one out; if it is even it goes to evens, if it is odd it goes to odds"""
    state['pending'] = []
    state['evens'] = []
    state['odds'] = []
    for i in range(1,21):
        state['pending'].append(i)
    return state

def extractor_node(state: AgentState) -> AgentState:
    "This node extracts all the numbers inside the pending list"
    extracted_number = state['pending'].pop()
    if extracted_number % 2 == 0:
        state['evens'].append(extracted_number)
    else:
        state['odds'].append(extracted_number)

    print(f"Extracted number: {extracted_number} (Evens list: {state['evens']}, Odds list: {state['odds']})")
    return state

def should_continue(state: AgentState) -> str:
    "This node decides if the counter should continue or not"
    if len(state['pending']) != 0:
        return "loop"
    else:
        print(f"We extracted all the pending numbers")
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
