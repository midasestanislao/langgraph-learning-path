from typing import TypedDict, List
from langgraph.graph import StateGraph, END


def fake_fetch(page: int) -> dict:
    """Simulates a paginated API. Returns 5 items and a has_next flag.
    Pages 1 to 4 have a next page. It ends at page 5."""
    items = [f"item-{page}-{i}" for i in range(1, 6)]
    has_next = page < 5
    return {"items": items, "has_next": has_next}


class AgentState(TypedDict):
    page: int
    items: List[str]
    has_next: bool
    max_pages: int


def init_node(state: AgentState) -> AgentState:
    """Prepares the state before starting to paginate"""
    state['page'] = 1
    state['items'] = []
    state['has_next'] = True
    state['max_pages'] = 10
    return state


def fetch_node(state: AgentState) -> AgentState:
    """Calls fake_fetch with the current page and accumulates the items"""
    response = fake_fetch(state['page'])
    state['items'].extend(response['items'])
    state['has_next'] = response['has_next']

    print(f"Page {state['page']}: +{len(response['items'])} items | has_next={state['has_next']}")

    state['page'] += 1
    return state


def should_continue(state: AgentState) -> str:
    """Exits if there is no next page OR if the page limit has already been passed"""
    if not state['has_next']:
        print("Exiting: there are no more pages")
        return "exit"
    elif state['page'] > state['max_pages']:
        print("Exiting: we reached the page limit")
        return "exit"
    else:
        return "loop"


graph = StateGraph(AgentState)

graph.add_node("init", init_node)
graph.add_node("fetch", fetch_node)
graph.add_edge("init", "fetch")

graph.add_conditional_edges(
    "fetch",
    should_continue,
    {
        "loop": "fetch",
        "exit": END
    }
)

graph.set_entry_point("init")

app = graph.compile()

result = app.invoke({})
print("\n--- Final result ---")
print(f"Total items collected: {len(result['items'])}")
print(f"Pages visited: {result['page'] - 1}")
print(f"First 3 items: {result['items'][:3]}")
print(f"Last 3 items: {result['items'][-3:]}")
