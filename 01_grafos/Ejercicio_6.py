from typing import TypedDict, List
from langgraph.graph import StateGraph, END


def fake_fetch(page: int) -> dict:
    """Simula una API paginada. Devuelve 5 items y un flag has_next.
    De la pagina 1 a la 4 hay siguiente. En la pagina 5 se acaba."""
    items = [f"item-{page}-{i}" for i in range(1, 6)]
    has_next = page < 5
    return {"items": items, "has_next": has_next}


class AgentState(TypedDict):
    page: int
    items: List[str]
    has_next: bool
    max_pages: int


def init_node(state: AgentState) -> AgentState:
    """Prepara el estado antes de empezar a paginar"""
    state['page'] = 1
    state['items'] = []
    state['has_next'] = True
    state['max_pages'] = 10
    return state


def fetch_node(state: AgentState) -> AgentState:
    """Llama al fake_fetch con la pagina actual y acumula los items"""
    response = fake_fetch(state['page'])
    state['items'].extend(response['items'])
    state['has_next'] = response['has_next']

    print(f"Pagina {state['page']}: +{len(response['items'])} items | has_next={state['has_next']}")

    state['page'] += 1
    return state


def should_continue(state: AgentState) -> str:
    """Sale si no hay siguiente pagina O si ya paso el limite de paginas"""
    if not state['has_next']:
        print("Salimos: no hay mas paginas")
        return "exit"
    elif state['page'] > state['max_pages']:
        print("Salimos: llegamos al limite de paginas")
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
print("\n--- Resultado final ---")
print(f"Total de items recolectados: {len(result['items'])}")
print(f"Paginas visitadas: {result['page'] - 1}")
print(f"Primeros 3 items: {result['items'][:3]}")
print(f"Ultimos 3 items: {result['items'][-3:]}")
