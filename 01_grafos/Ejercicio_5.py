from typing import TypedDict
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    status: str
    attempts: int
    max_attempts: int

def init_node(state: AgentState) -> AgentState:
    """Inicializa el estado antes de empezar los intentos"""
    state['status'] = "pending"
    state['attempts'] = 0
    state['max_attempts'] = 5
    return state

def attempt_node(state: AgentState) -> AgentState:
    """Simula un intento: 30% de probabilidad de exito, 70% de fallo"""
    state['attempts'] += 1

    if random.random() < 0.3:
        state['status'] = "success"
    else:
        state['status'] = "fail"

    print(f"Intento #{state['attempts']}: {state['status']}")
    return state

def should_continue(state: AgentState) -> str:
    """Decide si reintentar o salir. Sale si hubo exito O se agotaron los intentos."""
    if state['status'] == "success":
        print("Salimos: el intento fue exitoso")
        return "exit"
    elif state['attempts'] >= state['max_attempts']:
        print("Salimos: se agotaron los intentos")
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
print("\n--- Resultado final ---")
print(f"Status: {result['status']}")
print(f"Intentos usados: {result['attempts']}/{result['max_attempts']}")
