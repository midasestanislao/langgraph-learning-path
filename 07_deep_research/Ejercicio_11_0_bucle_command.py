"""
EJERCICIO 11 — Un bucle con Command                                  (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Bucle con Command -> un nodo puede hacer goto a SÍ MISMO.
    def contar(state) -> Command[Literal["contar", "despegar"]]:
        ...
        return Command(goto="contar", update={...})    # se vuelve a ejecutar

    Cada vez que vuelve, el nodo recibe el estado YA actualizado por el update anterior.

Salida del bucle -> en algún momento el goto debe apuntar a OTRO sitio (otro nodo o END).
    Si nunca sale, LangGraph lo corta con un error GraphRecursionError.
    Ese es su "freno de emergencia": el recursion_limit (máximo de pasos).
    En tu versión el límite por defecto es muy alto (10.007), pero puedes bajarlo:

        app.invoke({"numero": 3}, {"recursion_limit": 10})

Antes vs. ahora
    Antes (Ejercicio_1):  nodo + should_continue + add_conditional_edges
    Ahora:                el nodo decide solo si repite o sale

    Así hace su bucle el supervisor del deep research.

ENUNCIADO (ejemplo resuelto)

Una cuenta regresiva para un cohete: cuenta desde un número hasta 0 y luego despega.

    START → contar ──(numero > 0)──► contar   (vuelve a sí mismo)
                   └─(numero = 0)──► despegar → END

  Estado:  numero: int   y   mensaje: str

  - contar:   imprime el número.
              Si es mayor que 0 -> vuelve a "contar" con numero - 1
              Si es 0           -> va a "despegar"
  - despegar: pone mensaje = "🚀 ¡Despegue!"

Salida esperada con numero = 3:
    3...
    2...
    1...
    0...
    {'numero': 0, 'mensaje': '🚀 ¡Despegue!'}

Ejecuta:  uv run python 07_deep_research/Ejercicio_11_0_bucle_command.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. El estado
class CohetesState(TypedDict):
    numero: int
    mensaje: str


# 2. El nodo del BUCLE. Su Literal incluye su propio nombre: "contar".
def contar(state: CohetesState) -> Command[Literal["contar", "despegar"]]:
    print(f"{state['numero']}...")

    if state["numero"] > 0:
        # Vuelve a sí mismo con un número menos.
        # En la siguiente vuelta, state["numero"] ya valdrá numero - 1.
        return Command(goto="contar", update={"numero": state["numero"] - 1})

    # Salida del bucle: cuando llega a 0, va a otro nodo
    return Command(goto="despegar", update={})


# 3. Nodo normal
def despegar(state: CohetesState) -> dict:
    return {"mensaje": "🚀 ¡Despegue!"}


# 4. El grafo: solo 2 edges. Ninguno sale de "contar" (ni siquiera el del bucle).
graph = StateGraph(CohetesState)
graph.add_node("contar", contar)
graph.add_node("despegar", despegar)

graph.add_edge(START, "contar")
graph.add_edge("despegar", END)

app = graph.compile()


# 5. Probarlo
print(app.invoke({"numero": 3}))
