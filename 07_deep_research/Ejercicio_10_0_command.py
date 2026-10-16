"""
EJERCICIO 10 — Command: el nodo elige el camino                     (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Command -> un nodo devuelve DOS cosas a la vez:
    from langgraph.types import Command

    return Command(goto="otro_nodo", update={"campo": valor})

    goto   -> a qué nodo ir después
    update -> qué cambia en el estado (lo mismo que tu `return {...}` de siempre)

Literal en la firma -> la lista de nodos a los que PUEDE ir
    from typing import Literal

    def mi_nodo(state) -> Command[Literal["nodo_a", "nodo_b"]]:

Lo importante: del nodo que usa Command NO sale ningún add_edge.
El Command ya dice a dónde ir.

ENUNCIADO

Un programa que revisa una edad y saluda distinto a un adulto y a un menor.

    START → revisar_edad ──(edad >= 18)──► adulto → END
                         └─(edad < 18)───► menor  → END

  Estado:  edad: int   y   mensaje: str

  - revisar_edad: si la edad es 18 o más, va a "adulto"; si no, va a "menor".
                  No cambia nada del estado (update={}).
  - adulto:       pone mensaje = "Hola, eres mayor de edad"
  - menor:        pone mensaje = "Hola, eres menor de edad"

  adulto y menor son nodos normales: devuelven un dict, como siempre.

Prueba con dos edades:
    app.invoke({"edad": 25})  →  {'edad': 25, 'mensaje': 'Hola, eres mayor de edad'}
    app.invoke({"edad": 12})  →  {'edad': 12, 'mensaje': 'Hola, eres menor de edad'}

Ejecuta:  uv run python 07_deep_research/Ejercicio_10_0_command.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. El estado: las "cajas" que comparten todos los nodos
class EdadState(TypedDict):
    edad: int
    mensaje: str


# 2. El nodo que DECIDE. Usa Command en vez de devolver un dict.
#    El Literal dice a qué nodos puede ir: "adulto" o "menor".
def revisar_edad(state: EdadState) -> Command[Literal["adulto", "menor"]]:
    if state["edad"] >= 18:
        return Command(goto="adulto", update={})   # va a "adulto", no cambia nada
    return Command(goto="menor", update={})        # va a "menor", no cambia nada


# 3. Nodos normales: hacen su trabajo y devuelven un dict, como siempre
def adulto(state: EdadState) -> dict:
    return {"mensaje": "Hola, eres mayor de edad"}


def menor(state: EdadState) -> dict:
    return {"mensaje": "Hola, eres menor de edad"}


# 4. El grafo
graph = StateGraph(EdadState)
graph.add_node("revisar_edad", revisar_edad)
graph.add_node("adulto", adulto)
graph.add_node("menor", menor)

graph.add_edge(START, "revisar_edad")
# ⚠️ Aquí NO hay ningún add_edge que salga de "revisar_edad": lo decide el Command
graph.add_edge("adulto", END)
graph.add_edge("menor", END)

app = graph.compile()


# 5. Probarlo con dos edades
print(app.invoke({"edad": 25}))
print(app.invoke({"edad": 12}))
