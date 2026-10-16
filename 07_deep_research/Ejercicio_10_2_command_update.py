"""
EJERCICIO 10.2 — Command con update: decidir Y guardar datos         (sin LLM, gratis)
=====================================================================================
CONCEPTOS

update -> la parte del Command que CAMBIA el estado.
    return Command(goto="adulto", update={"categoria": "adulto"})

    Es exactamente lo mismo que el `return {"categoria": "adulto"}` de un nodo normal,
    pero junto con el goto.

¿Cuándo se aplica el update?
    ANTES de que se ejecute el nodo del goto. Por eso el siguiente nodo ya puede
    leer lo que guardó el Command:

    revisar_edad  →  update={"categoria": "adulto"}  →  adulto lee state["categoria"]

update vacío vs. update con datos
    update={}                         -> solo decide el camino, no cambia nada
    update={"categoria": "adulto"}    -> decide el camino Y guarda un dato
    update={"a": 1, "b": 2}           -> puede cambiar varios campos a la vez

ENUNCIADO

Igual que el 10.1, pero ahora revisar_edad también GUARDA la categoría en el estado,
y los otros nodos la usan para armar su mensaje.

    START → revisar_edad ──(>= 18)───► adulto      → END
                         ├─(13 a 17)─► adolescente → END
                         └─(< 13)────► nino        → END

  Estado:  edad: int,  categoria: str  (NUEVO),  mensaje: str

Salida esperada:
    {'edad': 25, 'categoria': 'adulto', 'mensaje': 'Hola, eres adulto. Ya puedes votar.'}
    {'edad': 15, 'categoria': 'adolescente', 'mensaje': 'Hola, eres adolescente. Te faltan 3 años para votar.'}
    {'edad': 8, 'categoria': 'niño', 'mensaje': 'Hola, eres niño. Te faltan 10 años para votar.'}

Ejecuta:  uv run python 07_deep_research/Ejercicio_10_2_command_update.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. El estado: ahora tiene un campo NUEVO, `categoria`
class EdadState(TypedDict):
    edad: int
    categoria: str     # lo rellena revisar_edad con su Command
    mensaje: str


# 2. El nodo que DECIDE y además GUARDA la categoría.
#    Antes: update={}   →   Ahora: update={"categoria": ...}
def revisar_edad(state: EdadState) -> Command[Literal["adulto", "adolescente", "nino"]]:
    if state["edad"] >= 18:
        return Command(goto="adulto", update={"categoria": "adulto"})
    elif state["edad"] >= 13:
        return Command(goto="adolescente", update={"categoria": "adolescente"})
    return Command(goto="nino", update={"categoria": "niño"})


# 3. Nodos normales que LEEN lo que guardó el Command.
#    Cuando llegan aquí, state["categoria"] ya tiene valor.
def adulto(state: EdadState) -> dict:
    return {"mensaje": f"Hola, eres {state['categoria']}. Ya puedes votar."}


def adolescente(state: EdadState) -> dict:
    faltan = 18 - state["edad"]
    return {"mensaje": f"Hola, eres {state['categoria']}. Te faltan {faltan} años para votar."}


def nino(state: EdadState) -> dict:
    faltan = 18 - state["edad"]
    return {"mensaje": f"Hola, eres {state['categoria']}. Te faltan {faltan} años para votar."}


# 4. El grafo: igual que en el 10.1
graph = StateGraph(EdadState)
graph.add_node("revisar_edad", revisar_edad)
graph.add_node("adulto", adulto)
graph.add_node("adolescente", adolescente)
graph.add_node("nino", nino)

graph.add_edge(START, "revisar_edad")
# Ningún add_edge sale de "revisar_edad": lo decide el Command
graph.add_edge("adulto", END)
graph.add_edge("adolescente", END)
graph.add_edge("nino", END)

app = graph.compile()


# 5. Probarlo. Fíjate en que `categoria` aparece en el resultado final:
#    la puso el update del Command.
print(app.invoke({"edad": 25}))
print(app.invoke({"edad": 15}))
print(app.invoke({"edad": 8}))
