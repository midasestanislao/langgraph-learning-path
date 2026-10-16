"""
EJERCICIO M05 — Reducers: por qué los mensajes se ACUMULAN en el estado   (sin LLM, gratis)
==========================================================================================
En tus ejercicios 7-9 escribiste:
    messages: Annotated[Sequence[BaseMessage], add_messages]

y tus nodos devolvían {"messages": [response]}, solo UN mensaje. ¿Por qué no se borraba
el resto del historial? Porque `add_messages` es un REDUCER: cuando un nodo devuelve un
valor para ese campo, LangGraph no lo REEMPLAZA, sino que llama a:

    nuevo_estado = add_messages(valor_anterior, valor_devuelto_por_el_nodo)

Sin reducer, el valor devuelto REEMPLAZA al anterior.
MessagesState (paso 1) es exactamente un TypedDict con ese campo ya declarado.

En el deep research también verás `Annotated[list[str], operator.add]` para notas.

Ejecuta:  uv run python 04_mensajes/M05_reducers.py
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages

# ---------- Parte A: reducers como funciones normales ----------
viejos = [HumanMessage("hola", id="1"), AIMessage("¿qué tal?", id="2")]
nuevos = [HumanMessage("bien, ¿y tú?", id="3")]

# TODO 1: llama a add_messages(viejos, nuevos) a mano. ¿Cuántos mensajes quedan?
# Un reducer es una función normal: recibe (lo que había, lo nuevo) y devuelve el resultado.
# El id "3" no existía -> se AÑADE al final -> 2 + 1 = 3 mensajes.
resultado_1 = add_messages(viejos, nuevos)
print(f"TODO 1: {len(resultado_1)} mensajes -> {[m.content for m in resultado_1]}")

# TODO 2: llama a add_messages(viejos, [AIMessage("¿cómo estás?", id="2")]).
#   Mira el resultado: ¿qué pasa cuando el id YA existe?
# El id "2" YA existía -> add_messages lo REEMPLAZA en su sitio en vez de añadirlo.
# Sigue habiendo 2 mensajes, pero el segundo ahora dice "¿cómo estás?".
# (Útil para "corregir" un mensaje del historial sin duplicarlo.)
resultado_2 = add_messages(viejos, [AIMessage("¿cómo estás?", id="2")])
print(f"TODO 2: {len(resultado_2)} mensajes -> {[m.content for m in resultado_2]}")

# TODO 3: operator.add con listas de strings es simplemente `a + b`.
#   Calcula operator.add(["nota 1"], ["nota 2", "nota 3"])
# operator.add(a, b) es literalmente a + b. Con listas, "+" las concatena.
# Es más simple que add_messages: no mira ids, solo pega una lista detrás de otra.
resultado_3 = operator.add(["nota 1"], ["nota 2", "nota 3"])
print(f"TODO 3: {resultado_3}")


# ---------- Parte B: con reducer vs. sin reducer en un grafo real ----------
class ConReducer(MessagesState):
    pass


class SinReducer(TypedDict):
    messages: list[AnyMessage]   # sin Annotated -> sin reducer


def responder(state):
    return {"messages": [AIMessage("Hola, soy el nodo")]}


# TODO 4: construye y compila DOS grafos idénticos (START -> "responder" -> END):
#   uno con ConReducer y otro con SinReducer. Invoca ambos con
#   {"messages": [HumanMessage("hola")]} e imprime cuántos mensajes tiene cada resultado.
# Como los dos grafos son iguales salvo el estado, hacemos un helper que recibe
# la clase de estado y devuelve el grafo compilado. Así no repetimos código.
def construir_grafo(clase_estado):
    graph = StateGraph(clase_estado)
    graph.add_node("responder", responder)
    graph.add_edge(START, "responder")
    graph.add_edge("responder", END)
    return graph.compile()


entrada = {"messages": [HumanMessage("hola")]}
resultado_con = construir_grafo(ConReducer).invoke(entrada)
resultado_sin = construir_grafo(SinReducer).invoke(entrada)

# CON reducer: [hola] + [Hola, soy el nodo]  -> 2 mensajes (se acumulan)
# SIN reducer: [Hola, soy el nodo]           -> 1 mensaje  (el "hola" se PERDIÓ)
print(f"\nTODO 4: con reducer = {len(resultado_con['messages'])} mensajes, "
      f"sin reducer = {len(resultado_sin['messages'])} mensaje")


# TODO 5 (reto): crea un estado con un campo `notas: Annotated[list[str], operator.add]`
#   y un grafo de DOS nodos seguidos donde cada uno devuelve {"notas": ["nota de X"]}.
#   Invócalo con {"notas": []}. Al final `notas` debe tener las 2 notas.
#   ¿Qué pasaría sin operator.add?
class EstadoNotas(TypedDict):
    notas: Annotated[list[str], operator.add]   # Annotated[tipo, reducer]


def investigador_a(state: EstadoNotas):
    # Devuelve SOLO su nota nueva; el reducer se encarga de sumarla a las anteriores.
    return {"notas": ["nota de A"]}


def investigador_b(state: EstadoNotas):
    return {"notas": ["nota de B"]}


graph_notas = StateGraph(EstadoNotas)
graph_notas.add_node("investigador_a", investigador_a)
graph_notas.add_node("investigador_b", investigador_b)
graph_notas.add_edge(START, "investigador_a")
graph_notas.add_edge("investigador_a", "investigador_b")
graph_notas.add_edge("investigador_b", END)

resultado_notas = graph_notas.compile().invoke({"notas": []})
print(f"TODO 5: {resultado_notas['notas']}")
# Sin operator.add, investigador_b REEMPLAZARÍA la nota de A y solo quedaría ["nota de B"].
# En el deep research esto es justo lo que pasa cuando varios investigadores trabajan:
# cada uno devuelve SUS notas y el reducer las junta todas.


# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert resultado_1 is not None and len(resultado_1) == 3, "TODO 1"
assert resultado_2 is not None and len(resultado_2) == 2 and resultado_2[1].content == "¿cómo estás?", \
    "TODO 2: con el mismo id, add_messages REEMPLAZA en vez de añadir"
assert resultado_3 == ["nota 1", "nota 2", "nota 3"], "TODO 3"
assert resultado_con is not None and len(resultado_con["messages"]) == 2, "TODO 4: con reducer deberían ser 2"
assert resultado_sin is not None and len(resultado_sin["messages"]) == 1, "TODO 4: sin reducer debería ser 1"
assert resultado_notas is not None and len(resultado_notas["notas"]) == 2, "TODO 5"
print("\n✅ M05 completado")
