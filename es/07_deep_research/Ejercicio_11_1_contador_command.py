"""
EJERCICIO 11.1 — Tu Ejercicio_1, pero con un bucle de Command        (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Los mismos del Ejercicio 11:
    - Un nodo hace goto a SÍ MISMO para repetirse.
    - En algún momento el goto apunta a otro sitio (otro nodo o END) para salir.
    - El Literal de la firma incluye el propio nombre del nodo:
          def counter(state) -> Command[Literal["counter", "__end__"]]:

ENUNCIADO

Reescribe tu 01_grafos/Ejercicio_1.py (contar desde 0 hasta un límite) SIN
should_continue y SIN add_conditional_edges.

    START → greeting → counter ──(count < limit)──► counter   (vuelve a sí mismo)
                               └─(count = limit)──► END

  Estado:  count: int,  message: str,  limit: int   (igual que en el Ejercicio_1)

  - greeting: pone el mensaje de bienvenida y count = 0. Es un nodo normal:
              devuelve un dict (sin modificar `state` directamente).
  - counter:  suma 1 a count e imprime "The count is now X".
              Si todavía no llegó al límite -> vuelve a "counter"
              Si ya llegó                   -> imprime "We reach the limit: X" y va a END
              En los dos casos, el update guarda el nuevo count.

  Pista: compara con el Ejercicio 11. Allí el bucle iba a otro nodo al salir;
         aquí sale directo a END (como en tu 10.3).

Salida esperada con limit = 3:
    The count is now 1
    The count is now 2
    The count is now 3
    We reach the limit: 3
    {'count': 3, 'message': 'Hello user, this program will count from 0 to 3', 'limit': 3}

Para pensar: ¿qué pasa si llamas a app.invoke({"limit": 30}, {"recursion_limit": 10})?
¿Hasta qué número llega a contar y por qué se detiene?
(Pista: lee lo que dice el Ejercicio 11 sobre el "freno de emergencia".)

Ejecuta:  uv run python es/07_deep_research/Ejercicio_11_1_contador_command.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. El estado
class CohetesState(TypedDict):
    numero: int
    mensaje: str
    limite: int

# 1. El nodo de saludo. Aqui iniciamos la variable
def inicio(state: CohetesState) -> Command[Literal["contar"]]:
    return Command(goto="contar", update={"numero":0})


# 2. El nodo del BUCLE. Su Literal incluye su propio nombre: "contar".
def contar(state: CohetesState) -> Command[Literal["contar", "despegar"]]:
    print(f"{state['numero']}...")

    if state["numero"] < state["limite"]:
        # Vuelve a sí mismo con un número menos.
        # En la siguiente vuelta, state["numero"] ya valdrá numero - 1.
        return Command(goto="contar", update={"numero": state["numero"] + 1})

    # Salida del bucle: cuando llega a limite, va a otro nodo
    return Command(goto="despegar", update={})


# 3. Nodo normal
def despegar(state: CohetesState) -> dict:
    return {"mensaje": "🚀 ¡Despegue!"}


# 4. El grafo: solo 2 edges. Ninguno sale de "contar" (ni siquiera el del bucle).
graph = StateGraph(CohetesState)
graph.add_node("inicio", inicio)
graph.add_node("contar", contar)
graph.add_node("despegar", despegar)

graph.add_edge(START, "inicio")
graph.add_edge("despegar", END)

app = graph.compile()


# 5. Probarlo
print(app.invoke({"limite": 3}))
