"""
EJERCICIO 16 — El supervisor de juguete: todas las piezas juntas     (usa LLM: solo el supervisor)
=====================================================================================
CONCEPTOS (ninguno nuevo: es un MONTAJE de lo que ya sabes)

    13   ConductResearch / ResearchComplete -> los formularios que rellena el supervisor
    10/11 Command -> el bucle supervisor ⇄ supervisor_tools, sin add_conditional_edges
    14/15 gather de app_investigador.ainvoke -> un investigador por formulario, todos a la vez
    M03  ToolMessage(content=..., tool_call_id=...) -> devolverle las notas al supervisor
    M05  notas: Annotated[list[str], operator.add] -> las notas se ACUMULAN ronda a ronda

El bucle

    START → supervisor ──► supervisor_tools ──(hay ConductResearch)──► supervisor   (otra ronda)
                                             └─(ResearchComplete, sin formularios
                                                o se acabaron las rondas)──────────► END

    supervisor:        el LLM lee el brief (y las notas que ya hay) y rellena formularios
    supervisor_tools:  tu código lee los formularios y lanza un investigador por cada uno

Cómo "ve" el supervisor lo que hicieron los investigadores
    Cada ConductResearch es un tool_call con su "id". Cuando el investigador termina,
    sus notas vuelven como ToolMessage(content=notas, tool_call_id=id), igual que en M03.
    En la siguiente ronda el LLM lee esas notas y decide: ¿falta algo? ¿o ya terminó?

Dos frenos (como en la Fase 2)
    Blando: el prompt dice cuántos temas delegar y cuándo parar.
    Duro:   MAX_RONDAS. En la ronda MAX_RONDAS, supervisor_tools termina sí o sí.

ENUNCIADO (ejemplo resuelto)

Brief: comparar el sistema educativo de Finlandia, Japón y Chile.
El supervisor debería delegar 3 temas (uno por país) en la primera ronda, recibir
las 3 notas a la vez (1 segundo, no 3) y en la segunda ronda decidir si termina.

Salida esperada (el LLM puede variar en los textos y en pedir una ronda más):
    --- Ronda 1: el supervisor pide ['ConductResearch', 'ConductResearch', 'ConductResearch']
        Lanzando 3 investigadores a la vez...
        ✓ Sistema educativo de Finlandia: resultados, horas de clase y formación docente
        ✓ Sistema educativo de Japón: ...
        ✓ Sistema educativo de Chile: ...
        (1.0 s)
    --- Ronda 2: el supervisor pide ['ResearchComplete']
        Fin: el supervisor dio por terminada la investigación

    NOTAS ACUMULADAS (3):
    - Notas sobre Sistema educativo de Finlandia...: dato A..., dato B...
    - ...

Para pensar:
    - Sube MAX_RONDAS a 1. ¿Qué pasa? ¿Quién corta el bucle, el LLM o tu código?
    - Cambia el brief por uno simple ("¿quién inventó el espresso?"). ¿Cuántos
      ConductResearch pide? ¿Sigue teniendo sentido lanzar investigadores en paralelo?
    - Lección real de este ejercicio: la primera versión del prompt decía
      "un tema por cada cosa comparada". El LLM lo entendió como un tema por ASPECTO
      (resultados, horas, docentes) × país = 9 temas, 3 por ronda, y el freno duro
      lo cortó en la ronda 3 sin terminar. Con "un tema por país, con TODOS los
      aspectos" hace 3 temas y cierra en la ronda 2. El prompt del supervisor decide
      cuánto trabajo (y dinero) genera.

Ejecuta:  uv run python es/07_deep_research/Ejercicio_16_0_supervisor_juguete.py
"""
import asyncio
import operator
import os
import time
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

MAX_RONDAS = 3              # freno duro: llamadas al supervisor como máximo
MAX_INVESTIGADORES = 3      # freno blando: se lo decimos al LLM en el prompt


def get_today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


# ============ 1. INVESTIGADOR DE JUGUETE (tu 14.1, sin cambios) ============
class InvestigadorState(TypedDict):
    tema: str
    resultados: list[str]
    notas: str


def buscar_falso(state: InvestigadorState) -> dict:
    time.sleep(1)       # simula la búsqueda
    return {"resultados": [f"dato A sobre {state['tema']}", f"dato B sobre {state['tema']}"]}


def comprimir_falso(state: InvestigadorState) -> dict:
    return {"notas": f"Notas sobre {state['tema']}: " + ", ".join(state["resultados"])}


graph_investigador = StateGraph(InvestigadorState)
graph_investigador.add_node("buscar_falso", buscar_falso)
graph_investigador.add_node("comprimir_falso", comprimir_falso)
graph_investigador.add_edge(START, "buscar_falso")
graph_investigador.add_edge("buscar_falso", "comprimir_falso")
graph_investigador.add_edge("comprimir_falso", END)
app_investigador = graph_investigador.compile()


# ============ 2. LOS FORMULARIOS DEL SUPERVISOR (Ejercicio 13) ============
class ConductResearch(BaseModel):
    """Delega la investigación de UN tema a un investigador. Úsala una vez por cada tema distinto."""
    research_topic: str = Field(description="El tema a investigar, descrito con detalle en una frase.")


class ResearchComplete(BaseModel):
    """Úsala cuando las notas recibidas ya cubren el brief y no hace falta investigar más."""


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
model_supervisor = llm.bind_tools([ConductResearch, ResearchComplete])

SUPERVISOR_PROMPT = """Eres un supervisor de investigación. Hoy es {date}.
Recibes un brief. Tú NO investigas: DELEGAS con ConductResearch, un formulario por tema.

Cómo trabajar:
- Divide el brief en POCOS temas independientes. En una comparación, un tema por cada
  elemento comparado (por ejemplo, un país), y ese tema debe incluir TODOS los aspectos
  que pide el brief. No hagas un tema por aspecto.
- Preguntas sencillas: 1 o 2 temas. Máximo {max_investigadores} temas por ronda.
- Cuando recibas las notas de los investigadores, revisa si cubren el brief.
  Si falta algo importante, delega SOLO lo que falta. Si ya está cubierto, usa ResearchComplete.
- No repitas un tema que ya tiene notas."""


# ============ 3. EL ESTADO DEL SUPERVISOR ============
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # brief + formularios + notas (ToolMessages)
    notas: Annotated[list[str], operator.add]                 # se acumulan ronda a ronda (M05)
    rondas: int


# ============ 4. NODO supervisor: el LLM rellena formularios ============
async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    prompt = SUPERVISOR_PROMPT.format(date=get_today(), max_investigadores=MAX_INVESTIGADORES)
    response = await model_supervisor.ainvoke([SystemMessage(content=prompt)] + state["messages"])

    print(f"--- Ronda {state['rondas'] + 1}: el supervisor pide {[c['name'] for c in response.tool_calls]}")
    return Command(
        goto="supervisor_tools",
        update={"messages": [response], "rondas": state["rondas"] + 1},
    )


# ============ 5. NODO supervisor_tools: tu código lee los formularios y actúa ============
async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    ultimo = state["messages"][-1]                        # el AIMessage con los formularios
    pedidos = [c for c in ultimo.tool_calls if c["name"] == "ConductResearch"]
    termino = any(c["name"] == "ResearchComplete" for c in ultimo.tool_calls)

    # a) ¿Hay que parar? (los dos frenos + el caso "no pidió nada")
    if termino:
        print("    Fin: el supervisor dio por terminada la investigación")
        return Command(goto=END, update={})
    if not pedidos:
        print("    Fin: el supervisor no pidió más investigaciones")
        return Command(goto=END, update={})
    if state["rondas"] >= MAX_RONDAS:
        print(f"    Fin: se alcanzó el límite de {MAX_RONDAS} rondas (freno duro)")
        return Command(goto=END, update={})

    # b) Un investigador por formulario, todos a la vez (Ejercicio 15)
    print(f"    Lanzando {len(pedidos)} investigadores a la vez...")
    inicio = time.perf_counter()
    tareas = [app_investigador.ainvoke({"tema": c["args"]["research_topic"]}) for c in pedidos]
    resultados = await asyncio.gather(*tareas)

    # c) Devolverle las notas al supervisor como ToolMessages (M03): un ToolMessage por formulario,
    #    enlazado por tool_call_id. zip empareja cada resultado con su formulario (mismo orden).
    tool_messages = []
    for resultado, pedido in zip(resultados, pedidos):
        print(f"    ✓ {pedido['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=resultado["notas"], tool_call_id=pedido["id"]))
    print(f"    ({time.perf_counter() - inicio:.1f} s)")

    return Command(
        goto="supervisor",                                            # otra ronda: el LLM leerá las notas
        update={"messages": tool_messages, "notas": [r["notas"] for r in resultados]},
    )


# ============ 6. EL GRAFO: un solo edge; el bucle lo hacen los Command (Ejercicio 11) ============
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("supervisor_tools", supervisor_tools)
graph.add_edge(START, "supervisor")
app_supervisor = graph.compile()


# ============ 7. Probarlo ============
BRIEF = """Quiero comparar el sistema educativo de Finlandia, Japón y Chile: resultados
académicos, horas de clase semanales y formación de los docentes."""


async def main():
    resultado = await app_supervisor.ainvoke(
        {"messages": [HumanMessage(content=BRIEF)], "notas": [], "rondas": 0}
    )
    print(f"\nNOTAS ACUMULADAS ({len(resultado['notas'])}):")
    for nota in resultado["notas"]:
        print(f"- {nota}")


if __name__ == "__main__":
    asyncio.run(main())
