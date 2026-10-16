"""
EJERCICIO 16.1 — Un supervisor que piensa antes de delegar           (usa LLM: solo el supervisor)
=====================================================================================
CONCEPTOS

Todo es lo del Ejercicio 16. Lo nuevo es darle al supervisor un think_tool (Ejercicio 12)
y que supervisor_tools sepa qué hacer con él.

Tres tipos de formulario en la misma respuesta
    Ahora un tool_call puede ser ConductResearch, ResearchComplete o think_tool.
    Sepáralos por nombre (como en el 13.2):
        pedidos     = [c for c in ultimo.tool_calls if c["name"] == "ConductResearch"]
        reflexiones = [c for c in ultimo.tool_calls if c["name"] == "think_tool"]

Ejecutar una tool a mano (M03 / M06)
    think_tool SÍ tiene código (devuelve "Reflexión registrada: ..."). Como aquí no
    hay ToolNode, la ejecutas tú:
        observacion = think_tool.invoke(r["args"])                  # r es el tool_call
        ToolMessage(content=observacion, tool_call_id=r["id"])      # y se la devuelves al LLM

Una ronda puede ser SOLO de pensar
    Si el supervisor solo pidió think_tool (sin ConductResearch), no hay nada que
    investigar: devuelves los ToolMessages de las reflexiones y vuelves a "supervisor"
    para que decida con la cabeza fría. ¡Cuidado con el freno "if not pedidos"! del
    16.0: ahora "no hay pedidos" NO significa "terminar".

ENUNCIADO

Copia tu Ejercicio_16_0 y haz estos cambios:

  1. Importa tool (from langchain_core.tools import tool) y define think_tool como en
     el Ejercicio 12, pero con un docstring para un SUPERVISOR: úsala ANTES de
     delegar (¿qué temas necesito?) y DESPUÉS de recibir notas (¿cubren el brief?).

  2. Añádela al bind_tools: [ConductResearch, ResearchComplete, think_tool].
     Y en SUPERVISOR_PROMPT, una línea más: "Usa think_tool antes de delegar y
     después de recibir notas, para razonar tu decisión."

  3. Sube MAX_RONDAS a 6: las rondas de pensar también cuentan.

  4. En supervisor_tools:
        a) separa los tool_calls en pedidos (ConductResearch) y reflexiones (think_tool)
        b) cambia el freno "if not pedidos" por "if not ultimo.tool_calls"
           (terminar solo si NO pidió NADA)
        c) ejecuta cada reflexión con think_tool.invoke(...), imprime el texto de la
           reflexión y crea su ToolMessage
        d) si no hay pedidos (solo pensó): vuelve a "supervisor" con esos ToolMessages
        e) si hay pedidos: lanza los investigadores como en el 16.0 y añade sus
           ToolMessages a la MISMA lista, antes del return

Salida esperada (los textos de las reflexiones varían):
    --- Ronda 1: el supervisor pide ['think_tool']
        💭 El brief solicita una comparación de los sistemas educativos de Finlandia, Japón y Chile...
    --- Ronda 2: el supervisor pide ['ConductResearch', 'ConductResearch', 'ConductResearch']
        Lanzando 3 investigadores a la vez...
        ✓ Sistema educativo de Finlandia: resultados académicos, horas de clase semanales y formación...
        ✓ Sistema educativo de Japón: ...
        ✓ Sistema educativo de Chile: ...
        (1.0 s)
    --- Ronda 3: el supervisor pide ['think_tool']
        💭 He recibido las notas sobre los sistemas educativos de Finlandia, Japón y Chile, cubriendo...
    --- Ronda 4: el supervisor pide ['ResearchComplete']
        Fin: el supervisor dio por terminada la investigación

    NOTAS ACUMULADAS (3): ...

Para pensar:
    - Ahora son 4 llamadas al LLM en vez de 2. ¿Qué ganas a cambio? Lee las dos
      reflexiones: ¿detectaría el supervisor si a las notas les faltara un país?
    - ¿Qué pasaría si en el paso 4b dejaras el freno del 16.0 ("if not pedidos")?
      Sigue la ronda 1 con la mano.

Ejecuta:  uv run python 07_deep_research/Ejercicio_16_1_supervisor_que_piensa.py
"""

import asyncio
import operator
import os
import time
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
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

@tool
def think_tool(reflection: str) -> str:
    """Úsala para pensar ANTES de delegar (¿qué temas necesito?) y DESPUÉS de recibir
    notas (¿cubren el brief? ¿qué falta?). Escribe tu razonamiento en reflection."""
    return f"Reflexión registrada: {reflection}"


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

model_supervisor = llm.bind_tools([ConductResearch, ResearchComplete, think_tool])

SUPERVISOR_PROMPT = """Eres un supervisor de investigación. Hoy es {date}.
Recibes un brief. Tú NO investigas: DELEGAS con ConductResearch, un formulario por tema.

Cómo trabajar:
- Divide el brief en POCOS temas independientes. En una comparación, un tema por cada
  elemento comparado (por ejemplo, un país), y ese tema debe incluir TODOS los aspectos
  que pide el brief. No hagas un tema por aspecto.
- Preguntas sencillas: 1 o 2 temas. Máximo {max_investigadores} temas por ronda.
- Cuando recibas las notas de los investigadores, revisa si cubren el brief.
  Si falta algo importante, delega SOLO lo que falta. Si ya está cubierto, usa ResearchComplete.
- No repitas un tema que ya tiene notas.
- No repitas un tema que ya tiene notas.
- Usa think_tool antes de delegar y después de recibir notas, para razonar tu decisión."""



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
    ultimo = state["messages"][-1]
    # 4a. Tres tipos de formulario, separados por nombre
    pedidos = [c for c in ultimo.tool_calls if c["name"] == "ConductResearch"]
    reflexiones = [c for c in ultimo.tool_calls if c["name"] == "think_tool"]
    termino = any(c["name"] == "ResearchComplete" for c in ultimo.tool_calls)

    # a) ¿Hay que parar?
    if termino:
        print("    Fin: el supervisor dio por terminada la investigación")
        return Command(goto=END, update={})
    if not ultimo.tool_calls:                        # 4b. terminar solo si no pidió NADA
        print("    Fin: el supervisor no pidió nada")
        return Command(goto=END, update={})
    if state["rondas"] >= MAX_RONDAS:
        print(f"    Fin: se alcanzó el límite de {MAX_RONDAS} rondas (freno duro)")
        return Command(goto=END, update={})

    # 4c. Las reflexiones se ejecutan a mano (M03/M06) y vuelven como ToolMessages
    tool_messages = []
    for r in reflexiones:
        observacion = think_tool.invoke(r["args"])
        print(f"    💭 {r['args']['reflection'][:110]}")
        tool_messages.append(ToolMessage(content=observacion, tool_call_id=r["id"]))

    # 4d. Si solo pensó, no hay nada que investigar: otra ronda
    if not pedidos:
        return Command(goto="supervisor", update={"messages": tool_messages})

    # b) Un investigador por formulario, todos a la vez (Ejercicio 15)
    print(f"    Lanzando {len(pedidos)} investigadores a la vez...")
    inicio = time.perf_counter()
    tareas = [app_investigador.ainvoke({"tema": c["args"]["research_topic"]}) for c in pedidos]
    resultados = await asyncio.gather(*tareas)

    # c) Las notas se añaden a la MISMA lista que las reflexiones (4e: sin tool_messages = [] aquí)
    for resultado, pedido in zip(resultados, pedidos):
        print(f"    ✓ {pedido['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=resultado["notas"], tool_call_id=pedido["id"]))
    print(f"    ({time.perf_counter() - inicio:.1f} s)")

    return Command(
        goto="supervisor",
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
