"""
DEEP RESEARCH — el agente completo                                   (usa LLM)
=====================================================================================
DE DÓNDE SALE CADA PIEZA

    FASE 1  conversación → brief            agent_Fase1_Part3.py   (import app_scope)
    FASE 2  tema → notas con fuentes        agent_Fase2_Part2.py   (import app_investigador)
    FASE 4  supervisor que delega           Ejercicio_16_0 + 16_1  (copiado aquí, con think_tool)
    FASE 5  notas → informe final           NUEVO: el nodo escribir_informe

EL FLUJO COMPLETO

    main()
      │  FASE 1: bucle con input() hasta tener el brief   (como agent_Fase2_Part3.py)
      ▼
    brief ──► GRAFO DE INVESTIGACIÓN (una sola app):
                START → supervisor ⇄ supervisor_tools ──► escribir_informe → END
                                          │ por cada ConductResearch:
                                          └─► app_investigador (REAL, Fase 2) en paralelo

    Diferencia con el 16.0: cuando el supervisor termina, supervisor_tools ya no va
    a END sino a "escribir_informe", que convierte todas las notas en un informe.

LO ÚNICO NUEVO: el nodo escribir_informe
    Recibe el brief y TODAS las notas (con sus [n] y sus Fuentes) y le pide al LLM
    un informe en Markdown que responda al brief citando las fuentes. Es texto libre
    (lo lee una persona), así que model sin herramientas y .content, como en comprimir.
    Lo guarda en outputs/informe_<fecha>.md, como hacías en el boletín.

Ejecuta:  uv run python 07_deep_research/deep_research.py
    User: Compara el sistema educativo de Finlandia, Japón y Chile: resultados académicos y formación docente
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

# ============ 1. LAS FASES QUE YA CONSTRUISTE ============
from agent_Fase1_Part3 import app as app_scope              # Fase 1: conversación → brief
from agent_Fase2_Part2 import app as app_investigador       # Fase 2: tema → notas con fuentes

load_dotenv()

MAX_RONDAS = 6              # freno duro del supervisor (las rondas de pensar también cuentan)
MAX_INVESTIGADORES = 3      # freno blando: se lo decimos al LLM en el prompt


def get_today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ============ 2. LAS HERRAMIENTAS DEL SUPERVISOR (Ejercicios 13 y 16.1) ============
class ConductResearch(BaseModel):
    """Delega la investigación de UN tema a un investigador. Úsala una vez por cada tema distinto."""
    research_topic: str = Field(description="El tema a investigar, descrito con detalle en una frase, en el idioma del brief.")


class ResearchComplete(BaseModel):
    """Úsala cuando las notas recibidas ya cubren el brief y no hace falta investigar más."""


@tool
def think_tool(reflection: str) -> str:
    """Úsala para pensar ANTES de delegar (¿qué temas necesito?) y DESPUÉS de recibir
    notas (¿cubren el brief? ¿qué falta?). Escribe tu razonamiento en reflection."""
    return f"Reflexión registrada: {reflection}"


model_supervisor = llm.bind_tools([ConductResearch, ResearchComplete, think_tool])

SUPERVISOR_PROMPT = """Eres un supervisor de investigación. Hoy es {date}.
Recibes un brief. Tú NO investigas: DELEGAS con ConductResearch, un formulario por tema.

Cómo trabajar:
- Usa think_tool antes de delegar y después de recibir notas, para razonar tu decisión.
- Divide el brief en POCOS temas independientes. En una comparación, un tema por cada
  elemento comparado (por ejemplo, un país), y ese tema debe incluir TODOS los aspectos
  que pide el brief. No hagas un tema por aspecto.
- Preguntas sencillas: 1 o 2 temas. Máximo {max_investigadores} temas por ronda.
- Cuando recibas las notas de los investigadores, revisa si cubren el brief.
  Si falta algo importante, delega SOLO lo que falta. Si ya está cubierto, usa ResearchComplete.
- No repitas un tema que ya tiene notas."""

INFORME_PROMPT = """Eres un redactor. Hoy es {date}.
Escribe un informe en Markdown que responda a este brief:

{brief}

Usa SOLO la información de estas notas (cada nota trae sus fuentes numeradas):

{notas}

Reglas:
- Estructura: título, una introducción breve, una sección por tema o elemento comparado,
  y una conclusión que responda directamente al brief.
- Cita las fuentes con [n] después de cada dato, y termina con una sección "Fuentes"
  con la lista de URLs. Si dos notas repiten la misma URL, inclúyela una sola vez.
- Si las notas no cubren algo que pide el brief, dilo explícitamente. No inventes.
- Escribe en el idioma del brief."""


# ============ 3. EL ESTADO ============
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # brief + formularios + notas (ToolMessages)
    notas: Annotated[list[str], operator.add]                 # se acumulan ronda a ronda
    rondas: int
    informe: str
    archivo: str


# ============ 4. NODO supervisor (Ejercicio 16) ============
async def supervisor(state: ResearchState) -> Command[Literal["supervisor_tools"]]:
    prompt = SUPERVISOR_PROMPT.format(date=get_today(), max_investigadores=MAX_INVESTIGADORES)
    response = await model_supervisor.ainvoke([SystemMessage(content=prompt)] + state["messages"])

    print(f"\n--- Ronda {state['rondas'] + 1}: el supervisor pide {[c['name'] for c in response.tool_calls]}")
    return Command(
        goto="supervisor_tools",
        update={"messages": [response], "rondas": state["rondas"] + 1},
    )


# ============ 5. NODO supervisor_tools (Ejercicio 16.1, con investigadores REALES) ============
async def supervisor_tools(state: ResearchState) -> Command[Literal["supervisor", "escribir_informe"]]:
    ultimo = state["messages"][-1]
    pedidos = [c for c in ultimo.tool_calls if c["name"] == "ConductResearch"]
    reflexiones = [c for c in ultimo.tool_calls if c["name"] == "think_tool"]
    termino = any(c["name"] == "ResearchComplete" for c in ultimo.tool_calls)

    # a) ¿Hay que parar? Si sí, ya no vamos a END: vamos a escribir el informe
    if termino:
        print("    Fin: el supervisor dio por terminada la investigación")
        return Command(goto="escribir_informe", update={})
    if not ultimo.tool_calls:
        print("    Fin: el supervisor no pidió nada")
        return Command(goto="escribir_informe", update={})
    if state["rondas"] >= MAX_RONDAS:
        print(f"    Fin: se alcanzó el límite de {MAX_RONDAS} rondas (freno duro)")
        return Command(goto="escribir_informe", update={})

    # b) Las reflexiones se ejecutan a mano (M03) y vuelven como ToolMessages
    tool_messages = []
    for r in reflexiones:
        print(f"    💭 {r['args']['reflection'][:120]}")
        tool_messages.append(ToolMessage(content=think_tool.invoke(r["args"]), tool_call_id=r["id"]))
    if not pedidos:                                   # solo pensó: otra ronda sin investigar
        return Command(goto="supervisor", update={"messages": tool_messages})

    # c) Un investigador REAL (Fase 2) por formulario, todos a la vez (Ejercicio 15)
    print(f"    Lanzando {len(pedidos)} investigadores a la vez...")
    inicio = time.perf_counter()
    tareas = [
        app_investigador.ainvoke({"vueltas": 0, "messages": [HumanMessage(content=c["args"]["research_topic"])]})
        for c in pedidos
    ]
    resultados = await asyncio.gather(*tareas)

    # d) Sus notas vuelven al supervisor como ToolMessages y se acumulan en el estado
    for resultado, pedido in zip(resultados, pedidos):
        print(f"    ✓ {pedido['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=resultado["notas"], tool_call_id=pedido["id"]))
    print(f"    ({time.perf_counter() - inicio:.0f} s)")

    return Command(
        goto="supervisor",
        update={"messages": tool_messages, "notas": [r["notas"] for r in resultados]},
    )


# ============ 6. NODO escribir_informe (NUEVO: Fase 5) ============
def escribir_informe(state: ResearchState) -> dict:
    brief = state["messages"][0].content                 # el primer mensaje es el brief
    prompt = INFORME_PROMPT.format(
        date=get_today(),
        brief=brief,
        notas="\n\n---\n\n".join(state["notas"]),
    )
    print("\n📝 Escribiendo el informe...")
    informe = llm.invoke([HumanMessage(content=prompt)]).content

    os.makedirs("outputs", exist_ok=True)
    ruta = f"outputs/informe_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(informe)

    return {"informe": informe, "archivo": ruta}


# ============ 7. EL GRAFO DE INVESTIGACIÓN ============
graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor)
graph.add_node("supervisor_tools", supervisor_tools)
graph.add_node("escribir_informe", escribir_informe)
graph.add_edge(START, "supervisor")
# supervisor ⇄ supervisor_tools y supervisor_tools → escribir_informe los deciden los Command
graph.add_edge("escribir_informe", END)
app_research = graph.compile()


# ============ 8. main(): FASE 1 con input(), y luego el grafo de investigación ============
async def main():
    config = {"configurable": {"thread_id": "1"}}
    brief = None

    while True:
        texto = input("User: ").strip()
        if not texto:
            continue
        elif texto.lower() == "exit":
            break

        resultado = app_scope.invoke({"messages": [HumanMessage(content=texto)]}, config)
        print(f"AI: {resultado['messages'][-1].content}")

        if "research_brief" in resultado:
            brief = resultado["research_brief"]
            print(f"\nBRIEF:\n{brief}")
            break

    if brief is None:
        print("Sin brief, no hay nada que investigar. ¡Hasta luego!")
        return

    print("\n🔎 Investigando (esto tarda unos minutos)...")
    inicio = time.perf_counter()
    resultado = await app_research.ainvoke(
        {"messages": [HumanMessage(content=brief)], "notas": [], "rondas": 0}
    )

    print(f"\n{'=' * 70}\n{resultado['informe']}\n{'=' * 70}")
    print(f"\n💾 Informe guardado en {resultado['archivo']}  ({time.perf_counter() - inicio:.0f} s en total)")


if __name__ == "__main__":
    asyncio.run(main())
