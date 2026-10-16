"""
FASE 2 — PASO 1: el investigador                                    (usa LLM)
=====================================================================================
DÓNDE ESTÁS

    FASE 1 ✅  conversación → brief                     (agent_Part1/2/3.py)
    FASE 2     brief → investigar ⇄ tools → comprimir → notas con fuentes
                       ▲
                       └── PASO 1 (AHORA): el agente que investiga a partir del brief

    El código de abajo es una COPIA de tu Ejercicio 12.1. Ya tiene casi todo:
    search, think_tool, el bucle con should_continue y el límite de vueltas.

CONCEPTOS

Dos frenos: uno blando y uno duro
    Blando (prompt):      "preguntas sencillas: 2-3 búsquedas; complejas: hasta 5"
                          El LLM lo respeta casi siempre, pero no está garantizado.
    Duro (MAX_VUELTAS):   tu contador del 12.1. Pase lo que pase, en la vuelta
                          MAX_VUELTAS el agente responde sin herramientas.

    El proyecto de referencia usa los dos: el prompt guía y el código garantiza.

El brief es la entrada
    En el 12.1 el agente recibía una pregunta corta. Ahora recibe el brief de la
    Fase 1: un texto largo con todo lo que el usuario quiere saber.

        inputs = {"vueltas": 0, "messages": [HumanMessage(content=BRIEF)]}

El system prompt de investigador
    Le explica CÓMO investigar: leer el brief, buscar de lo general a lo concreto,
    pensar después de cada búsqueda, cuándo parar y cómo entregar la respuesta.

ENUNCIADO — PASO 1

  1. Añade el helper get_today() de la Fase 1 (devuelve "13/10/2026").

  2. Sustituye SYSTEM_PROMPT por este RESEARCHER_PROMPT (cópialo tal cual):

        RESEARCHER_PROMPT = '''Eres un asistente de investigación. Hoy es {date}.
        Vas a recibir un brief con lo que el usuario quiere investigar.
        Usa search para buscar información y think_tool para pensar entre búsquedas.

        Cómo investigar:
        1. Lee el brief: ¿qué información concreta necesitas?
        2. Empieza con búsquedas amplias y luego hazlas más específicas.
        3. Después de CADA búsqueda, usa think_tool: ¿qué encontré? ¿qué me falta?
           ¿sigo buscando o ya puedo responder?
        4. Para cuando puedas responder bien el brief, o cuando tus últimas
           2 búsquedas no aporten nada nuevo.

        Límites:
        - Preguntas sencillas: 2-3 búsquedas como máximo.
        - Preguntas complejas: hasta 5 búsquedas.

        Respuesta final: en el idioma del brief, con la información encontrada
        y las URLs de las fuentes que usaste.'''

     En model_call, rellénalo con .format(date=get_today()) antes de crear el
     SystemMessage.

  3. Sube MAX_VUELTAS a 10 (5 búsquedas + 5 reflexiones como máximo).

  4. Crea una constante BRIEF con un brief como el que genera tu Fase 1:

        BRIEF = '''Quiero investigar a los escritores latinoamericanos que han ganado
        el Premio Nobel de Literatura: quiénes son, de qué país, en qué año lo ganaron
        y por qué obras. No tengo preferencia de fuentes.'''

     y úsalo en inputs en lugar de la pregunta del Nobel.

  5. Al terminar, imprime cuántas vueltas usó y la respuesta final
     (resultado["vueltas"] y resultado["messages"][-1].content).

  El grafo, search, think_tool y should_continue NO cambian.

Qué revisar en la salida:
    - ¿Cuántas búsquedas hizo? ¿Respetó el límite blando del prompt?
    - ¿La respuesta final trae las URLs de las fuentes?
    - ¿Responde TODO lo del brief (quién, país, año, obras)?

Para pensar: cambia el brief por uno complejo (por ejemplo, "compara el sistema
educativo de Finlandia y Japón: resultados, horas de clase y formación docente").
¿Hace más búsquedas que con el del Nobel?

Ejecuta:  uv run python es/07_deep_research/agent_Fase2_Part1.py
"""
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

MAX_VUELTAS = 10

# 1. El estado: igual que en el Ejercicio 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    vueltas: int

# Helpers
def get_today() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y")


# 2. Herramienta de búsqueda: tu Ejercicio 8, con 3 resultados y manejo de errores

@tool
def search(query: str) -> str:
    """Busca información actual en internet. Devuelve hasta 3 resultados."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"La búsqueda falló: {e}"

    if not results:
        return "No se encontraron resultados."

    return "\n\n".join(f"{r['title']}\n{r['body']}\n({r['href']})" for r in results)


# 3. LO NUEVO: think_tool. No busca nada, no calcula nada: solo "registra" lo que pensó.
@tool
def think_tool(reflection: str) -> str:
    """Úsala DESPUÉS de cada búsqueda para reflexionar antes de seguir:
    qué información encontraste, qué te falta todavía, y si necesitas buscar
    más o ya puedes responder."""
    return f"Reflexión registrada: {reflection}"


tools = [search, think_tool]

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

model_con_tools = model.bind_tools(tools)
model_sin_tools = model


# 4. El system prompt refuerza CUÁNDO usar cada herramienta
RESEARCHER_PROMPT = '''Eres un asistente de investigación. Hoy es {date}.
    Vas a recibir un brief con lo que el usuario quiere investigar.
    Usa search para buscar información y think_tool para pensar entre búsquedas.

    Cómo investigar:
    1. Lee el brief: ¿qué información concreta necesitas?
    2. Empieza con búsquedas amplias y luego hazlas más específicas.
    3. Después de CADA búsqueda, usa think_tool: ¿qué encontré? ¿qué me falta?
        ¿sigo buscando o ya puedo responder?
    4. Para cuando puedas responder bien el brief, o cuando tus últimas
        2 búsquedas no aporten nada nuevo.

    Límites:
    - Preguntas sencillas: 2-3 búsquedas como máximo.
    - Preguntas complejas: hasta 5 búsquedas.

    Respuesta final: en el idioma del brief, con la información encontrada
    y las URLs de las fuentes que usaste.'''

BRIEF = '''Quiero investigar a los escritores latinoamericanos que han ganado
    el Premio Nobel de Literatura: quiénes son, de qué país, en qué año lo ganaron
    y por qué obras. No tengo preferencia de fuentes.'''

def model_call(state: AgentState) -> dict:
    if state["vueltas"] >= MAX_VUELTAS:
        modelo = model_sin_tools
    else:
        modelo = model_con_tools

    prompt = RESEARCHER_PROMPT.format(date=get_today())

    response = modelo.invoke([SystemMessage(content=prompt)] + state["messages"])
    print(f"--- vuelta {state['vueltas'] + 1} -> pide: {[c['name'] for c in response.tool_calls]}")
    return {"messages": [response], "vueltas": state['vueltas'] + 1}


def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# 5. El grafo: idéntico al del Ejercicio 8
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

app = graph.compile()


# 6. Probarlo: imprime cada mensaje a medida que se genera
def print_stream(stream):
    for event in stream:
        event["messages"][-1].pretty_print()


inputs = {"vueltas": 0, "messages": [HumanMessage(BRIEF)]}
print_stream(app.stream(inputs, stream_mode="values"))

# resultado = app.invoke(inputs)
# print("Vueltas:", resultado["vueltas"])
# print("Resultado:", resultado["messages"][-1].content)

