"""
EJERCICIO 12.1 — Ponerle un límite al agente                         (usa LLM)
=====================================================================================
CONCEPTOS

El problema
    En el Ejercicio 12 el agente decide SOLO cuándo parar. Puede buscar 2 veces...
    o 10. Cada vuelta es una llamada al LLM (dinero y tiempo). Un agente de
    investigación siempre necesita un freno.

Contador en el estado -> un campo que cada vuelta suma 1
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        vueltas: int

    def model_call(state):
        ...
        return {"messages": [response], "vueltas": state["vueltas"] + 1}

    (Es lo mismo que el count de tu Ejercicio_1, pero junto a messages.)

Un modelo CON herramientas y otro SIN herramientas
    llm = ChatOpenAI(...)                  # el modelo "normal"
    model_con_tools = llm.bind_tools(tools)
    model_sin_tools = llm                  # sin bind_tools: NO puede pedir herramientas

    Si llamas a model_sin_tools, el LLM no tiene más remedio que RESPONDER con lo
    que ya encontró. Así el agente siempre termina con una respuesta, no a medias.

ENUNCIADO

Copia tu Ejercicio 12 y añádele un límite de vueltas.

    START → our_agent ──(pidió herramientas)──► tools ──► our_agent
                      └─(ya responde)─────────► END

    El grafo NO cambia. Cambia solo lo que pasa DENTRO de our_agent:

        vueltas < MAX_VUELTAS   → usa model_con_tools   (puede buscar y pensar)
        vueltas >= MAX_VUELTAS  → usa model_sin_tools   (tiene que responder)

  1. Añade vueltas: int al estado.
  2. Crea llm, model_con_tools y model_sin_tools (ver CONCEPTOS).
  3. Crea la constante MAX_VUELTAS = 2.
  4. En model_call:
        - elige el modelo según state["vueltas"]
        - imprime en qué vuelta va y si usa herramientas o no
        - devuelve el mensaje Y vueltas + 1
  5. should_continue y el grafo NO cambian.
  6. En el invoke, empieza con el contador a 0:
        app.invoke({"vueltas": 0, "messages": [HumanMessage("...")]})
     Al final, imprime resultado["vueltas"] y la respuesta.

Salida esperada con MAX_VUELTAS = 2 (las herramientas pueden variar):
    --- vuelta 1 (con herramientas) -> pide: ['search']
    --- vuelta 2 (con herramientas) -> pide: ['think_tool']
    --- vuelta 3 (SIN herramientas) -> pide: []
    Vueltas: 3
    Los escritores latinoamericanos que han ganado el Premio Nobel son: ...

    Pista para el print: response.tool_calls es una lista de dicts (M03);
    [c["name"] for c in response.tool_calls] te da los nombres.

Para pensar: prueba con MAX_VUELTAS = 0, 2 y 6 y la misma pregunta.
    - ¿Con 0 también responde? ¿De dónde saca la información?
    - ¿Con 6 usa las 6 vueltas, o termina antes por su cuenta?

Ejecuta:  uv run python 07_deep_research/Ejercicio_12_1_limite_vueltas.py
"""
import os
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

MAX_VUELTAS = 2

# 1. El estado: igual que en el Ejercicio 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    vueltas: int


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
SYSTEM_PROMPT = """Eres un investigador. Responde la pregunta del usuario buscando en internet.
- Después de CADA búsqueda, usa think_tool para reflexionar antes de seguir.
- Cuando tengas suficiente información, responde sin llamar más herramientas.
Responde en español."""


def model_call(state: AgentState) -> dict:
    if state["vueltas"] >= MAX_VUELTAS:
        modelo = model_sin_tools
    else:
        modelo = model_con_tools

    response = modelo.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
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


inputs = {"vueltas": 0, "messages": [HumanMessage("¿Qué escritores latinoamericanos han ganado el Premio Nobel de Literatura?")]}
print_stream(app.stream(inputs, stream_mode="values"))

resultado = app.invoke(inputs)
print("Vueltas:", resultado["vueltas"])

