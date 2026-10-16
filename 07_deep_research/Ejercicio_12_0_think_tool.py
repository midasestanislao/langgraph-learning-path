"""
EJERCICIO 12 — think_tool: que el agente piense entre búsquedas        (usa LLM)
=====================================================================================
CONCEPTOS

think_tool -> una herramienta que NO HACE NADA: devuelve lo mismo que recibe.
    @tool
    def think_tool(reflection: str) -> str:
        \"\"\"Úsala DESPUÉS de cada búsqueda para reflexionar...\"\"\"
        return f"Reflexión registrada: {reflection}"

¿Para qué sirve una herramienta que no hace nada?
    Para OBLIGAR al LLM a escribir su razonamiento entre búsquedas:
        ¿qué encontré?  ¿qué me falta?  ¿busco más o ya puedo responder?
    Ese razonamiento queda guardado en el historial (como un ToolMessage), así que
    en la siguiente vuelta el LLM lo vuelve a leer y decide mejor.

    Sin think_tool:  buscar → buscar → buscar → responder      (a ciegas)
    Con think_tool:  buscar → pensar → buscar → pensar → responder

El docstring es la instrucción (M06)
    El LLM no ve el código de think_tool, solo su nombre y su docstring.
    Por eso el docstring dice CUÁNDO usarla ("después de cada búsqueda").
    El system prompt lo refuerza.

Todo lo demás es tu Ejercicio 8: @tool, bind_tools, ToolNode, should_continue.

ENUNCIADO (ejemplo resuelto)

Un agente que responde una pregunta buscando en internet, y que reflexiona con
think_tool después de cada búsqueda.

    START → our_agent ──(pidió herramientas)──► tools ──► our_agent   (bucle)
                      └─(ya responde)─────────► END

    Herramientas: search (DDGS, como en tu Ejercicio 8) y think_tool (NUEVA)

Al ejecutarlo, fíjate en los mensajes "Tool Message — think_tool": ahí está lo que
pensó el agente antes de decidir el siguiente paso.

Ejecuta:  uv run python 07_deep_research/Ejercicio_12_0_think_tool.py
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


# 1. El estado: igual que en el Ejercicio 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
).bind_tools(tools)


# 4. El system prompt refuerza CUÁNDO usar cada herramienta
SYSTEM_PROMPT = """Eres un investigador. Responde la pregunta del usuario buscando en internet.
- Después de CADA búsqueda, usa think_tool para reflexionar antes de seguir.
- Cuando tengas suficiente información, responde sin llamar más herramientas.
Responde en español."""


def model_call(state: AgentState) -> dict:
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


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


inputs = {"messages": [HumanMessage("¿Qué escritores latinoamericanos han ganado el Premio Nobel de Literatura?")]}
print_stream(app.stream(inputs, stream_mode="values"))
