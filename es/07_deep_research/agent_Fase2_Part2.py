"""
FASE 2 — PASO 2: comprimir la investigación                         (usa LLM)
=====================================================================================
DÓNDE ESTÁS

    brief → our_agent ⇄ tools ──► comprimir ──► END
                                   ▲
                                   └── PASO 2 (AHORA): un nodo nuevo al final

    Paso 1 ✅  el investigador busca, piensa y responde      (agent_Fase2_Part1.py)
    Paso 2 ⬅️  comprimir: convertir lo investigado en notas limpias con fuentes
    Paso 3     unir la Fase 1 y la Fase 2

    El código de abajo es una COPIA de tu Paso 1.

CONCEPTOS

¿Por qué comprimir?
    Al terminar, state["messages"] tiene de todo: el brief, llamadas a
    herramientas, resultados de búsqueda repetidos, reflexiones de think_tool y la
    respuesta final. En la Fase 4 un supervisor lanzará VARIOS investigadores; si
    cada uno le devuelve todo ese historial, el contexto explota.
    Por eso cada investigador entrega solo sus NOTAS: limpias, sin repetir y con
    las fuentes numeradas.

¿Por qué no usar la respuesta final?
    La respuesta final del agente es un resumen: puede dejar fuera datos que SÍ
    encontró en las búsquedas. comprimir trabaja sobre los resultados CRUDOS de
    search, así no se pierde nada.

filter_messages (M04) -> quedarte solo con los resultados de search
    from langchain_core.messages import filter_messages

    resultados = filter_messages(state["messages"], include_names=["search"])
    # → solo los ToolMessage de search (sin think_tool, sin mensajes de la IA)

El brief está en el primer mensaje
    state["messages"][0].content    # el HumanMessage con el que empezó todo

Texto libre, no structured output
    Las notas las leerá otro LLM (el supervisor o el escritor del informe), no tu
    código. Por eso basta con model_sin_tools.invoke(...).content (M07, regla:
    si los datos los usa TU CÓDIGO → structured output; si los lee un LLM o una
    persona → texto).

ENUNCIADO — PASO 2

  1. Añade notas: str al estado.

  2. Usa este prompt (cópialo tal cual):

        COMPRESS_PROMPT = '''Eres un asistente que limpia los resultados de una investigación.
        Hoy es {date}.

        Brief de la investigación:
        {brief}

        Resultados de las búsquedas:
        {resultados}

        Escribe unas NOTAS LIMPIAS:
        - Conserva TODOS los datos relevantes para el brief (nombres, fechas, cifras).
          No resumas de más.
        - Elimina lo repetido y lo que no tenga que ver con el brief.
        - Después de cada dato, indica su fuente con un número entre corchetes, ej. [1].
        - Al final, una sección "Fuentes" con la lista numerada de URLs.
        - No inventes nada que no esté en los resultados.'''

  3. Crea el nodo comprimir(state) -> dict:
        - saca los resultados de search con filter_messages
        - únelos en un solo texto ("

".join(m.content for m in resultados))
        - rellena COMPRESS_PROMPT con date, brief y resultados
        - llama a model_sin_tools con un HumanMessage y quédate con .content
        - devuelve {"notas": ...}

  4. Cambia el grafo: cuando el agente termina ya no va a END, va a "comprimir".
        - en add_conditional_edges, "end" apunta a "comprimir"
        - añade el nodo y el edge comprimir → END

  5. Al final, imprime resultado["notas"]. (Usa app.invoke; si quieres ver el
     proceso, deja el print de las vueltas que ya tienes en model_call.)

Qué revisar en la salida (imprime también resultado["messages"][-1].content):
    - ¿Cada dato lleva su [número] y al final hay una lista de Fuentes?
    - ¿Las notas tienen los 6 escritores? Compara con la respuesta final del agente.
      Si a las notas les falta alguno que SÍ está en la respuesta final, busca su
      nombre en los resultados de search: ¿aparece? Si no aparece, ¿de dónde sacó
      el agente ese dato? (Pista: "No inventes nada que no esté en los resultados".)

Para pensar: imprime len(str(resultado["messages"])) y len(resultado["notas"]).
¿Cuántas veces más pequeño es lo que entregarías al supervisor?

Ejecuta:  uv run python es/07_deep_research/agent_Fase2_Part2.py
"""
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, filter_messages
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
    notas: str


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

COMPRESS_PROMPT = '''Eres un asistente que limpia los resultados de una investigación.
Hoy es {date}.

Brief de la investigación:
{brief}

Resultados de las búsquedas:
{resultados}

Escribe unas NOTAS LIMPIAS:
- Conserva TODOS los datos relevantes para el brief (nombres, fechas, cifras).
    No resumas de más.
- Elimina lo repetido y lo que no tenga que ver con el brief.
- Después de cada dato, indica su fuente con un número entre corchetes, ej. [1].
- Al final, una sección "Fuentes" con la lista numerada de URLs.
- No inventes nada que no esté en los resultados.'''

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

def comprimir(state: AgentState) -> dict:
    resultados = filter_messages(state["messages"], include_names=["search"])
    texto_resultados = "\n\n".join(m.content for m in resultados)
    prompt = COMPRESS_PROMPT.format(
        date=get_today(),
        brief=state["messages"][0].content, # el primer mensaje es el brief
        resultados=texto_resultados
    )
    respuesta = model_sin_tools.invoke([HumanMessage(content=prompt)])
    return {"notas": respuesta.content}


def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# 5. El grafo: idéntico al del Ejercicio 8
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("comprimir", comprimir)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": "comprimir"})
graph.add_edge("tools", "our_agent")
graph.add_edge("comprimir", END)

app = graph.compile()


# 6. Probarlo: imprime cada mensaje a medida que se genera
def print_stream(stream):
    for event in stream:
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    inputs = {"vueltas": 0, "messages": [HumanMessage(BRIEF)]}
    resultado = app.invoke(inputs)
    print("Vueltas:", resultado["vueltas"])
    print("\n\nNotas:", resultado["notas"])
    print("\n\nResultado:", resultado["messages"][-1].content)

