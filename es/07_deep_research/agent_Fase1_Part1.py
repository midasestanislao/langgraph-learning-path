"""
FASE 1 — Scoping: aclarar la petición antes de investigar           (usa LLM)
=====================================================================================
VISIÓN GENERAL (lo que tendrás al terminar la fase)

    START → clarificar ──(falta información)──► END             (le hace una PREGUNTA)
                       └─(todo claro)─────────► escribir_brief → END

    Paso 1 (AHORA): el nodo clarificar decide con el LLM y Command
    Paso 2:         escribir_brief convierte la conversación en un brief con el LLM
    Paso 3:         conversación de varios turnos con checkpointer e input()

CONCEPTOS (todos los practicaste ya)

Structured output (M07) -> el LLM devuelve un objeto, no texto
    modelo_decision = model.with_structured_output(ClarifyWithUser)
    decision = modelo_decision.invoke([HumanMessage(content=prompt)])
    decision.need_clarification   # True o False

Command (Ejercicio 10.3) -> decidir el camino y guardar un mensaje a la vez
    return Command(goto=END, update={"messages": [AIMessage(content="...")]})

MessagesState (M05) -> un estado que ya trae `messages` con el reducer add_messages
    class ScopeState(MessagesState):
        research_brief: str

Mensajes a texto (M01) -> un prompt solo acepta texto, el historial es una lista
    texto = "\\n".join(f"{m.type}: {m.content}" for m in state["messages"])

La fecha en el prompt (M06) -> el LLM no sabe qué día es hoy
    datetime.now().strftime("%d/%m/%Y")   # devuelve un str, ej. "12/10/2026"

ENUNCIADO — PASO 1

  1. Arregla get_today (abajo): debe DEVOLVER un str con la fecha (no la hora).

  2. Crea el schema ClarifyWithUser(BaseModel) con 3 campos, cada uno con su
     Field(description=...):
        need_clarification: bool   -> usa EXACTAMENTE esta descripción:
            "True SOLO si falta el tema, el lugar o el objetivo. Si la petición ya
             dice qué investigar, False. En caso de duda, False."
        question: str              -> la pregunta ("" si no hace falta)
        verification: str          -> mensaje de "empiezo a investigar" ("" si hace falta preguntar)

     (La descripción de need_clarification es importante: con una descripción
      simple, GPT-4o pregunta casi siempre, incluso con peticiones claras.)

  3. Crea el estado ScopeState(MessagesState) con research_brief: str

  4. Usa este prompt (cópialo tal cual):

        CLARIFY_PROMPT = '''Estos son los mensajes de la conversación con el usuario:
        <Mensajes>
        {messages}
        </Mensajes>

        Hoy es {date}.

        Decide si necesitas hacerle una pregunta antes de empezar a investigar.
        - Pregunta SOLO si falta el tema, el lugar o el objetivo de la investigación.
        - Si la petición ya dice qué investigar, NO preguntes: los detalles que falten
          los decidirá el investigador.
        - No preguntes datos que se pueden buscar en internet.
        - Si YA le hiciste una pregunta antes, no vuelvas a preguntar.
        - En caso de duda, NO preguntes.
        Responde en el mismo idioma que el usuario.'''

  5. Nodo clarificar(state) -> Command[Literal["escribir_brief", "__end__"]]:
        - convierte state["messages"] a texto
        - rellena CLARIFY_PROMPT con .format(messages=..., date=...)
        - llama al modelo con structured output (ClarifyWithUser)
        - si need_clarification -> goto END con un AIMessage de la pregunta
        - si no                 -> goto "escribir_brief" con un AIMessage de la verificación

  6. Nodo escribir_brief PROVISIONAL (en el Paso 2 lo hará el LLM):
        return {"research_brief": "(pendiente)"}

  7. Grafo: START → clarificar, escribir_brief → END. Nada sale de clarificar.

Prueba con dos peticiones (cada una en su propio invoke):
    vaga  = "Investiga cafés"
        → último mensaje: una PREGUNTA (¿qué ciudad? ¿qué criterio?...)
    clara = "Compara precio y calidad de las 5 cafeterías de especialidad mejor
             valoradas en Google Maps en Madrid en 2026"
        → último mensaje: la verificación, y research_brief = "(pendiente)"

    Imprime resultado["messages"][-1].content y resultado.get("research_brief").

Para pensar: cuando funcione, cambia la descripción de need_clarification por algo
simple ("Si hace falta preguntarle algo al usuario") y ejecuta la petición clara
3 veces. ¿Qué pasa? Acuérdate de lo que viste en M07.

Ejecuta:  uv run python es/07_deep_research/agent_Fase1_Part1.py
"""

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

def get_today() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y")

class ClarifyWithUser(BaseModel):
    """Decide si hay que hacerle una pregunta al usuario antes de investigar."""
    need_clarification: bool = Field(description="True SOLO si falta el tema, el lugar o el objetivo. Si la petición ya dice qué investigar, False. En caso de duda, False")
    question: str = Field(description="La pregunta para el usuario, breve y en su idioma ('' si no hace falta preguntar)")
    verification: str = Field(description="Mensaje de que empiezas a investigar, resumiendo lo que entendiste ('' si hace falta preguntar)")

class ScopeState(MessagesState):
    research_brief: str

CLARIFY_PROMPT = '''Estos son los mensajes de la conversación con el usuario:
<Mensajes>
{messages}
</Mensajes>

Hoy es {date}.

Decide si necesitas hacerle una pregunta antes de empezar a investigar.
- Pregunta SOLO si falta el tema, el lugar o el objetivo de la investigación.
- Si la petición ya dice qué investigar, NO preguntes: los detalles que falten
  los decidirá el investigador.
- No preguntes datos que se pueden buscar en internet.
- Si YA le hiciste una pregunta antes, no vuelvas a preguntar.
- En caso de duda, NO preguntes.
Responde en el mismo idioma que el usuario.'''

def clarificar(state: ScopeState) -> Command[Literal["escribir_brief", "__end__"]]:
    texto = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = CLARIFY_PROMPT.format(messages=texto, date=get_today())
    modelo_decision = model.with_structured_output(ClarifyWithUser)
    decision = modelo_decision.invoke([HumanMessage(content=prompt)])
    if decision.need_clarification:
        return Command(goto=END, update={"messages":[AIMessage(content=decision.question)]})
    else:
        return Command(goto="escribir_brief", update={"messages":[AIMessage(content=decision.verification)]})
    
def escribir_brief(state: ScopeState) -> dict:
    return {"research_brief": "(pendiente)"}
        
graph = StateGraph(ScopeState)
graph.add_node("clarificarIntencion", clarificar)
graph.add_node("escribir_brief", escribir_brief)

graph.add_edge(START, "clarificarIntencion")
graph.add_edge("escribir_brief", END)

app = graph.compile()

resultado = app.invoke({"messages": [HumanMessage('Investiga escritores')]})
print(resultado["messages"][-1].content)
print(resultado.get("research_brief"))
resultado = app.invoke({"messages": [HumanMessage('Investiga a escritores latinoamericanos que hayan ganado un premio nobel')]})
print(resultado["messages"][-1].content)
print(resultado.get("research_brief"))
