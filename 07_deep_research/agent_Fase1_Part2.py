"""
FASE 1 — PASO 2: el LLM escribe el research brief                  (usa LLM)
=====================================================================================
DÓNDE ESTÁS

    START → clarificar ──(falta información)──► END
                       └─(todo claro)─────────► escribir_brief → END
                                                  ▲
                                                  └── AHORA: este nodo usa el LLM

    Paso 1 ✅  clarificar decide con el LLM y Command      (agent_Part1.py)
    Paso 2 ⬅️  escribir_brief convierte la conversación en un brief
    Paso 3     conversación de varios turnos con checkpointer e input()

¿QUÉ ES EL RESEARCH BRIEF?

    Es la conversación con el usuario convertida en UNA pregunta de investigación
    detallada. En la Fase 2 el investigador solo recibirá este texto, no la
    conversación, así que tiene que incluir TODO lo que pidió el usuario.

    Usuario:  "Investiga a escritores latinoamericanos que hayan ganado un Nobel"
    Brief:    "Quiero conocer a los escritores latinoamericanos que han ganado el
               Premio Nobel de Literatura: quiénes son, en qué año lo ganaron y por
               qué obras. No he indicado un periodo, así que considera todos."

CONCEPTOS (ninguno nuevo)

Structured output (M07) -> igual que en clarificar, pero con otra clase
    class ResearchQuestion(BaseModel):
        research_brief: str = Field(description="...")

    modelo_brief = model.with_structured_output(ResearchQuestion)
    respuesta = modelo_brief.invoke([HumanMessage(content=prompt)])
    respuesta.research_brief      # el texto del brief

Nodo normal vs. nodo con Command
    clarificar      -> decide el camino   -> devuelve Command(goto=..., update=...)
    escribir_brief  -> siempre va a END   -> devuelve un dict: {"research_brief": ...}

ENUNCIADO — PASO 2

  1. Crea el schema ResearchQuestion(BaseModel) con un solo campo:
        research_brief: str  -> describe que es UNA pregunta de investigación
                                detallada, en primera persona y en el idioma del usuario

  2. Usa este prompt (cópialo tal cual):

        BRIEF_PROMPT = '''Estos son los mensajes de la conversación con el usuario:
        <Mensajes>
        {messages}
        </Mensajes>

        Hoy es {date}.

        Convierte esta conversación en UNA pregunta de investigación detallada.
        - Incluye TODOS los detalles y preferencias que dio el usuario.
        - No inventes preferencias que el usuario no dijo. Si falta un detalle
          (fechas, presupuesto, fuentes...), di que queda abierto.
        - Escríbela en primera persona, como si la hiciera el usuario.
        - Escríbela en el mismo idioma que el usuario.'''

  3. Cambia el nodo escribir_brief para que use el LLM:
        - convierte state["messages"] a texto (la misma línea que en clarificar)
        - rellena BRIEF_PROMPT con .format(messages=..., date=...)
        - llama al modelo con structured output (ResearchQuestion)
        - devuelve {"research_brief": respuesta.research_brief}

  4. El grafo NO cambia.

Prueba con la petición del Nobel e imprime resultado.get("research_brief").
Revisa el brief:
    - ¿Incluye todo lo que pediste?
    - ¿Inventó alguna preferencia que no dijiste?

Para pensar: los mensajes que lee escribir_brief incluyen la verificación que
guardó clarificar ("Voy a investigar..."). ¿Por qué? (Pista: el update del
Command se aplica ANTES de que corra el siguiente nodo — Ejercicio 10.2.)

Ejecuta:  uv run python 07_deep_research/agent_Fase1_Part2.py
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

class ResearchQuestion(BaseModel):
    """Decide si hay que hacerle una pregunta al usuario antes de investigar."""
    research_brief: str = Field(description="Una pregunta de investigación detallada, en primera persona y en el idioma del usuario")

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

BRIEF_PROMPT = '''Estos son los mensajes de la conversación con el usuario:
<Mensajes>
{messages}
</Mensajes>

Hoy es {date}.

Convierte esta conversación en UNA pregunta de investigación detallada.
- Incluye TODOS los detalles y preferencias que dio el usuario.
- No inventes preferencias que el usuario no dijo. Si falta un detalle
    (fechas, presupuesto, fuentes...), di que queda abierto.
- Escríbela en primera persona, como si la hiciera el usuario.
- Escríbela en el mismo idioma que el usuario.'''

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
    texto = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = BRIEF_PROMPT.format(messages=texto, date=get_today())
    modelo_brief = model.with_structured_output(ResearchQuestion)
    respuesta = modelo_brief.invoke([HumanMessage(content=prompt)])
    return {"research_brief": respuesta.research_brief}
        
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
