"""
FASE 1 — PASO 3: una conversación de varios turnos                  (usa LLM)
=====================================================================================
DÓNDE ESTÁS

    Paso 1 ✅  clarificar decide con el LLM y Command          (agent_Part1.py)
    Paso 2 ✅  escribir_brief escribe el brief con el LLM       (agent_Part2.py)
    Paso 3 ⬅️  el usuario puede RESPONDER a la pregunta y seguir

    Hasta ahora, si el agente preguntaba algo, el programa terminaba y no podías
    contestar. Al terminar este paso funcionará así:

        Usuario: Investiga escritores
        IA:      ¿Sobre qué aspecto específico te gustaría que investigue?
        Usuario: Latinoamericanos que hayan ganado el Nobel
        IA:      Voy a investigar...
        BRIEF:   Quiero investigar a los escritores latinoamericanos...

CONCEPTOS (ninguno nuevo)

Checkpointer (Ejercicio 9) -> guarda el estado entre una invocación y la siguiente
    from langgraph.checkpoint.memory import InMemorySaver

    app = graph.compile(checkpointer=InMemorySaver())

thread_id (Ejercicio 9) -> el "número de conversación". Mismo thread_id = misma
conversación; el checkpointer recuerda todos sus mensajes.
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"messages": [HumanMessage("...")]}, config)

Solo envías el mensaje NUEVO
    En tu M08 reenviabas todo el historial a mano. Aquí no: el checkpointer
    ya tiene los mensajes anteriores y add_messages añade el nuevo al final.

Bucle con input() (M08) -> pedir texto al usuario hasta que se cumpla algo
    while True:
        texto = input("Tú: ")
        ...
        if <condición>:
            break

ENUNCIADO — PASO 3

  1. Compila el grafo CON checkpointer (InMemorySaver).

  2. Crea el config con un thread_id fijo, por ejemplo "1".

  3. Sustituye las pruebas del final por un bucle while True:
        - pide el texto al usuario con input()
        - invoca el grafo con SOLO el mensaje nuevo y el config
        - imprime el último mensaje de la IA
        - si el resultado ya tiene "research_brief" -> imprímelo y sal del bucle
          (pista: "research_brief" in resultado)

  4. El grafo y los nodos NO cambian.

Prueba con esta conversación:
    Tú:  Investiga escritores
         → debe PREGUNTAR (no hay brief todavía, el bucle sigue)
    Tú:  Latinoamericanos que hayan ganado el premio Nobel
         → debe decir "Voy a investigar..." e imprimir el brief, y el programa termina

Revisa que el brief incluya lo de tu PRIMER mensaje (escritores) y lo de tu
SEGUNDO mensaje (latinoamericanos, Nobel). Si es así, el checkpointer funciona.

Para pensar: ¿qué pasaría si usaras un thread_id distinto en cada vuelta del bucle?
(Pista: tu Ejercicio 9, turno 3.)

Ejecuta:  uv run python es/07_deep_research/agent_Fase1_Part3.py
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
    """Pregunta de investigación detallada que guiará al investigador."""
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
- No añadas objetivos que el usuario no pidió (por ejemplo "el impacto" o
  "la influencia" si no los mencionó).
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

app = graph.compile(checkpointer=InMemorySaver())

def main():
    config = {"configurable": {"thread_id": "1"}}

    while True:
        texto = input("User: ").strip()
        if not texto:
            continue
        elif texto.lower() == "exit":
            break

        resultado = app.invoke({"messages": [HumanMessage(content=texto)]}, config)
        print(f"AI: {resultado['messages'][-1].content}")

        if "research_brief" in resultado:
            print(f"\nBRIEF:\n{resultado['research_brief']}") 
            break






if __name__ == "__main__":
    main()

# resultado = app.invoke({"messages": [HumanMessage('Investiga escritores')]})
# print(resultado["messages"][-1].content)
# print(resultado.get("research_brief"))
# resultado = app.invoke({"messages": [HumanMessage('Investiga a escritores latinoamericanos que hayan ganado un premio nobel')]})
# print(resultado["messages"][-1].content)
# print(resultado.get("research_brief"))
