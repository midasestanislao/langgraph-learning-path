"""
EJERCICIO 13 — Una clase Pydantic como herramienta                    (usa LLM, 3 llamadas)
=====================================================================================
CONCEPTOS

El supervisor de la Fase 4 no busca nada él mismo: DELEGA. Para delegar, el LLM
necesita una "herramienta" que en realidad es solo un formulario a rellenar.

Una clase Pydantic como tool -> describe QUÉ datos quieres, no qué hacer con ellos
    from pydantic import BaseModel, Field

    class ConductResearch(BaseModel):
        \"\"\"Delega la investigación de UN tema a un investigador.\"\"\"
        research_topic: str = Field(description="El tema a investigar, con detalle.")

    class ResearchComplete(BaseModel):
        \"\"\"Indica que la investigación ya terminó.\"\"\"     # sin campos: es solo una señal

    model_con_tools = llm.bind_tools([ConductResearch, ResearchComplete])

    Igual que en M06: el LLM ve el NOMBRE de la clase, el DOCSTRING (cuándo usarla)
    y los Field (qué rellenar). El código de la clase no existe: no hay nada que ejecutar.
    (El curso de referencia le pone @tool encima a la clase; funciona igual con o sin él.)

Lo que devuelve el LLM -> tool_calls, igual que en M03
    response.tool_calls
    [{"name": "ConductResearch", "args": {"research_topic": "Precio del café en Madrid"}, "id": "call_1"},
     {"name": "ConductResearch", "args": {"research_topic": "Precio del café en Buenos Aires"}, "id": "call_2"}]

    El LLM decide CUÁNTAS veces la usa: 0, 1 o varias en la misma respuesta.
    Cada tool_call es una tarea para un investigador distinto.

Diferencia con with_structured_output (M07)
    with_structured_output(Clase) -> SIEMPRE devuelve exactamente UN objeto.
    bind_tools([Clase])           -> el LLM ELIGE si la usa, cuál, y cuántas veces.

ENUNCIADO (ejemplo resuelto)

Sin grafo: solo el modelo con herramientas y tres preguntas distintas.

    a) una comparación de dos ciudades  -> el LLM delega DOS investigaciones
    b) un tema simple                   -> delega UNA
    c) "ya tengo toda la información"   -> usa ResearchComplete

Salida esperada (los textos pueden variar):
    --- Pregunta a: 2 tool_calls
    ConductResearch {'research_topic': 'Precio del café en Madrid, España'}
       Investigador 1 recibe: Precio del café en Madrid, España
    ConductResearch {'research_topic': 'Precio del café en Buenos Aires, Argentina'}
       Investigador 2 recibe: Precio del café en Buenos Aires, Argentina
    --- Pregunta b: 1 tool_calls
    ...
    --- Pregunta c: 1 tool_calls
    ResearchComplete {}

Para pensar:
    - Nadie ejecuta ConductResearch: la clase no tiene código. Entonces, cuando tu
      programa ve un tool_call con name == "ConductResearch", ¿qué debería hacer?
      (Esa es la pregunta que resuelve el supervisor.)
    - ¿Qué pasa en la pregunta c si quitas ResearchComplete del bind_tools?

Ejecuta:  uv run python 07_deep_research/Ejercicio_13_0_pydantic_tool.py
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# 1. El modelo, como siempre
llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# 2. LO NUEVO: dos clases que son "formularios" para el LLM.
#    El docstring dice CUÁNDO usarla; el Field dice QUÉ escribir en cada campo.
class ConductResearch(BaseModel):
    """Delega la investigación de UN tema a un investigador. Úsala una vez por cada tema distinto."""
    research_topic: str = Field(description="El tema a investigar, descrito con detalle.")


class ResearchComplete(BaseModel):
    """Úsala cuando ya no haga falta investigar nada más."""


# 3. bind_tools acepta las clases directamente (no hace falta @tool)
model_con_tools = llm.bind_tools([ConductResearch, ResearchComplete])


# 4. Tres preguntas para ver cuántas veces decide usar cada herramienta
preguntas = {
    "a": "Compara el precio del café en Madrid y en Buenos Aires. Delega cada ciudad a un investigador distinto.",
    "b": "Investiga quién inventó el espresso.",
    "c": "Ya tengo toda la información que necesito sobre el espresso. No hace falta investigar nada más.",
}

for letra, pregunta in preguntas.items():
    response = model_con_tools.invoke(pregunta)
    print(f"--- Pregunta {letra}: {len(response.tool_calls)} tool_calls")

    investigador = 0
    for call in response.tool_calls:
        print(call["name"], call["args"])

        # 5. Leer el formulario: el tema está en args["research_topic"]
        if call["name"] == "ConductResearch":
            investigador += 1
            print(f"   Investigador {investigador} recibe: {call['args']['research_topic']}")
