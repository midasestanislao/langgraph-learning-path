"""
EJERCICIO 13.2 — Un planificador de viajes que delega                 (usa LLM, 3 llamadas)
=====================================================================================
CONCEPTOS

Todo es lo del Ejercicio 13. Lo nuevo es solo esto:

Formularios con VARIOS campos y de distintos tipos
    class ReservarHotel(BaseModel):
        \"\"\"Reserva un hotel en una ciudad.\"\"\"
        ciudad: str = Field(description="Ciudad del hotel")
        noches: int = Field(description="Número de noches")

    El tipo importa: si el campo es int, en call["args"]["noches"] te llega un 3,
    no un "3". El LLM rellena cada campo guiándose por su description.

Repartir cada tool_call según su nombre -> un if/elif por herramienta
    for call in response.tool_calls:
        if call["name"] == "ReservarHotel":
            ...
        elif call["name"] == ...:
            ...

    Esto es exactamente lo que hará el nodo supervisor_tools de la Fase 4:
    mirar el nombre de cada tool_call y decidir qué hacer con él.

ENUNCIADO

Un asistente de viajes que NO reserva nada él mismo: solo rellena formularios.

  1. Crea el llm con temperature=0 (como en el 13).
  2. Define tres clases:
        ReservarVuelo -> origen (str), destino (str), fecha (str)
                         En la description de fecha pide el formato DD/MM y que escriba
                         'sin fecha' si el usuario no la dice.
        ReservarHotel -> ciudad (str), noches (int)
        PlanCompleto  -> sin campos: el usuario ya no necesita nada más
     Escribe un docstring en cada una diciendo CUÁNDO usarla.
  3. Crea model_con_tools con las tres.
  4. Invoca el modelo con estas tres peticiones:
        a) "Quiero ir de Madrid a Roma el 5 de noviembre y quedarme 3 noches en un
            hotel. Luego vuelvo a Madrid el 8 de noviembre."
        b) "Necesito un vuelo de Lima a Cusco."
        c) "Gracias, eso es todo, ya tengo todo reservado."
  5. Por cada respuesta imprime cuántos tool_calls hay y reparte cada uno con if/elif:
        ReservarVuelo -> "Vuelo: Madrid -> Roma (05/11)"
        ReservarHotel -> "Hotel: 3 noches en Roma"
        PlanCompleto  -> "Plan completo, no hay nada más que reservar"
  6. Si una respuesta tiene 0 tool_calls, imprime response.content
     (ahí el LLM te habla con texto en vez de rellenar un formulario).

Salida esperada (puede variar):
    --- Petición a: 3 tool_calls
    Vuelo: Madrid -> Roma (05/11)
    Hotel: 3 noches en Roma
    Vuelo: Roma -> Madrid (08/11)
    --- Petición b: 0 tool_calls
    ¿Podrías proporcionarme la fecha en la que deseas viajar de Lima a Cusco? ...
    --- Petición c: 1 tool_calls
    Plan completo, no hay nada más que reservar

Para pensar:
    - En la petición b el Field decía "escribe 'sin fecha'", pero el LLM prefirió no
      usar la herramienta y preguntarte. ¿Qué cambiarías en el docstring de
      ReservarVuelo para que reserve igual, aunque falte la fecha?
    - En la petición a, ¿cuántos ReservarVuelo hay? ¿Quién decidió que eran dos?

Ejecuta:  uv run python es/07_deep_research/Ejercicio_13_2_planificador_viajes.py
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
    temperature=0
)

class ReservarVuelo(BaseModel):
    """Usalo cuando el usuario quiera un vuelo"""
    origen: str = Field(description="La ciudad de origen del vuelo")
    fecha: str = Field(description="Fecha del vuelo en formato DD/MM, o 'sin fecha' si el usuario no la dice.")
    destino: str = Field(description="La ciudad de destino del vuelo")

class ReservarHotel(BaseModel):
    """Usalo cuando el usuario quiera eservar un hotel en una ciudad."""
    ciudad: str = Field(description="Ciudad del hotel")
    noches: int = Field(description="Numero de noches")

class  PlanCompleto(BaseModel):
    """Úsala cuando ya no haga falta reservar nada más."""

modelo_con_tools = llm.bind_tools([ReservarVuelo, ReservarHotel, PlanCompleto])

# 4. Tres preguntas para ver cuántas veces decide usar cada herramienta
peticiones = {
    "a": "Quiero ir de Madrid a Roma el 5 de noviembre y quedarme 3 noches en un hotel. Luego vuelvo a Madrid el 8 de noviembre.",
    "b": "Necesito un vuelo de Lima a Cusco.",
    "c": "Gracias, eso es todo, ya tengo todo reservado.",
}

for letra, peticion in peticiones.items():
    response = modelo_con_tools.invoke(peticion)
    print(f"--- Petición {letra}: {len(response.tool_calls)} tool_calls")
    if len(response.tool_calls) == 0:
        print(response.content)
    else:
        for call in response.tool_calls:
            args = call["args"]
            if call["name"] == "ReservarVuelo":
                print(f"Vuelo: {args['origen']} -> {args['destino']} ({args['fecha']})")
            
            elif call["name"] == "ReservarHotel":
                print(f"Hotel: {args['noches']} noches en {args['ciudad']}")
            
            elif call["name"] == "PlanCompleto":
                print("Plan completo, no hay nada más que reservar")
