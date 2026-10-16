"""
EJERCICIO 13.1 — Una cafetería: el LLM escribe la comanda             (usa LLM, 2 llamadas)
=====================================================================================
CONCEPTOS

La idea en una frase
    El LLM es el CAMARERO: escucha al cliente y escribe COMANDAS.
    Tu código es la COCINA: lee cada comanda y la prepara.

La comanda es una clase Pydantic (un "formulario" sin código)
    from pydantic import BaseModel, Field

    class PedirCafe(BaseModel):
        \"\"\"Úsala cuando el cliente pida un café. Una vez por cada café.\"\"\"
        tipo: str = Field(description="Tipo de café, por ejemplo 'capuchino'")

    - El docstring le dice al LLM CUÁNDO usarla.
    - El Field le dice QUÉ escribir en ella.

Darle las comandas al camarero
    model_con_tools = llm.bind_tools([PedirCafe])

Las comandas que escribió -> response.tool_calls (una LISTA de dicts)
    response = model_con_tools.invoke("Quiero un capuchino y un latte")

    response.tool_calls
    [{"name": "PedirCafe", "args": {"tipo": "capuchino"}, "id": "..."},
     {"name": "PedirCafe", "args": {"tipo": "latte"},     "id": "..."}]

    Para leer el tipo de café de una comanda:  call["args"]["tipo"]

    Si el cliente no pide nada, la lista está VACÍA y el camarero responde con
    texto normal en response.content.

ENUNCIADO

  1. Crea el llm (OpenRouter, gpt-4o, temperature=0), como en el 13.0.

  2. Crea la clase PedirCafe tal como está en CONCEPTOS.

  3. Crea model_con_tools con bind_tools([PedirCafe]).

  4. Pedido 1: invoca model_con_tools con "Quiero un capuchino y un latte, por favor".
        - imprime cuántas comandas hay:  len(response.tool_calls)
        - recorre response.tool_calls con un for y, por cada comanda, imprime:
              Preparando un capuchino...

  5. Pedido 2: invoca con "Hola, ¿a qué hora abren?".
        - imprime cuántas comandas hay (deberían ser 0)
        - como no hay comandas, imprime lo que respondió: response.content

Salida esperada:
    Comandas: 2
    Preparando un capuchino...
    Preparando un latte...
    Comandas: 0
    Respuesta: Hola, el horario de apertura puede variar según la ubicación...

Para pensar:
    - ¿Quién decidió que eran 2 comandas y no 1? Tú no escribiste ningún if.
    - Prueba el pedido "Quiero tres espressos". ¿Cuántas comandas escribe?
      ¿Escribe 1 comanda con "tres espressos" o 3 comandas con "espresso"?
      (Mira el docstring: "Una vez por cada café".)

Ejecuta:  uv run python 07_deep_research/Ejercicio_13_1_pedido_cafe.py
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

class PedirCafe(BaseModel):
    """Usala cuando el cliente pida un cafe. Una vez por cada cafe"""
    tipo: str = Field(description="Tipo de cafe, por ejemplo 'capuchino'")

model_con_tools = llm.bind_tools([PedirCafe])

response = model_con_tools.invoke("Quiero un capuchino y un latte")

print(f"Comandas: {len(response.tool_calls)} tool_calls")

for call in response.tool_calls:
    print(f"Preparando: {call["args"]["tipo"]}")

response = model_con_tools.invoke("Hola, ¿a qué hora abren?")

print(f"Comandas: {len(response.tool_calls)} tool_calls")
print(f"Respuesta {response.content}")