"""
EJERCICIO M03 — tool_calls y ToolMessage a mano                     (sin LLM, gratis)
=====================================================================================
¿Qué pasa por dentro cuando un agente usa una herramienta? (en Ejercicio_7 lo hacía ToolNode)

1. El LLM responde con un AIMessage cuyo .content suele estar VACÍO, pero con
   .tool_calls = [{"name": "add", "args": {"a": 40, "b": 75}, "id": "call_1"}, ...]
   -> "No te respondo aún: ejecuta estas funciones por mí."
2. TU código ejecuta cada herramienta.
3. Por cada tool_call devuelves un ToolMessage(content=resultado, tool_call_id=<mismo id>).
   El id es lo que enlaza "pregunta" y "respuesta". Si no coincide, la API da error.
4. Vuelves a llamar al LLM con todo el historial, y ahora sí responde en texto.

Aquí SIMULAMOS el paso 1 (creamos el AIMessage nosotros) y tú haces 2 y 3.
Esto es exactamente el `tool_node` que escribirás en el paso 2 del deep research.

Ejecuta:  uv run python 04_mensajes/M03_tool_calls_a_mano.py
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


tools = [add, multiply]

# Esto es lo que devolvería el LLM (simulado):
ai_msg = AIMessage(
    content="",
    tool_calls=[
        {"name": "add", "args": {"a": 40, "b": 75}, "id": "call_1"},
        {"name": "multiply", "args": {"a": 7, "b": 8}, "id": "call_2"},
    ],
)

# TODO 1: recorre ai_msg.tool_calls e imprime name, args e id de cada uno.
#   Pregúntate: ¿cada tool_call es un objeto o un dict? ¿Cómo accedes al nombre?
for tool_call in ai_msg.tool_calls:
    print(f"{tool_call['name']}, {tool_call['args']}, {tool_call['id']}")

# TODO 2: construye un dict {nombre_de_la_tool: tool} a partir de `tools`
#   Pista: cada tool tiene el atributo .name
tools_by_name = {"add": add, "multiply": multiply}


# TODO 3: por cada tool_call:
#   - busca la tool en tools_by_name
#   - ejecútala con .invoke(args)
#   - crea un ToolMessage(content=str(resultado), name=..., tool_call_id=...)
#   y guárdalos todos en la lista `tool_messages`
tool_messages = []
for tool_call in ai_msg.tool_calls:
    tool = tools_by_name[tool_call['name']]
    resultado = tool.invoke(tool_call["args"])
    mensage = ToolMessage(
        content=str(resultado),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )
    tool_messages.append(mensage)


# TODO 4: escribe el helper que decide si hay que ejecutar herramientas o terminar.
#   Esto es el `should_continue` de tus ejercicios 7-9.
def hay_tool_calls(mensaje) -> bool:
    # .tool_calls es una lista: llena si el LLM pidió herramientas, vacía [] si no.
    # bool([]) es False y bool([algo]) es True.
    return bool(mensaje.tool_calls)


# TODO 5: monta la conversación completa:
#   [HumanMessage("Suma 40+75 y multiplica 7x8"), ai_msg, *tool_messages,
#    AIMessage("40+75 = 115 y 7x8 = 56")]
#   y llama a .pretty_print() sobre cada mensaje. Fíjate en cómo se ve el AIMessage
#   con content vacío y en cómo cada ToolMessage muestra su tool_call_id.
conversacion = [
    HumanMessage("Suma 40+75 y multiplica 7x8"),   # 1. el usuario pregunta
    ai_msg,                                        # 2. el LLM pide herramientas
    *tool_messages,                                # 3. los resultados (el * "desempaqueta" la lista)
    AIMessage("40+75 = 115 y 7x8 = 56"),           # 4. el LLM responde con los resultados
]

for m in conversacion:
    m.pretty_print()


# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert tools_by_name is not None and set(tools_by_name) == {"add", "multiply"}, "TODO 2"
assert len(tool_messages) == 2 and all(isinstance(m, ToolMessage) for m in tool_messages), "TODO 3"
assert [m.content for m in tool_messages] == ["115", "56"], "TODO 3: resultados incorrectos"
assert [m.tool_call_id for m in tool_messages] == ["call_1", "call_2"], "TODO 3: los tool_call_id no coinciden"
assert hay_tool_calls(ai_msg) is True, "TODO 4: ai_msg SÍ tiene tool_calls"
assert hay_tool_calls(AIMessage("Hola")) is False, "TODO 4: un AIMessage normal NO tiene tool_calls"
assert conversacion is not None and [m.type for m in conversacion] == ["human", "ai", "tool", "tool", "ai"], "TODO 5"
print("\n✅ M03 completado")
