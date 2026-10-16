"""
EJERCICIO M04 — Filtrar y extraer información de un historial       (sin LLM, gratis)
=====================================================================================
Tras investigar, un agente acumula un historial largo: system, human, ai con tool_calls,
tool (resultados de búsqueda), más ai... En el deep research necesitas sacar partes:
  - solo los resultados de herramientas  -> para guardarlos como "raw_notes"
  - la respuesta final del agente        -> lo que devuelve el investigador
  - todo menos el system prompt          -> para resumir

filter_messages(lista, include_types=[...] / exclude_types=[...]) lo hace por ti.

Ejecuta:  uv run python 04_mensajes/M04_filtrar_mensajes.py
"""
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages,
)

historial = [
    SystemMessage("Eres un investigador."),
    HumanMessage("Investiga el precio del café de especialidad en Madrid."),
    AIMessage("", tool_calls=[{"name": "search", "args": {"query": "precio café especialidad Madrid"}, "id": "c1"}]),
    ToolMessage("Un flat white cuesta entre 3 y 4,5 euros.", name="search", tool_call_id="c1"),
    AIMessage("", tool_calls=[{"name": "think_tool", "args": {"reflection": "Necesito más fuentes."}, "id": "c2"}]),
    ToolMessage("Reflection recorded: Necesito más fuentes.", name="think_tool", tool_call_id="c2"),
    AIMessage("", tool_calls=[{"name": "search", "args": {"query": "precio espresso Madrid 2026"}, "id": "c3"}]),
    ToolMessage("Un espresso cuesta entre 1,5 y 2,5 euros.", name="search", tool_call_id="c3"),
    AIMessage("En Madrid, un espresso cuesta 1,5-2,5 € y un flat white 3-4,5 €."),
]

# TODO 1: con filter_messages, quédate solo con los mensajes de tipo "tool"
# include_types = "dame SOLO estos tipos". Recibe una LISTA de tipos (los .type de M01).
solo_tools = filter_messages(historial, include_types=["tool"])
print("TODO 1 - solo tools:")
for m in solo_tools:
    print(f"  {m.type}: {m.content}")

# TODO 2: con filter_messages, todo MENOS el system
# exclude_types = "dame TODO excepto estos tipos".
sin_system = filter_messages(historial, exclude_types=["system"])
print(f"\nTODO 2 - sin system: {len(sin_system)} mensajes (el historial tiene {len(historial)})")


# TODO 3: escribe tu propia versión de filtrar con una list comprehension sobre m.type
# Se lee: "dame cada m de mensajes, SI su tipo está en la lista de tipos".
# Equivale a:
#     resultado = []
#     for m in mensajes:
#         if m.type in tipos:
#             resultado.append(m)
#     return resultado
def filtrar(mensajes, tipos: list[str]):
    return [m for m in mensajes if m.type in tipos]


# TODO 4: solo los resultados de la herramienta "search" (ignora think_tool).
#   Pista: filter_messages también tiene include_names=[...]
# include_names filtra por el campo `name` del ToolMessage (el que pusiste en M03).
# Así descartamos las "reflexiones" de think_tool, que no son datos de internet.
solo_search = filter_messages(historial, include_names=["search"])
print("\nTODO 4 - solo resultados de search:")
for m in solo_search:
    print(f"  [{m.tool_call_id}] {m.content}")

# TODO 5: "raw notes" como en el deep research: une con "\n" el .content de todos
#   los mensajes tool y ai (los ai con content vacío aportan líneas vacías, no pasa nada)
# Dos pasos:
#   1) filtrar tool + ai
#   2) "\n".join(...) pega todos los textos en UN solo string, separados por saltos de línea
mensajes_tool_y_ai = filter_messages(historial, include_types=["tool", "ai"])
raw_notes = "\n".join(m.content for m in mensajes_tool_y_ai)
print(f"\nTODO 5 - raw notes:\n{raw_notes}")


# TODO 6: helper que devuelve la ÚLTIMA respuesta de la IA que NO tiene tool_calls
#   (la respuesta final del investigador). Si no hay, devuelve None.
# reversed() recorre la lista de atrás hacia adelante, así el PRIMERO que cumpla
# la condición es el ÚLTIMO de la conversación. `return` sale en cuanto lo encuentra.
def respuesta_final(mensajes):
    for m in reversed(mensajes):
        if m.type == "ai" and not m.tool_calls:   # es IA y NO pidió herramientas
            return m
    return None                                   # recorrimos todo y no hubo ninguno


print(f"\nTODO 6 - respuesta final: {respuesta_final(historial).content}")


# TODO 7: helper que cuenta mensajes por tipo -> {"system": 1, "human": 1, "ai": 4, "tool": 3}
# dict.get(clave, 0) devuelve el valor actual o 0 si la clave aún no existe.
# Así la primera vez que vemos un tipo, empieza en 0 y le sumamos 1.
def contar_por_tipo(mensajes) -> dict:
    conteo = {}
    for m in mensajes:
        conteo[m.type] = conteo.get(m.type, 0) + 1
    return conteo


print(f"\nTODO 7 - conteo: {contar_por_tipo(historial)}")


# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert solo_tools is not None and len(solo_tools) == 3, "TODO 1: deberían ser 3 ToolMessage"
assert sin_system is not None and len(sin_system) == 8 and sin_system[0].type == "human", "TODO 2"
assert filtrar(historial, ["tool"]) == solo_tools, "TODO 3: tu filtro no coincide con filter_messages"
assert filtrar(historial, ["human", "ai"]) == filter_messages(historial, include_types=["human", "ai"]), "TODO 3"
assert solo_search is not None and [m.tool_call_id for m in solo_search] == ["c1", "c3"], "TODO 4"
assert raw_notes and "flat white" in raw_notes and "espresso cuesta" in raw_notes and "Eres un" not in raw_notes, "TODO 5"
assert respuesta_final(historial) is historial[-1], "TODO 6"
assert respuesta_final(historial[:-1]) is None, "TODO 6: sin respuesta final debería devolver None"
assert contar_por_tipo(historial) == {"system": 1, "human": 1, "ai": 4, "tool": 3}, "TODO 7"
print("\n✅ M04 completado")
