"""
EJERCICIO M01 — Anatomía de un mensaje                     (sin LLM, gratis)
=========================================================================
Un LLM de chat NO recibe "un texto": recibe una LISTA de mensajes.
Cada mensaje es un objeto con, como mínimo:
    .type     -> quién habla: "system", "human", "ai" o "tool"
    .content  -> el texto del mensaje

Los 4 tipos:
    SystemMessage -> instrucciones / rol del modelo (normalmente va primero)
    HumanMessage  -> lo que dice el usuario
    AIMessage     -> lo que respondió el modelo
    ToolMessage   -> el resultado de ejecutar una herramienta (lo verás en M03)

Atajo: en tu Ejercicio_7 escribiste ("user", "Add 40 + 75...").
Esa tupla es un atajo que LangChain convierte en HumanMessage por ti.

Ejecuta:  uv run python 04_mensajes/M01_anatomia.py
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, convert_to_messages

# TODO 1: crea un SystemMessage con el texto "Eres un asistente que responde en español."
system = SystemMessage(content="Eres un asistente que responde en español.")


# TODO 2: crea un HumanMessage con el texto "¿Cuál es la capital de Francia?"
pregunta = HumanMessage(content="¿Cuál es la capital de Francia?")

# TODO 3: crea un AIMessage con el texto "La capital de Francia es París."
respuesta = AIMessage(content="La capital de Francia es París.")

# TODO 4: junta los 3 mensajes en una lista, en el orden en que ocurren en un chat
conversacion = [system, pregunta, respuesta]

# TODO 5: recorre `conversacion` e imprime "<type>: <content>" por cada mensaje.
#   Salida esperada:
#     system: Eres un asistente que responde en español.
#     human: ¿Cuál es la capital de Francia?
#     ai: La capital de Francia es París.
for mensage in conversacion:
    print(f"{mensage.type}: {mensage.content}")


# TODO 6: llama a .pretty_print() sobre `respuesta` y fíjate en cómo lo muestra.
respuesta.pretty_print()

# TODO 7: imprime type(respuesta) y type(respuesta.content).
#   Pregúntate: ¿cuál de los dos meterías en un f-string para mostrarlo al usuario?
print(type(respuesta))
print(type(respuesta.content))

# TODO 8: crea la MISMA conversación pero con tuplas ("system", ...), ("user", ...),
#   ("assistant", ...) y conviértela con convert_to_messages(...)

conversacion_con_tuplas = convert_to_messages([
    ("system", "Eres un asistente que responde en español."),
    ("human", "¿Cuál es la capital de Francia?"),
    ("assistant", "La capital de Francia es París."),
])




# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert isinstance(system, SystemMessage), "TODO 1: `system` debe ser un SystemMessage"
assert isinstance(pregunta, HumanMessage), "TODO 2: `pregunta` debe ser un HumanMessage"
assert isinstance(respuesta, AIMessage), "TODO 3: `respuesta` debe ser un AIMessage"
assert isinstance(conversacion, list) and len(conversacion) == 3, "TODO 4: lista de 3 mensajes"
assert [m.type for m in conversacion] == ["system", "human", "ai"], "TODO 4: revisa el orden"
assert conversacion_con_tuplas is not None, "TODO 8: falta convertir las tuplas"
assert [type(m) for m in conversacion_con_tuplas] == [type(m) for m in conversacion], \
    "TODO 8: los tipos no coinciden con `conversacion`"
assert [m.content for m in conversacion_con_tuplas] == [m.content for m in conversacion], \
    "TODO 8: los textos no coinciden con `conversacion`"
print("\n✅ M01 completado")
