"""
EJERCICIO M02 — El LLM no tiene memoria: la memoria es la lista   (usa LLM, ~4 llamadas)
=========================================================================================
Cada llamada a model.invoke(...) es independiente. El modelo NO recuerda nada.
Si "parece" que recuerda, es porque le reenvías la conversación entera cada vez.

En tu Ejercicio_9 el InMemorySaver (checkpointer) parecía "dar memoria". Lo que hacía
en realidad era GUARDAR la lista de mensajes por thread_id y reenviarla. Aquí lo
harás a mano para ver que no hay magia.

Ejecuta:  uv run python es/04_mensajes/M02_historial_con_llm.py
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

NOMBRE = "Gustavo"  # cámbialo si quieres

# ---------- Parte A: sin historial ----------
# TODO 1: invoca el modelo con una lista que solo tenga HumanMessage(f"Hola, me llamo {NOMBRE}")
#   Guarda el resultado en `r1` e imprime r1.content
r1 = model.invoke([HumanMessage(f"Hola, me llamo {NOMBRE}")])
print(r1.content)

# TODO 2: invoca el modelo con una lista NUEVA que solo tenga HumanMessage("¿Cómo me llamo?")
#   Guarda en `r2` e imprime r2.content.  -> ¿Sabe tu nombre? ¿Por qué?
r2 = model.invoke([HumanMessage("¿Cómo me llamo?")])
print(r2.content)

# ---------- Parte B: con historial manual ----------
# TODO 3: crea `historial` = [SystemMessage("Responde en una sola frase.")]
historial = [SystemMessage("Responde en una sola frase.")]

# TODO 4:
#   a) añade (append) HumanMessage(f"Hola, me llamo {NOMBRE}") al historial
#   b) invoca el modelo con el historial y AÑADE la respuesta al historial
#      (pista: lo que devuelve invoke ya es un AIMessage, no hace falta crear uno)
#   c) añade HumanMessage("¿Cómo me llamo?") al historial
#   d) invoca otra vez con el historial, guarda en `r4` y añádelo también

# a)
historial.append(HumanMessage(f"Hola, me llamo {NOMBRE}"))

# b)
respuesta = model.invoke(historial)
historial.append(respuesta)

# c)
historial.append(HumanMessage("¿Cómo me llamo?"))

# d)
r4 = model.invoke(historial)
historial.append(r4)

# TODO 5: imprime cada mensaje del historial como "<type>: <content>"
for message in historial:
    print(f"{message.type}: {message.content}")

# ---------- Parte C: qué más trae un AIMessage ----------
# TODO 6: imprime r4.usage_metadata  (tokens de entrada/salida: lo que pagas)
#         e imprime r4.response_metadata.get("model_name")
print(r4.usage_metadata)
print(r4.response_metadata.get("model_name"))

# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert isinstance(r1, AIMessage) and isinstance(r2, AIMessage), "TODO 1/2: guarda lo que devuelve invoke"
assert isinstance(historial, list) and isinstance(historial[0], SystemMessage), "TODO 3"
assert [m.type for m in historial] == ["system", "human", "ai", "human", "ai"], \
    f"TODO 4: el historial debería ser system,human,ai,human,ai y es {[m.type for m in historial]}"
assert historial[-1] is r4, "TODO 4d: añade r4 al final del historial"
assert NOMBRE.lower() in r4.content.lower(), "TODO 4: con historial, el modelo debería saber tu nombre"
print("\n✅ M02 completado")
