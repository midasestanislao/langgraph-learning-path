"""
EJERCICIO M08 — Mini-proyecto: chat con historial y comandos        (usa LLM)
=====================================================================================
Junta todo lo anterior en un chat de consola, SIN LangGraph, solo con mensajes y helpers.
(Tus 02_chatbots/chatbot.py y Agent_Bot.py olvidaban todo en cada turno: este no.)

Requisitos:
  1. El historial empieza con un SystemMessage que incluya la fecha de hoy
     (usa tu get_today_str de M06).
  2. Cada turno: añade el HumanMessage, invoca el modelo con TODO el historial,
     añade la respuesta e imprímela.
  3. Comandos (no se envían al LLM):
       /historial -> imprime cada mensaje como "<type>: <content>" (numerado)
       /stats     -> imprime cuántos mensajes hay de cada tipo (tu contar_por_tipo de M04)
                     y el total de tokens gastados (suma de usage_metadata["total_tokens"]
                     de cada AIMessage)
       /reset     -> borra la conversación pero CONSERVA el SystemMessage
       /salir     -> termina
  4. Pon cada comando en su propia función helper (ej. mostrar_historial(historial)).

No hay comprobaciones automáticas: pruébalo tú.
  - Dile tu nombre, pregúntale por él, haz /reset y vuelve a preguntar.
  - Pregúntale "¿qué día es hoy?" (¿lo sabe gracias al SystemMessage?)
  - Usa /stats antes y después de varias preguntas y mira cómo crecen los tokens:
    cada turno reenvía TODO el historial. Por eso el deep research comprime.

Ejecuta:  uv run python es/04_mensajes/M08_mini_chat.py
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# ===================== HELPERS (reutilizados de M04 y M06) =====================

def get_today_str() -> str:
    """De M06: la fecha de hoy como texto."""
    return datetime.now().strftime("%a %b %d, %Y")


def contar_por_tipo(mensajes) -> dict:
    """De M04: cuántos mensajes hay de cada tipo."""
    conteo = {}
    for m in mensajes:
        conteo[m.type] = conteo.get(m.type, 0) + 1
    return conteo


def crear_system_message() -> SystemMessage:
    """El SystemMessage inicial. Es una función porque /reset necesita crearlo de nuevo."""
    return SystemMessage(
        f"Eres un asistente amable que responde en español. Hoy es {get_today_str()}."
    )


# ===================== HELPERS DE LOS COMANDOS =====================

def mostrar_historial(historial):
    """/historial -> cada mensaje numerado como '<n>. <type>: <content>'."""
    print()
    for i, m in enumerate(historial, 1):
        print(f"  {i}. {m.type}: {m.content}")
    print()


def mostrar_stats(historial):
    """/stats -> mensajes por tipo y tokens totales gastados."""
    # Solo los AIMessage traen usage_metadata (los tokens de esa llamada al LLM).
    # Sumamos total_tokens de cada respuesta de la IA.
    total_tokens = 0
    for m in historial:
        if isinstance(m, AIMessage) and m.usage_metadata:
            total_tokens += m.usage_metadata["total_tokens"]

    print(f"\n  Mensajes por tipo: {contar_por_tipo(historial)}")
    print(f"  Tokens gastados:   {total_tokens}\n")


def resetear(historial):
    """/reset -> deja solo el SystemMessage (el primero de la lista)."""
    # historial[:1] = una lista con solo el primer elemento (el SystemMessage).
    return historial[:1]


def responder(historial, texto_usuario: str):
    """Un turno normal de chat: añade la pregunta, llama al LLM y añade la respuesta."""
    historial.append(HumanMessage(texto_usuario))
    respuesta = model.invoke(historial)      # se envía TODO el historial (memoria)
    historial.append(respuesta)              # la respuesta ya es un AIMessage
    print(f"\nAI: {respuesta.content}\n")


# ===================== BUCLE PRINCIPAL =====================

def main():
    historial = [crear_system_message()]
    print("Mini chat. Comandos: /historial  /stats  /reset  /salir\n")

    while True:
        texto = input("User: ").strip()

        if not texto:                 # el usuario pulsó Enter sin escribir nada
            continue
        elif texto == "/salir":
            print("¡Hasta luego!")
            break
        elif texto == "/historial":
            mostrar_historial(historial)
        elif texto == "/stats":
            mostrar_stats(historial)
        elif texto == "/reset":
            historial = resetear(historial)
            print("\n  Conversación borrada (se conserva el SystemMessage).\n")
        else:
            responder(historial, texto)


if __name__ == "__main__":
    main()
