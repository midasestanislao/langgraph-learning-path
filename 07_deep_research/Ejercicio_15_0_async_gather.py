"""
EJERCICIO 15 — async / await / gather: hacer varias cosas a la vez   (sin LLM, gratis)
=====================================================================================
CONCEPTOS

El problema (lo viste en el 14.1)
    3 investigadores que tardan 1 s cada uno, UNO DETRÁS DE OTRO  -> 3 s.
    Casi todo ese tiempo es ESPERA (a internet, al LLM). Mientras uno espera,
    los otros podrían estar trabajando.

async def -> una función que puede PAUSARSE mientras espera
    async def investigar(tema):
        ...
    Llamarla NO la ejecuta: investigar("Chile") solo crea la "tarea pendiente".
    Se ejecuta cuando alguien hace await sobre ella.

await -> "espera aquí, pero deja que los demás avancen mientras tanto"
    notas = await investigar("Chile")

asyncio.sleep(1) vs time.sleep(1)
    time.sleep(1)           -> congela TODO el programa 1 segundo
    await asyncio.sleep(1)  -> esta tarea espera 1 s, pero las demás siguen

asyncio.gather(*tareas) -> lanza TODAS a la vez y espera a que TODAS terminen
    resultados = await asyncio.gather(*[investigar(t) for t in temas])
    Devuelve una lista con los resultados EN EL MISMO ORDEN que las tareas.
    El * desempaqueta la lista (lo viste en M03 con *tool_messages).

asyncio.run(main()) -> arranca el "motor" async desde código normal
    Solo se llama UNA vez, al final del archivo. Dentro de main() ya puedes usar await.

La conexión con LangGraph (la usarás en el 15.1)
    app.invoke(...)          ->  await app.ainvoke(...)
    Todo grafo compilado tiene las dos versiones. La segunda se puede meter en gather.

ENUNCIADO (ejemplo resuelto)

La misma simulación del 14.1 (3 temas, 1 s cada uno), pero SIN LangGraph, hecha
de las dos formas: uno detrás de otro, y todos a la vez. Compara tiempos y fíjate
en el ORDEN de los prints.

Salida esperada:
    === Uno detrás de otro ===
      empieza: Finlandia
      termina: Finlandia
      empieza: Japón
      termina: Japón
      empieza: Chile
      termina: Chile
    Tiempo: 3.0 s

    === Todos a la vez (gather) ===
      empieza: Finlandia
      empieza: Japón
      empieza: Chile
      termina: Finlandia
      termina: Japón
      termina: Chile
    Tiempo: 1.0 s
    ['Notas sobre Finlandia', 'Notas sobre Japón', 'Notas sobre Chile']

Para pensar:
    - En la versión gather, los tres "empieza" salen ANTES que cualquier "termina".
      ¿Qué estaba haciendo el programa durante ese segundo?
    - Cambia asyncio.sleep(1) por time.sleep(1) en investigar(). ¿Cuánto tarda
      ahora la versión gather? ¿Por qué?

Ejecuta:  uv run python 07_deep_research/Ejercicio_15_0_async_gather.py
"""
import asyncio
import time


# 1. LO NUEVO: una función async. Simula un investigador que tarda 1 segundo.
async def investigar(tema: str) -> str:
    print(f"  empieza: {tema}")
    await asyncio.sleep(1)          # espera 1 s SIN congelar a los demás
    print(f"  termina: {tema}")
    return f"Notas sobre {tema}"


# 2. Versión A: uno detrás de otro (como tu 14.1)
async def uno_detras_de_otro(temas: list[str]) -> list[str]:
    notas = []
    for tema in temas:
        notas.append(await investigar(tema))     # espera a que termine antes de seguir
    return notas


# 3. Versión B: todos a la vez con gather
async def todos_a_la_vez(temas: list[str]) -> list[str]:
    tareas = [investigar(tema) for tema in temas]    # crea las 3 tareas (aún no corren)
    return await asyncio.gather(*tareas)             # las lanza juntas y espera a las 3


# 4. main() también es async para poder usar await dentro
async def main():
    temas = ["Finlandia", "Japón", "Chile"]

    print("=== Uno detrás de otro ===")
    inicio = time.perf_counter()
    await uno_detras_de_otro(temas)
    print(f"Tiempo: {time.perf_counter() - inicio:.1f} s\n")

    print("=== Todos a la vez (gather) ===")
    inicio = time.perf_counter()
    notas = await todos_a_la_vez(temas)
    print(f"Tiempo: {time.perf_counter() - inicio:.1f} s")
    print(notas)


# 5. Arrancar el motor async (una sola vez, desde código normal)
asyncio.run(main())
