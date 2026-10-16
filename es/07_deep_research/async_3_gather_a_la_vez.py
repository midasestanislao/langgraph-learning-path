"""
ASYNC 3 — Idea 4: lanzar varias tareas a la vez con gather       (sin LLM)
=====================================================================================
IDEA 4: asyncio.gather(tarea1, tarea2) arranca TODAS y espera a que TODAS terminen

    resultados = await asyncio.gather(investigar("Chile"), investigar("Japón"))

    - investigar("Chile") e investigar("Japón") solo crean dos tareas pendientes.
    - gather las arranca juntas; el await espera a que las DOS terminen.
    - Devuelve una LISTA con los resultados, en el mismo orden:
          ['Notas sobre Chile', 'Notas sobre Japón']

Por qué tarda 1 segundo y no 2 (línea de tiempo)

    t = 0 s   gather arranca Chile → "empieza: Chile"
                                   → llega a await asyncio.sleep(1) → Chile SE PAUSA
              como Chile está pausada, Python arranca Japón → "empieza: Japón"
                                   → llega a su sleep → Japón se pausa
              (las dos esperan su segundo AL MISMO TIEMPO)
    t = 1 s   el segundo de Chile acabó → "termina: Chile"
              el de Japón también       → "termina: Japón"

    La clave: cuando una tarea llega a un await, dice "yo estoy esperando, que
    pase otro". En async_2 no había "otro" a quien dar paso; con gather, sí.

asyncio.sleep vs time.sleep
    await asyncio.sleep(1) → esta tarea espera, pero deja pasar a las demás
    time.sleep(1)          → CONGELA todo el programa: nadie más avanza
    Pruébalo: cambia la línea del sleep por time.sleep(1) y gather tardará 2.0 s.
    Por eso dentro de una función async siempre se usa asyncio.sleep.

Salida esperada:
    empieza: Chile
    empieza: Japón          ← Japón arranca ANTES de que Chile termine
    termina: Chile
    termina: Japón
    Tiempo: 1.0 s
    ['Notas sobre Chile', 'Notas sobre Japón']

Lo que sigue (idea 5): pasarle a gather una LISTA de tareas con *, en vez de
escribirlas una por una. Es lo que hace el Ejercicio_15_0.

Ejecuta:  uv run python es/07_deep_research/async_3_gather_a_la_vez.py
"""
import asyncio
import time


# 1. Igual que en async_2, pero ahora DEVUELVE un resultado.
async def investigar(tema):
    print(f"empieza: {tema}")
    await asyncio.sleep(1)               # al llegar aquí, la tarea se pausa y deja pasar a otra
    print(f"termina: {tema}")
    return f"Notas sobre {tema}"


# 2. LO NUEVO: gather arranca las dos tareas a la vez y espera a que las dos terminen.
async def main():
    inicio = time.perf_counter()
    resultados = await asyncio.gather(investigar("Chile"), investigar("Japón"))
    print(f"Tiempo: {time.perf_counter() - inicio:.1f} s")   # 1.0 s, no 2.0
    print(resultados)                                        # lista, en el mismo orden


asyncio.run(main())
