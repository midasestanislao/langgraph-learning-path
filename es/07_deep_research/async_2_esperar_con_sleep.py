"""
ASYNC 2 — Idea 3: una tarea que tiene que esperar                 (sin LLM)
=====================================================================================
IDEA 3: await asyncio.sleep(1) = "espera 1 segundo aquí"

    Un investigador real ESPERA: a internet, al LLM. Para simular esa espera:

        await asyncio.sleep(1)

    asyncio.sleep(1) también es una tarea pendiente ("esperar 1 segundo"), por eso
    lleva await igual que await saludar().

Dos await seguidos = uno detrás de otro

    await investigar("Chile")    # main() espera a que Chile termine DEL TODO...
    await investigar("Japon")    # ...y solo entonces empieza Japón

    Por eso tarda 2 segundos: 1 + 1. Tener async NO ha hecho nada más rápido todavía.
    Esto es exactamente lo que hace tu Ejercicio_14_1 (delegar con un for).
    Lo rápido viene en async_3, con un solo cambio en main().

Salida esperada:
    empieza: Chile
    termina: Chile
    empieza: Japon
    termina: Japon
    Tiempo: 2.0 s

Ejecuta:  uv run python es/07_deep_research/async_2_esperar_con_sleep.py
"""
import asyncio
import time


# 1. Un investigador de juguete: imprime, ESPERA 1 segundo, imprime.
async def investigar(tema):
    print(f"empieza: {tema}")
    await asyncio.sleep(1)  # ← espera 1 segundo (simula una búsqueda)
    print(f"termina: {tema}")


# 2. Dos await seguidos: el segundo no empieza hasta que el primero termina.
async def main():
    inicio =  time.perf_counter()        # el cronómetro arranca
    await investigar("Chile")            # 1 segundo
    await investigar("Japon")            # otro segundo
    print(f"Tiempo: {time.perf_counter() - inicio:.1f} s")   # 2.0 s


asyncio.run(main())
