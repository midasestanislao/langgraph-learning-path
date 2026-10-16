"""
ASYNC 1 — Ideas 1 y 2: llamar no ejecuta; await sí               (sin LLM)
=====================================================================================
IDEA 1: una función async NO se ejecuta al llamarla

    async def saludar():
        print("Hola!")

    saludar()        # ← NO imprime nada. Solo crea una "tarea pendiente" (un post-it).
                     #   Python avisa: "coroutine 'saludar' was never awaited"

    Con una función normal (sin async), saludar() SÍ imprimiría "Hola!".
    Con async, llamarla solo escribe el post-it: "pendiente: saludar".

IDEA 2: await es lo que ejecuta la tarea

    await saludar()  # ← coge el post-it, hace el trabajo y espera a que termine

    Regla: await SOLO se puede escribir DENTRO de una función async def.
    Entonces, ¿quién arranca la primera función async? asyncio.run(...):
    es el "botón de encendido". Se pulsa UNA vez, al final del archivo.

Salida esperada:
    Hola!

Para probar la idea 1: comenta las líneas de main() y asyncio.run(), y deja solo
saludar() suelto. Verás que no imprime nada y sale el aviso.

Ejecuta:  uv run python es/07_deep_research/async_1_llamar_vs_await.py
"""
import asyncio


# 1. Una función async. Llamarla (saludar()) NO la ejecuta: crea una tarea pendiente.
async def saludar():
    print("Hola!")


# 2. Para usar await hace falta estar dentro de otra función async.
async def main():
    await saludar()  # <- coge el post-it y hace el trabajo


# 3. El botón de encendido: arranca main() desde código normal. Solo una vez.
asyncio.run(main())
