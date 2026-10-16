"""
FASE 2 — PASO 3: unir la Fase 1 y la Fase 2                         (usa LLM)
=====================================================================================
DÓNDE ESTÁS

    FASE 1 (agent_Fase1_Part3.py)              FASE 2 (agent_Fase2_Part2.py)
    conversación ──► brief  ───────────────►   brief ──► investigar ──► notas
                       └──────── PASO 3 (AHORA): conectar las dos ────────┘

    Al terminar tendrás un programa que: te pregunta lo necesario, escribe el
    brief, investiga en internet y te entrega notas con fuentes.

CONCEPTOS

Importar de tus propios archivos -> reutilizar sin copiar y pegar
    from agent_Fase1_Part3 import app as app_scope
    from agent_Fase2_Part2 import app as app_investigador

    - "agent_Fase1_Part3" es el nombre del archivo SIN el .py
    - "import app" trae la variable app de ese archivo
    - "as app_scope" le pone otro nombre aquí, porque los DOS archivos llaman
      "app" a su grafo y si no, uno taparía al otro

    Funciona porque los 3 archivos están en la misma carpeta (07_deep_research).

¡Cuidado! Importar un archivo EJECUTA todo su código
    Cuando haces "from agent_Fase2_Part2 import app", Python ejecuta ese archivo
    entero, de arriba abajo. Si al final tiene un app.invoke(...) suelto,
    ¡lanzará una investigación completa solo por importarlo!

if __name__ == "__main__": -> código que SOLO corre si ejecutas ESE archivo
    if __name__ == "__main__":
        resultado = app.invoke(inputs)     # corre con: uv run python 07_deep_research/agent_Fase2_Part2.py
                                           # NO corre si otro archivo lo importa

    Tu agent_Fase1_Part3.py ya lo usa (main() está dentro de ese if).
    Tu agent_Fase2_Part2.py todavía no.

La salida de un grafo es la entrada del otro
    resultado_fase1["research_brief"]   →   HumanMessage(content=brief) para la Fase 2

ENUNCIADO — PASO 3

  1. En agent_Fase2_Part2.py: mete la prueba del final (inputs, app.invoke y los
     print) dentro de un  if __name__ == "__main__":
     Comprueba que ese archivo sigue funcionando solo:
        uv run python 07_deep_research/agent_Fase2_Part2.py

  2. En ESTE archivo, importa los dos grafos con nombres distintos
     (app_scope y app_investigador) y HumanMessage.

  3. Escribe main() en dos partes:

     FASE 1 — el bucle de agent_Fase1_Part3.py, usando app_scope
        (con su config y thread_id). Cuando haya research_brief, guárdalo en una
        variable brief y sal del bucle con break.

     FASE 2 — fuera del bucle:
        - imprime "Investigando..." (tarda un poco)
        - invoca app_investigador con {"vueltas": 0, "messages": [HumanMessage(content=brief)]}
        - imprime las notas: resultado["notas"]

  4. Al final del archivo:
        if __name__ == "__main__":
            main()

Prueba con esta conversación:
    User: Investiga escritores
    AI:   ¿Sobre qué aspecto...?                       ← Fase 1 pregunta
    User: Latinoamericanos que hayan ganado el Nobel
    AI:   Voy a investigar...                           ← Fase 1 escribe el brief
    BRIEF: ...
    Investigando...
    --- vuelta 1 -> pide: ['search']                    ← Fase 2 investiga
    ...
    NOTAS: ... con [1], [2]... y la lista de Fuentes    ← resultado final

Para pensar: si escribes "exit" en la Fase 1, el bucle termina sin brief.
¿Qué pasaría entonces en la Fase 2? ¿Cómo lo evitarías?

Ejecuta:  uv run python 07_deep_research/agent_Fase2_Part3.py
"""
from agent_Fase1_Part3 import app as app_scope
from agent_Fase2_Part2 import app as app_investigador

from langchain_core.messages import HumanMessage

def main():
    config = {"configurable": {"thread_id": "1"}}
    brief = None
    while True:
        texto = input("User: ").strip()
        if not texto:
            continue
        elif texto.lower() == "exit":
            break

        resultado = app_scope.invoke({"messages": [HumanMessage(content=texto)]}, config)
        print(f"AI: {resultado['messages'][-1].content}")

        if "research_brief" in resultado:
            brief = resultado["research_brief"]
            print(f"\nBRIEF:\n{resultado['research_brief']}")
            break

    # Si el usuario salió con "exit", no hay brief: no hay nada que investigar
    if brief is None:
        print("Sin brief, no hay nada que investigar. ¡Hasta luego!")
        return

    print("\nInvestigando...")
    resultado = app_investigador.invoke({"vueltas": 0, "messages": [HumanMessage(content=brief)]})
    print("\nNOTAS:\n", resultado["notas"])

if __name__ == "__main__":
    main()
