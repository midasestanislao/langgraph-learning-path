"""
EJERCICIO 14.1 — Un supervisor de juguete con investigadores de juguete   (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Todo es lo del Ejercicio 14 (un grafo llamado dentro del nodo de otro).
Lo nuevo es solo esto:

time.sleep(segundos) -> detiene el programa ese tiempo
    import time
    time.sleep(1)        # "tarda 1 segundo", como una búsqueda real en internet

time.perf_counter() -> un reloj para medir cuánto tarda algo
    inicio = time.perf_counter()
    ...lo que quieres medir...
    print(f"Tiempo: {time.perf_counter() - inicio:.1f} segundos")

El grafo interno imita tu Fase 2 en pequeño
    buscar_falso    -> hace de search     (devuelve "resultados" inventados)
    comprimir_falso -> hace de comprimir  (junta los resultados en "notas")

ENUNCIADO

    SUPERVISOR:    START → delegar → escribir_informe → END
                             │ por cada tema
    INVESTIGADOR:  START → buscar_falso → comprimir_falso → END

  1. GRAFO INTERNO (investigador). Estado:  tema: str,  resultados: list[str],  notas: str
        - buscar_falso: espera 1 segundo con time.sleep(1) y devuelve
              {"resultados": [f"dato A sobre {tema}", f"dato B sobre {tema}"]}
        - comprimir_falso: devuelve
              {"notas": f"Notas sobre {tema}: " + ", ".join(resultados)}
        Compílalo en app_investigador.

  2. GRAFO EXTERNO (supervisor). Estado:  temas: list[str],  notas: list[str],  informe: str
        - delegar: por cada tema de state["temas"], invoca app_investigador con
          {"tema": tema}, guarda resultado["notas"] en una lista e imprime
          "Investigador terminó: <tema>". Devuelve {"notas": lista}.
        - escribir_informe: une todas las notas con "\\n" y devuelve {"informe": ...}
        Compílalo en app_supervisor.

  3. Invoca app_supervisor con {"temas": ["Finlandia", "Japón", "Chile"]}, MIDIENDO
     el tiempo con perf_counter. Imprime el informe y el tiempo.

Salida esperada:
    Investigador terminó: Finlandia
    Investigador terminó: Japón
    Investigador terminó: Chile

    INFORME:
    Notas sobre Finlandia: dato A sobre Finlandia, dato B sobre Finlandia
    Notas sobre Japón: dato A sobre Japón, dato B sobre Japón
    Notas sobre Chile: dato A sobre Chile, dato B sobre Chile

    Tiempo: 3.0 segundos

Para pensar:
    - ¿Por qué tarda 3 segundos si cada búsqueda tarda 1? Prueba con 6 temas:
      ¿cuánto tarda? Los investigadores trabajan UNO DETRÁS DE OTRO.
      Con investigadores reales (10-30 s cada uno), eso es mucho tiempo.
      En el Ejercicio 15 aprenderás a lanzarlos TODOS A LA VEZ.
    - Compara delegar con el main() de agent_Fase2_Part3.py: ¿qué línea de los dos
      hace lo mismo?

Ejecuta:  uv run python 07_deep_research/Ejercicio_14_1_investigador_juguete.py
"""
import time
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class InvestigadorState(TypedDict):
  tema: str
  resultados: list[str]
  notas: str

def buscar_falso(state: InvestigadorState) -> dict:
  time.sleep(1)
  return {"resultados": [f"dato A sobre {state['tema']}", f"dato B sobre {state['tema']}"]}

def comprimir_falso(state: InvestigadorState) -> dict:
  return {"notas": f"Notas sobre {state['tema']}: " + ", ".join(state["resultados"])}

graph_investigador = StateGraph(InvestigadorState)
graph_investigador.add_node("buscar_falso", buscar_falso)
graph_investigador.add_node("comprimir_falso", comprimir_falso)
graph_investigador.add_edge(START, "buscar_falso")
graph_investigador.add_edge("buscar_falso", "comprimir_falso")
graph_investigador.add_edge("comprimir_falso", END)

app_investigador = graph_investigador.compile()    

class JefeState(TypedDict):
  temas: list[str]       # las tareas
  notas: list[str]
  informe: str   

def delegar(state: JefeState) -> dict:
    notas = []
    for tema in state['temas']:
        # a) construir la entrada del trabajador con lo que tiene el jefe
        resultado = app_investigador.invoke({"tema": tema})
        # b) leer del resultado solo lo que el jefe necesita
        notas.append(resultado["notas"])
        print(f"Investigador terminó: {tema}")
    return {"notas": notas}

def escribir_informe(state: JefeState) -> dict:
    return {"informe": "\n".join(state["notas"])}

graph_supervisor = StateGraph(JefeState)
graph_supervisor.add_node("delegar", delegar)
graph_supervisor.add_node("escribir_informe", escribir_informe)
graph_supervisor.add_edge(START, "delegar")
graph_supervisor.add_edge("delegar", "escribir_informe")
graph_supervisor.add_edge("escribir_informe", END)

app_supervisor = graph_supervisor.compile()

inicio = time.perf_counter()
resultado = app_supervisor.invoke({"temas": ["Finlandia", "Japón", "Chile"]})
print("\nINFORME:\n" + resultado['informe'])
print(f"\nTiempo: {time.perf_counter() - inicio:.1f} segundos")
