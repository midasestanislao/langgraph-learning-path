"""
EJERCICIO 15.1 — Tu supervisor de juguete, en paralelo               (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Todo es lo del Ejercicio 15. Lo nuevo es aplicarlo a LangGraph:

Un nodo puede ser async
    async def delegar(state):
        ...
        resultados = await asyncio.gather(...)
        return {...}
    LangGraph lo acepta igual que un nodo normal. Lo único: el grafo que lo
    contiene se tiene que invocar con ainvoke (abajo).

invoke -> ainvoke
    app_investigador.invoke({"tema": tema})          # versión normal (14.1)
    await app_investigador.ainvoke({"tema": tema})   # versión async: se puede meter en gather

    app_supervisor.invoke(...)                       # 14.1
    await app_supervisor.ainvoke(...)                # 15.1, dentro de un main() async

No hace falta tocar el investigador
    Sus nodos siguen siendo normales (con time.sleep). LangGraph se encarga de que
    varios ainvoke del mismo grafo corran a la vez.

ENUNCIADO

Copia tu Ejercicio_14_1 y cambia SOLO el nodo delegar y la prueba final:

  1. Añade import asyncio.

  2. Convierte delegar en async def. Dentro:
        - crea la lista de tareas:  [app_investigador.ainvoke({"tema": t}) for t in state["temas"]]
        - lánzalas todas:            resultados = await asyncio.gather(*tareas)
        - saca las notas de cada resultado (es una lista de estados del investigador)
        - imprime "Investigador terminó: <tema>" por cada tema
        - devuelve {"notas": ...}
     Pista: para recorrer temas y resultados a la vez, zip(state["temas"], resultados)
     (o una list comprehension sobre resultados, y luego el print en otro for).

  3. La prueba final va dentro de un  async def main():  con el cronómetro, y
     el supervisor se invoca con  await app_supervisor.ainvoke({...}).
     Al final del archivo:  asyncio.run(main())

  Los dos estados, el investigador (buscar_falso, comprimir_falso), escribir_informe
  y los dos grafos NO cambian.

Salida esperada (el informe es el mismo que en el 14.1):
    Investigador terminó: Finlandia
    Investigador terminó: Japón
    Investigador terminó: Chile

    INFORME:
    Notas sobre Finlandia: dato A sobre Finlandia, dato B sobre Finlandia
    Notas sobre Japón: dato A sobre Japón, dato B sobre Japón
    Notas sobre Chile: dato A sobre Chile, dato B sobre Chile

    Tiempo: 1.0 segundos

Para pensar:
    - Prueba con 6 temas. ¿Cuánto tarda ahora? ¿Y en el 14.1?
    - gather devuelve los resultados en el mismo orden que las tareas. ¿Por qué
      eso importa para emparejar cada nota con su tema?

Ejecuta:  uv run python 07_deep_research/Ejercicio_15_1_supervisor_paralelo.py
"""
import time
import asyncio
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

async def delegar(state: JefeState) -> dict:
    notas = []
    tareas = [app_investigador.ainvoke({"tema": t}) for t in state["temas"]]
    
    resultados = await asyncio.gather(*tareas)
    
    for resultado in resultados: 
        notas.append(resultado["notas"])   
        print(f"Investigador terminó: {resultado['tema']}")
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

async def main():
    inicio = time.perf_counter()
    resultado = await app_supervisor.ainvoke({"temas": ["Finlandia", "Japón", "Chile"]})
    print("\nINFORME:\n" + resultado['informe'])
    print(f"\nTiempo: {time.perf_counter() - inicio:.1f} segundos")

asyncio.run(main())