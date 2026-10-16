"""
EJERCICIO 14 — Un grafo dentro del nodo de otro grafo                 (sin LLM, gratis)
=====================================================================================
CONCEPTOS

Un grafo compilado es un objeto con .invoke(...)
    app_saludo = graph.compile()
    app_saludo.invoke({"nombre": "ana"})

    Hasta ahora lo llamabas desde FUERA (al final del archivo o en main()).
    Lo nuevo: también puedes llamarlo DENTRO de un nodo de otro grafo.
    No hay sintaxis nueva: es una llamada a función más.

Jefe y trabajador
    GRAFO EXTERNO (jefe)      → tiene un nodo que, por cada tarea, llama al interno
    GRAFO INTERNO (trabajador) → hace su trabajo y devuelve su estado final

    def delegar_saludos(state):
        for nombre in state["nombres"]:
            resultado = app_saludo.invoke({"nombre": nombre})   # ← el jefe llama al trabajador
            ...resultado["saludo"]...

Cada grafo tiene SU estado: el nodo del jefe traduce
    1. construye la ENTRADA del interno con lo que tiene:   {"nombre": nombre}
    2. del RESULTADO del interno lee solo lo que necesita:  resultado["saludo"]

    Ya lo hiciste en agent_Fase2_Part3.py (main() pasaba el brief de la Fase 1 a
    la Fase 2). Aquí la misma idea vive dentro de un nodo.

Por qué importa
    El supervisor de la Fase 4 hará exactamente esto: por cada ConductResearch,
    app_investigador.invoke({...}) y guardar resultado["notas"].

ENUNCIADO (ejemplo resuelto)

    JEFE:          START → delegar_saludos → END
                              │ por cada nombre
    TRABAJADOR:    START → poner_mayuscula → saludar → END

    El trabajador recibe UN nombre y devuelve UN saludo.
    El jefe recibe una LISTA de nombres y devuelve una LISTA de saludos.

Salida esperada:
    Trabajador recibe: ana   -> devuelve: ¡Hola, Ana!
    Trabajador recibe: luis  -> devuelve: ¡Hola, Luis!
    Trabajador recibe: marta -> devuelve: ¡Hola, Marta!
    {'nombres': ['ana', 'luis', 'marta'], 'saludos': ['¡Hola, Ana!', '¡Hola, Luis!', '¡Hola, Marta!']}

Para pensar:
    - El estado del jefe no tiene "nombre" ni "saludo", y el del trabajador no
      tiene "nombres" ni "saludos". ¿Quién hace la traducción entre los dos?
    - ¿Qué pasaría si en delegar_saludos escribieras app_saludo.invoke(state)
      directamente, sin construir el dict?

Ejecuta:  uv run python es/07_deep_research/Ejercicio_14_0_subgrafo.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ============ 1. GRAFO INTERNO: el trabajador (un grafo normal, como los del 01_grafos) ============
class SaludoState(TypedDict):
    nombre: str
    saludo: str


def poner_mayuscula(state: SaludoState) -> dict:
    return {"nombre": state["nombre"].capitalize()}      # "ana" -> "Ana"


def saludar(state: SaludoState) -> dict:
    return {"saludo": f"¡Hola, {state['nombre']}!"}


graph_saludo = StateGraph(SaludoState)
graph_saludo.add_node("poner_mayuscula", poner_mayuscula)
graph_saludo.add_node("saludar", saludar)
graph_saludo.add_edge(START, "poner_mayuscula")
graph_saludo.add_edge("poner_mayuscula", "saludar")
graph_saludo.add_edge("saludar", END)

app_saludo = graph_saludo.compile()        # ← el trabajador, listo para que lo llamen


# ============ 2. GRAFO EXTERNO: el jefe ============
class JefeState(TypedDict):
    nombres: list[str]       # las tareas
    saludos: list[str]       # los resultados


# 3. LO NUEVO: un nodo que llama a OTRO grafo por dentro
def delegar_saludos(state: JefeState) -> dict:
    saludos = []
    for nombre in state["nombres"]:
        # a) construir la entrada del trabajador con lo que tiene el jefe
        resultado = app_saludo.invoke({"nombre": nombre})
        # b) leer del resultado solo lo que el jefe necesita
        saludos.append(resultado["saludo"])
        print(f"Trabajador recibe: {nombre:<5} -> devuelve: {resultado['saludo']}")
    return {"saludos": saludos}


graph_jefe = StateGraph(JefeState)
graph_jefe.add_node("delegar_saludos", delegar_saludos)
graph_jefe.add_edge(START, "delegar_saludos")
graph_jefe.add_edge("delegar_saludos", END)

app_jefe = graph_jefe.compile()


# 4. Probarlo: solo se invoca al JEFE; él se encarga de llamar al trabajador
print(app_jefe.invoke({"nombres": ["ana", "luis", "marta"]}))
