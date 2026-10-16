"""
EJERCICIO M07 — Structured output: el LLM devuelve un objeto, no un mensaje   (usa LLM, ~14 llamadas)
=====================================================================================================
model.invoke(...)                                   -> AIMessage (texto libre en .content)
model.with_structured_output(MiClase).invoke(...)   -> un objeto MiClase con campos tipados

Es la base del paso 1 (scoping): en vez de leer "creo que necesito más info..." y
adivinar, recibes `need_clarification=True` y lo usas en un `if`.

El LLM rellena los campos guiándose por el docstring de la clase y por el
`description` de cada Field: son instrucciones, escríbelas bien.

Ejecuta:  uv run python es/04_mensajes/M07_structured_output.py
"""
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ---------- Parte A: extraer datos ----------

# TODO 1: completa la clase con estos campos (todos con Field(description=...)):
#   titulo: str, anio: int, director: str, generos: list[str]
# Cada campo es:   nombre: tipo = Field(description="instrucción para el LLM")
# El TIPO importa: si pones `int`, Pydantic garantiza que recibes un número, no "2014".
class Pelicula(BaseModel):
    """Datos básicos de una película."""
    titulo: str = Field(description="Título original de la película")
    anio: int = Field(description="Año de estreno")
    director: str = Field(description="Nombre del director o directora")
    generos: list[str] = Field(description="Lista de géneros de la película")


# TODO 2: crea el modelo estructurado e invócalo con [HumanMessage("Háblame de Interstellar")]
#   Imprime type(pelicula) y pelicula. Luego accede a pelicula.anio + 1 (¡es un int de verdad!)
# with_structured_output(Clase) crea un "modelo envoltorio": se invoca igual que el normal,
# pero devuelve un objeto Pelicula en lugar de un AIMessage.
modelo_pelicula = model.with_structured_output(Pelicula)
pelicula = modelo_pelicula.invoke([HumanMessage("Háblame de Interstellar")])

print(f"TODO 2: {type(pelicula)}")
print(f"  {pelicula}")
print(f"  título: {pelicula.titulo} | director: {pelicula.director}")
print(f"  anio + 1 = {pelicula.anio + 1}   <- se puede sumar porque es int, no texto")


# ---------- Parte B: tomar decisiones (preview del paso 1) ----------

# TODO 3: crea la clase Decision con:
#   necesita_aclaracion: bool  -> si la petición es demasiado vaga para investigar
#   pregunta: str              -> la pregunta para el usuario ("" si no hace falta)
class Decision(BaseModel):
    """Decide si una petición de investigación necesita aclaración."""
    necesita_aclaracion: bool = Field(description="True si la petición es demasiado vaga para investigar")
    pregunta: str = Field(description="Pregunta para el usuario, o \"\" si no hace falta aclaración")


# TODO 4: invoca el modelo estructurado con Decision para estas dos peticiones
#   y guarda los resultados. Imprime ambos.
vaga = "Investiga sobre cafés"
clara = "Compara precio y calidad de las 5 cafeterías de especialidad mejor valoradas de Madrid en 2026"

# Creamos el modelo estructurado UNA vez y lo reutilizamos para las dos peticiones.
modelo_decision = model.with_structured_output(Decision)
decision_vaga = modelo_decision.invoke([HumanMessage(vaga)])
decision_clara = modelo_decision.invoke([HumanMessage(clara)])

print(f"\nTODO 4:")
print(f"  vaga  -> {decision_vaga}")
print(f"  clara -> {decision_clara}")


# TODO 5: así lo hace el nodo clarify_with_user: si decision_vaga.necesita_aclaracion,
#   convierte la pregunta en un AIMessage para añadirla a la conversación.
# El objeto Decision NO es un mensaje: no se puede meter en el historial.
# Por eso sacamos el texto (.pregunta) y lo envolvemos en un AIMessage,
# porque desde el punto de vista del usuario, es la IA quien le pregunta.
if decision_vaga.necesita_aclaracion:
    mensaje_para_usuario = AIMessage(content=decision_vaga.pregunta)
    print(f"\nTODO 5:")
    mensaje_para_usuario.pretty_print()


# TODO 6 (experimento — el más importante de este ejercicio):
#   Mira decision_clara. Con una descripción simple, GPT-4o suele pedir aclaración
#   INCLUSO para la petición clara (ej. "¿qué cafeterías quieres comparar?"), porque
#   no sabe que él mismo podrá buscar en internet.
#
#   a) Invoca la petición clara 5 veces en un bucle y cuenta cuántas veces sale True.
#   b) Mejora el docstring de Decision y el description de `necesita_aclaracion`
#      (ej. explica que es un agente que BUSCA EN INTERNET y que los datos que se
#      pueden buscar no se le piden al usuario). Repite a) y compara.
#
#   Lección: aunque pongas temperature=0, la decisión no es 100% estable. Por eso en el
#   paso 1 el prompt dice "si YA preguntaste una vez, casi nunca vuelvas a preguntar":
#   el diseño del flujo compensa que el modelo sea imperfecto.

# b) Misma estructura, pero con instrucciones mucho más precisas en docstring y Field.
class DecisionMejorada(BaseModel):
    """Eres el primer paso de un agente de investigación que BUSCA EN INTERNET por sí mismo.
    Decide si la petición del usuario necesita aclaración antes de empezar a investigar."""
    necesita_aclaracion: bool = Field(description=(
        "True SOLO si la petición es tan vaga que no se sabe qué investigar "
        "(falta el tema, el lugar o el objetivo). False si ya se puede empezar: "
        "los datos que se pueden encontrar buscando en internet (nombres, rankings, "
        "precios) NO se le piden al usuario."
    ))
    pregunta: str = Field(description="Pregunta concisa para el usuario, o \"\" si no hace falta aclaración")


# Helper para no repetir el bucle con cada clase.
def contar_aclaraciones(clase_schema, peticion: str, veces: int = 5) -> int:
    modelo = model.with_structured_output(clase_schema)
    total = 0
    for _ in range(veces):              # "_" = variable que no usamos
        if modelo.invoke([HumanMessage(peticion)]).necesita_aclaracion:
            total += 1
    return total


print("\nTODO 6 - petición CLARA, ¿cuántas veces de 5 pide aclaración?")
print(f"  a) Decision (descripción simple):   {contar_aclaraciones(Decision, clara)}/5")
print(f"  b) DecisionMejorada:                {contar_aclaraciones(DecisionMejorada, clara)}/5")
print("  (Lo ideal sería 0/5. Si no llega a 0, es la lección: el modelo no es 100% fiable.)")


# ----------------------------- Comprobaciones (no tocar) -----------------------------
assert isinstance(pelicula, Pelicula), "TODO 2: debe ser un objeto Pelicula, no un AIMessage"
assert isinstance(pelicula.anio, int) and pelicula.anio == 2014, "TODO 1/2: Interstellar es de 2014"
assert isinstance(pelicula.generos, list), "TODO 1: generos debe ser list[str]"
assert isinstance(decision_vaga, Decision) and isinstance(decision_clara, Decision), "TODO 4"
assert decision_vaga.necesita_aclaracion is True, "TODO 4: la petición vaga debería necesitar aclaración"
# (decision_clara no se comprueba a propósito: ver TODO 6)
assert isinstance(mensaje_para_usuario, AIMessage) and mensaje_para_usuario.content == decision_vaga.pregunta, "TODO 5"
print("\n✅ M07 completado")
