"""
EJERCICIO M06 — Helpers vs. tools                                   (sin LLM, gratis)
=====================================================================================
Dos cosas que parecen iguales (ambas son funciones de Python) pero NO lo son:

  HELPER  -> una función normal que llama TU código. El LLM ni sabe que existe.
             Ej: get_today_str() para meter la fecha en un prompt, formatear
             resultados, quitar duplicados...

  TOOL    -> una función con @tool que el LLM DECIDE llamar (vía tool_calls).
             El LLM no ve tu código: solo ve el NOMBRE, el DOCSTRING (como
             descripción) y los TYPE HINTS (como esquema de argumentos).
             Por eso @tool exige docstring: ese era tu error en 07_deep_research/agent.py.

Regla: si el LLM no necesita decidir cuándo usarla -> es un helper.

Ejecuta:  uv run python 04_mensajes/M06_helpers_vs_tools.py
"""
from datetime import datetime

from langchain_core.tools import tool


# ---------- Parte A: helpers ----------

# TODO 1: devuelve la fecha de hoy como texto, ej. "Thu Oct 08, 2026".
#   Usa datetime.now().strftime(...). Recuerda: "%-d" falla en Windows; usa "%d".
# datetime.now() devuelve un OBJETO datetime; .strftime() lo convierte en TEXTO.
#   %a = día abreviado (Thu)   %b = mes abreviado (Oct)
#   %d = día con 2 cifras (08) %Y = año con 4 cifras (2026)
def get_today_str() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


print(f"TODO 1: {get_today_str()}")

# TODO 2: rellena este template con .format(...) usando `peticion` y get_today_str()
PROMPT = """The user asked:
<Request>
{request}
</Request>

Today's date is {date}.
Decide if you need to ask a clarifying question."""

peticion = "Quiero investigar los mejores cafés de Madrid."
# .format(nombre=valor) reemplaza cada {nombre} del template por su valor.
# Los nombres deben coincidir EXACTAMENTE con los huecos: {request} y {date}.
prompt_final = PROMPT.format(request=peticion, date=get_today_str())
print(f"\nTODO 2:\n{prompt_final}")


# Resultados de búsqueda de ejemplo (así los devuelve un buscador como DDGS):
resultados = [
    {"title": "Cafés de Madrid", "url": "https://a.com", "body": "Los 10 mejores cafés..."},
    {"title": "Guía café",       "url": "https://b.com", "body": "Dónde tomar café..."},
    {"title": "Cafés de Madrid", "url": "https://a.com", "body": "Los 10 mejores cafés..."},  # repetido
]


# TODO 3: quita duplicados por "url" conservando el PRIMERO de cada url y el orden.
# Guardamos las urls ya vistas en un set (búsqueda rápida "¿ya lo vi?").
# Si la url es nueva, guardamos el resultado; si ya la vimos, lo saltamos.
def quitar_duplicados(resultados: list[dict]) -> list[dict]:
    urls_vistas = set()
    unicos = []
    for r in resultados:
        if r["url"] not in urls_vistas:
            urls_vistas.add(r["url"])
            unicos.append(r)
    return unicos


# TODO 4: convierte la lista en un texto para el LLM con EXACTAMENTE este formato
#   (fíjate en la línea en blanco entre fuentes):
#
#   --- FUENTE 1: Cafés de Madrid ---
#   URL: https://a.com
#   Los 10 mejores cafés...
#
#   --- FUENTE 2: Guía café ---
#   URL: https://b.com
#   Dónde tomar café...
#
#   Pista: enumerate(resultados, 1) y "\n\n".join(...)
# enumerate(lista, 1) da pares (número, elemento) empezando en 1: (1, r1), (2, r2)...
# Construimos un bloque de texto por fuente y al final los pegamos con "\n\n"
# (= una línea en blanco entre bloques).
def formatear_fuentes(resultados: list[dict]) -> str:
    bloques = []
    for i, r in enumerate(resultados, 1):
        bloque = f"--- FUENTE {i}: {r['title']} ---\nURL: {r['url']}\n{r['body']}"
        bloques.append(bloque)
    return "\n\n".join(bloques)


unicos = quitar_duplicados(resultados)
print(f"\nTODO 3: {len(resultados)} resultados -> {len(unicos)} únicos")
print(f"\nTODO 4:\n{formatear_fuentes(unicos)}")


# ---------- Parte B: tools ----------

def sumar(a: int, b: int) -> int:
    return a + b


# TODO 5: crea `sumar_tool` aplicando @tool a una función igual que `sumar`,
#   PERO con docstring. Luego imprime sumar_tool.name, sumar_tool.description
#   y sumar_tool.args. Eso es TODO lo que ve el LLM de tu herramienta.
# @tool convierte la función en un OBJETO tool. Fíjate en lo que extrae:
#   .name        <- el nombre de la función
#   .description <- el docstring  (por eso es obligatorio: es lo que lee el LLM)
#   .args        <- los parámetros y sus tipos, sacados de los type hints
@tool
def sumar_tool(a: int, b: int) -> int:
    """Suma dos números enteros y devuelve el resultado."""
    return a + b


print(f"\nTODO 5:")
print(f"  name:        {sumar_tool.name}")
print(f"  description: {sumar_tool.description}")
print(f"  args:        {sumar_tool.args}")

# TODO 6: ejecuta la tool como lo haría un tool_node: .invoke({"a": 2, "b": 3})
# Una tool NO se llama como sumar_tool(2, 3): se llama con .invoke(dict_de_args),
# porque así es como llegan los args desde tool_call["args"] (lo viste en M03).
resultado_tool = sumar_tool.invoke({"a": 2, "b": 3})
print(f"\nTODO 6: {resultado_tool}")

# TODO 7 (pregunta): prueba a poner @tool sobre `sumar` (sin docstring). ¿Qué error da?
#   ¿Por qué tiene sentido que LangChain lo exija?
# Respuesta: da "ValueError: Function must have a docstring if description not provided."
# (el mismo error que te salió en 07_deep_research/agent.py). Tiene sentido porque el LLM no puede
# leer tu código: el docstring es la ÚNICA explicación que recibe de qué hace la tool
# y cuándo usarla. Sin él, no sabría para qué sirve.
try:
    tool(sumar)
except ValueError as e:
    print(f"\nTODO 7: error al usar @tool sin docstring -> {e}")


# ----------------------------- Comprobaciones (no tocar) -----------------------------
hoy = get_today_str()
assert isinstance(hoy, str) and "%" not in hoy and str(datetime.now().year) in hoy, "TODO 1"
assert prompt_final and "{" not in prompt_final, "TODO 2: quedan {huecos} sin rellenar"
assert peticion in prompt_final and hoy in prompt_final, "TODO 2: falta la petición o la fecha"
unicos = quitar_duplicados(resultados)
assert [r["url"] for r in unicos] == ["https://a.com", "https://b.com"], "TODO 3"
assert formatear_fuentes(unicos) == (
    "--- FUENTE 1: Cafés de Madrid ---\nURL: https://a.com\nLos 10 mejores cafés...\n\n"
    "--- FUENTE 2: Guía café ---\nURL: https://b.com\nDónde tomar café..."
), "TODO 4: el formato no coincide exactamente"
assert sumar_tool is not None and sumar_tool.description, "TODO 5: la tool necesita docstring"
assert set(sumar_tool.args) == {"a", "b"}, "TODO 5"
assert resultado_tool == 5, "TODO 6"
print("\n✅ M06 completado")
