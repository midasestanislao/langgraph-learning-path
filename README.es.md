<p align="right"><a href="README.md">🇬🇧 English</a> · 🇪🇸 Español</p>

# LangGraph learning path

Ruta de aprendizaje de [LangGraph](https://github.com/langchain-ai/langgraph), de cero a un agente de investigación multiagente.
Cada ejercicio introduce **un solo concepto nuevo**, explicado en la cabecera del archivo, y se apoya en el anterior.

La ruta completa está en dos idiomas, como dos carpetas paralelas con los mismos ejercicios: [`en/`](en/) (inglés) y [`es/`](es/) (español). El agente terminado vive en su propio repositorio: [deep-research-agent](https://github.com/midasestanislao/deep-research-agent).

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![LangGraph 1.x](https://img.shields.io/badge/LangGraph-1.x-1C3C3C)
![Idiomas: EN · ES](https://img.shields.io/badge/idiomas-EN%20%C2%B7%20ES-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## La ruta

```text
01 grafos ─► 02 chatbots ─► 03 tools y memoria ─► 04 mensajes ─► 05 jev ─► 07 deep research
   │             │               │                    │              │           │
 StateGraph    1 nodo +        @tool, ToolNode,     qué es un      decisión   Command, think_tool,
 nodos, edges, LLM (nube       búsqueda web,        mensaje,       con un     subgrafos, async,
 bucles con    y local)        checkpointer         tool_calls,    SDK        supervisor → agente
 condiciones                                        reducers,                 completo
                                                    structured output
```

| Carpeta | Qué se aprende | Archivos | ¿Usa LLM? |
|---|---|---|---|
| `01_grafos/` | Fundamentos de `StateGraph`: estado, nodos, edges y bucles con `conditional_edges` | `Hello_World`, `Ejercicio_1`–`6` | No |
| `02_chatbots/` | Un grafo de un nodo que llama a un LLM: OpenRouter (`Agent_Bot`) y modelos locales con LM Studio (`local_agent`, `bonsai_agent`) | 3 scripts | Sí |
| `03_tools_y_memoria/` | Agentes con herramientas: `@tool` + `ToolNode`, búsqueda web con DDGS, memoria entre turnos con `InMemorySaver` | `Ejercicio_7`–`9` | Sí |
| `04_mensajes/` | Qué es un mensaje y cómo se usa: historial, `tool_calls` y `ToolMessage` a mano, `filter_messages`, reducers, helpers frente a tools, structured output, un mini chat | `M01`–`M08` | Algunos |
| `05_jev/` | Un grafo donde la decisión la toma un SDK tipado (`typesafe_sdk`, `Choice`) | `lowerHigher` | Sí |
| `07_deep_research/` | Las piezas del agente final, una por ejercicio: `Command` (10–11), `think_tool` y límites (12), clases Pydantic como herramientas (13), un grafo dentro de otro (14), `async`/`gather` (15), un supervisor de juguete (16); y las fases reales `agent_Fase1_*` (scoping) y `agent_Fase2_*` (investigador), que `deep_research.py` une | 27 scripts | Casi todos |

## Habilidades que se practican

| Habilidad | Dónde se practica |
|---|---|
| Grafos de estado: estado tipado, nodos, edges, bucles condicionales | `01_grafos/` |
| Llamar a un LLM desde un nodo, en la nube (OpenRouter) y en local (LM Studio) | `02_chatbots/` |
| Herramientas con `@tool` + `ToolNode`, búsqueda web, memoria con checkpointer | `03_tools_y_memoria/` |
| Tipos de mensaje, `tool_calls` y `ToolMessage` a mano, `filter_messages`, reducers, helpers frente a tools, structured output | `04_mensajes/` (M01–M08) |
| Decisión delegada a un SDK tipado | `05_jev/` |
| Enrutado y bucles con `Command`, sin edges condicionales | `07_deep_research/` Ejercicios 10–11 |
| Reflexión forzada (`think_tool`) y límites duros de vueltas | Ejercicios 12.0–12.1 |
| Clases Pydantic como herramientas que el LLM "rellena" | Ejercicios 13.0–13.2 |
| Un grafo compilado invocado dentro del nodo de otro grafo | Ejercicios 14.0–14.1 |
| Trabajadores en paralelo con `asyncio.gather` y `ainvoke` | Ejercicios 15.0–15.1 |
| Patrón supervisor: delegación, devolución por `ToolMessage`, rondas | Ejercicios 16.0–16.1 |
| Scoping con structured output, investigador con compresión, informe final | `agent_Fase1_*`, `agent_Fase2_*`, `deep_research.py` |

## Formato de los ejercicios

A partir del Ejercicio 10, cada concepto viene en pareja:

- **`Ejercicio_N_0_<tema>.py`**: ejemplo resuelto y comentado, con la salida esperada.
- **`Ejercicio_N_1_<tema>.py`** (y `_2`, `_3`…): subejercicio para hacer uno mismo. Solo trae la cabecera: `CONCEPTOS` (lo nuevo, con su sintaxis mínima) y `ENUNCIADO` (qué construir, diagrama, pistas, salida esperada y una pregunta "para pensar").

La serie `M01`–`M08` de `04_mensajes/` usa otro formato: huecos `TODO` numerados y comprobaciones automáticas al final del archivo (`assert`), que indican qué `TODO` falla.

Los archivos de `07_deep_research/` conservan las explicaciones con las que se construyó cada parte; la versión limpia del agente está en [deep-research-agent](https://github.com/midasestanislao/deep-research-agent).

## Cómo ejecutar

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/) y una clave de [OpenRouter](https://openrouter.ai) para los ejercicios que usan LLM.

```bash
git clone https://github.com/midasestanislao/langgraph-learning-path.git
cd langgraph-learning-path
uv sync

cp .env.example .env        # y pon tu OPENROUTER_API_KEY
uv run python es/01_grafos/Hello_World.py      # versión en español
uv run python en/01_graphs/Hello_World.py      # versión en inglés
```

Cada archivo indica en su cabecera cómo ejecutarlo, siempre desde la raíz del repositorio. Los chatbots locales de `02_chatbots/` necesitan [LM Studio](https://lmstudio.ai) en `http://localhost:1234/v1` (las variables `LMSTUDIO_*` de `.env.example` tienen valores por defecto).

## Referencias

- [LangGraph](https://langchain-ai.github.io/langgraph/) y [LangChain](https://python.langchain.com/).
- La arquitectura del agente final sigue el tutorial [`deep_research_from_scratch`](https://github.com/langchain-ai/deep_research_from_scratch) de LangChain.

## Licencia

[MIT](LICENSE)
