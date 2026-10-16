<p align="right">🇬🇧 English · <a href="README.es.md">🇪🇸 Español</a></p>

# LangGraph learning path

A step-by-step path to learn [LangGraph](https://github.com/langchain-ai/langgraph), from the first graph to a multi-agent deep research system.
Each exercise introduces **one new concept**, explained in the header of the file, and builds on the previous one.

The whole path is available in two languages, as two parallel folders with the same exercises: [`en/`](en/) (English) and [`es/`](es/) (Spanish). The finished agent lives in its own repository: [deep-research-agent](https://github.com/midasestanislao/deep-research-agent).

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![LangGraph 1.x](https://img.shields.io/badge/LangGraph-1.x-1C3C3C)
![Languages: EN · ES](https://img.shields.io/badge/languages-EN%20%C2%B7%20ES-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## The path

```text
01 graphs ─► 02 chatbots ─► 03 tools & memory ─► 04 messages ─► 05 jev ─► 07 deep research
   │             │               │                   │              │           │
 StateGraph,   1 node +        @tool, ToolNode,    what a         decision   Command, think_tool,
 nodes, edges, an LLM (cloud   web search,         message is,    made by    subgraphs, async,
 conditional   and local)      checkpointer        tool_calls,    an SDK     supervisor → full agent
 loops                                             reducers,
                                                   structured output
```

| Folder | What you learn | Files | Uses an LLM? |
|---|---|---|---|
| `01_graphs/` | `StateGraph` fundamentals: state, nodes, edges and loops with `conditional_edges` | `Hello_World`, `Exercise_1`–`6` | No |
| `02_chatbots/` | A one-node graph that calls an LLM: OpenRouter (`Agent_Bot`) and local models through LM Studio (`local_agent`, `bonsai_agent`) | 3 scripts | Yes |
| `03_tools_and_memory/` | Agents with tools: `@tool` + `ToolNode`, web search with DDGS, memory across turns with `InMemorySaver` | `Exercise_7`–`9` | Yes |
| `04_messages/` | What a message is and how it is used: history, `tool_calls` and `ToolMessage` by hand, `filter_messages`, reducers, helpers vs. tools, structured output, a mini chat | `M01`–`M08` | Some |
| `05_jev/` | A graph where the decision is made by a typed SDK (`typesafe_sdk`, `Choice`) | `lowerHigher` | Yes |
| `07_deep_research/` | The pieces of the final agent, one per exercise: `Command` (10–11), `think_tool` and limits (12), Pydantic classes as tools (13), a graph inside another graph (14), `async`/`gather` (15), a toy supervisor (16); plus the real phases `agent_phase1_*` (scoping) and `agent_phase2_*` (researcher), which `deep_research.py` joins together | 27 scripts | Most |

## Exercise format

From Exercise 10 on, every concept comes as a pair:

- **`Exercise_N_0_<topic>.py`**: a solved, commented example with its expected output.
- **`Exercise_N_1_<topic>.py`** (and `_2`, `_3`…): a sub-exercise to do yourself. It only contains the header: `CONCEPTS` (what is new, with the minimal syntax) and `EXERCISE` (what to build, diagram, hints, expected output and a "think about it" question).

The `M01`–`M08` series in `04_messages/` uses a different format: numbered `TODO` gaps and automatic checks (`assert`) at the end of the file that tell you which `TODO` fails.

The files in `07_deep_research/` keep the explanations with which each part was built; the clean version of the agent is in [deep-research-agent](https://github.com/midasestanislao/deep-research-agent).

## How to run

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) and an [OpenRouter](https://openrouter.ai) API key for the exercises that use an LLM.

```bash
git clone https://github.com/midasestanislao/langgraph-learning-path.git
cd langgraph-learning-path
uv sync

cp .env.example .env        # then put your OPENROUTER_API_KEY in .env
uv run python en/01_graphs/Hello_World.py      # English version
uv run python es/01_grafos/Hello_World.py      # Spanish version
```

Every file says in its header how to run it, always from the repository root. The local chatbots in `02_chatbots/` need [LM Studio](https://lmstudio.ai) at `http://localhost:1234/v1` (the `LMSTUDIO_*` variables in `.env.example` have defaults).

## References

- [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/).
- The architecture of the final agent follows the [`deep_research_from_scratch`](https://github.com/langchain-ai/deep_research_from_scratch) tutorial by LangChain.

## License

[MIT](LICENSE)
