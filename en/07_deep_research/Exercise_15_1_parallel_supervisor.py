"""
EXERCISE 15.1 — Your toy supervisor, in parallel                       (no LLM, free)
=====================================================================================
CONCEPTS

Everything is from Exercise 15. What's new is applying it to LangGraph:

A node can be async
    async def delegate(state):
        ...
        results = await asyncio.gather(...)
        return {...}
    LangGraph accepts it just like a normal node. The only thing: the graph that
    contains it has to be invoked with ainvoke (below).

invoke -> ainvoke
    researcher_app.invoke({"topic": topic})          # normal version (14.1)
    await researcher_app.ainvoke({"topic": topic})   # async version: it can go into gather

    supervisor_app.invoke(...)                       # 14.1
    await supervisor_app.ainvoke(...)                # 15.1, inside an async main()

No need to touch the researcher
    Its nodes are still normal ones (with time.sleep). LangGraph takes care of
    running several ainvoke of the same graph at the same time.

EXERCISE

Copy your Exercise_14_1 and change ONLY the delegate node and the final test:

  1. Add import asyncio.

  2. Turn delegate into an async def. Inside:
        - build the list of tasks:   [researcher_app.ainvoke({"topic": t}) for t in state["topics"]]
        - launch them all:           results = await asyncio.gather(*tasks)
        - take the notes out of each result (it is a list of researcher states)
        - print "Researcher finished: <topic>" for each topic
        - return {"notes": ...}
     Hint: to walk through topics and results at the same time, zip(state["topics"], results)
     (or a list comprehension over results, and then the print in another for).

  3. The final test goes inside an  async def main():  with the stopwatch, and
     the supervisor is invoked with  await supervisor_app.ainvoke({...}).
     At the end of the file:  asyncio.run(main())

  The two states, the researcher (fake_search, fake_compress), write_report
  and the two graphs do NOT change.

Expected output (the report is the same as in 14.1):
    Researcher finished: Finland
    Researcher finished: Japan
    Researcher finished: Chile

    REPORT:
    Notes on Finland: fact A about Finland, fact B about Finland
    Notes on Japan: fact A about Japan, fact B about Japan
    Notes on Chile: fact A about Chile, fact B about Chile

    Time: 1.0 seconds

Think about it:
    - Try with 6 topics. How long does it take now? And in 14.1?
    - gather returns the results in the same order as the tasks. Why does that
      matter for pairing each note with its topic?

Run:  uv run python en/07_deep_research/Exercise_15_1_parallel_supervisor.py
"""
import time
import asyncio
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class ResearcherState(TypedDict):
  topic: str
  results: list[str]
  notes: str

def fake_search(state: ResearcherState) -> dict:
  time.sleep(1)
  return {"results": [f"fact A about {state['topic']}", f"fact B about {state['topic']}"]}

def fake_compress(state: ResearcherState) -> dict:
  return {"notes": f"Notes on {state['topic']}: " + ", ".join(state["results"])}

researcher_graph = StateGraph(ResearcherState)
researcher_graph.add_node("fake_search", fake_search)
researcher_graph.add_node("fake_compress", fake_compress)
researcher_graph.add_edge(START, "fake_search")
researcher_graph.add_edge("fake_search", "fake_compress")
researcher_graph.add_edge("fake_compress", END)

researcher_app = researcher_graph.compile()

class BossState(TypedDict):
  topics: list[str]       # the tasks
  notes: list[str]
  report: str

async def delegate(state: BossState) -> dict:
    notes = []
    tasks = [researcher_app.ainvoke({"topic": t}) for t in state["topics"]]

    results = await asyncio.gather(*tasks)

    for result in results:
        notes.append(result["notes"])
        print(f"Researcher finished: {result['topic']}")
    return {"notes": notes}

def write_report(state: BossState) -> dict:
    return {"report": "\n".join(state["notes"])}

supervisor_graph = StateGraph(BossState)
supervisor_graph.add_node("delegate", delegate)
supervisor_graph.add_node("write_report", write_report)
supervisor_graph.add_edge(START, "delegate")
supervisor_graph.add_edge("delegate", "write_report")
supervisor_graph.add_edge("write_report", END)

supervisor_app = supervisor_graph.compile()

async def main():
    start = time.perf_counter()
    result = await supervisor_app.ainvoke({"topics": ["Finland", "Japan", "Chile"]})
    print("\nREPORT:\n" + result['report'])
    print(f"\nTime: {time.perf_counter() - start:.1f} seconds")

asyncio.run(main())
