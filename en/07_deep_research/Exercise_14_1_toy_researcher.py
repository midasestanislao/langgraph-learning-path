"""
EXERCISE 14.1 — A toy supervisor with toy researchers                  (no LLM, free)
=====================================================================================
CONCEPTS

Everything is from Exercise 14 (a graph called inside a node of another graph).
The only new things are these:

time.sleep(seconds) -> pauses the program for that long
    import time
    time.sleep(1)        # "takes 1 second", like a real internet search

time.perf_counter() -> a clock to measure how long something takes
    start = time.perf_counter()
    ...what you want to measure...
    print(f"Time: {time.perf_counter() - start:.1f} seconds")

The inner graph imitates your Phase 2 in miniature
    fake_search    -> plays the role of search    (returns made-up "results")
    fake_compress  -> plays the role of compress  (joins the results into "notes")

EXERCISE

    SUPERVISOR:    START → delegate → write_report → END
                             │ for each topic
    RESEARCHER:    START → fake_search → fake_compress → END

  1. INNER GRAPH (researcher). State:  topic: str,  results: list[str],  notes: str
        - fake_search: waits 1 second with time.sleep(1) and returns
              {"results": [f"fact A about {topic}", f"fact B about {topic}"]}
        - fake_compress: returns
              {"notes": f"Notes on {topic}: " + ", ".join(results)}
        Compile it into researcher_app.

  2. OUTER GRAPH (supervisor). State:  topics: list[str],  notes: list[str],  report: str
        - delegate: for each topic in state["topics"], invoke researcher_app with
          {"topic": topic}, store result["notes"] in a list and print
          "Researcher finished: <topic>". Return {"notes": the_list}.
        - write_report: join all the notes with "\\n" and return {"report": ...}
        Compile it into supervisor_app.

  3. Invoke supervisor_app with {"topics": ["Finland", "Japan", "Chile"]}, TIMING
     it with perf_counter. Print the report and the time.

Expected output:
    Researcher finished: Finland
    Researcher finished: Japan
    Researcher finished: Chile

    REPORT:
    Notes on Finland: fact A about Finland, fact B about Finland
    Notes on Japan: fact A about Japan, fact B about Japan
    Notes on Chile: fact A about Chile, fact B about Chile

    Time: 3.0 seconds

Think about it:
    - Why does it take 3 seconds if each search takes 1? Try with 6 topics:
      how long does it take? The researchers work ONE AFTER ANOTHER.
      With real researchers (10-30 s each), that is a lot of time.
      In Exercise 15 you will learn to launch them ALL AT ONCE.
    - Compare delegate with the main() of agent_phase2_part3.py: which line in
      each one does the same thing?

Run:  uv run python en/07_deep_research/Exercise_14_1_toy_researcher.py
"""
import time
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

def delegate(state: BossState) -> dict:
    notes = []
    for topic in state['topics']:
        # a) build the worker's input from what the boss has
        result = researcher_app.invoke({"topic": topic})
        # b) read from the result only what the boss needs
        notes.append(result["notes"])
        print(f"Researcher finished: {topic}")
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

start = time.perf_counter()
result = supervisor_app.invoke({"topics": ["Finland", "Japan", "Chile"]})
print("\nREPORT:\n" + result['report'])
print(f"\nTime: {time.perf_counter() - start:.1f} seconds")
