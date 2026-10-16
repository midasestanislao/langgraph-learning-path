"""
EXERCISE 14 — A graph inside a node of another graph                  (no LLM, free)
=====================================================================================
CONCEPTS

A compiled graph is an object with .invoke(...)
    greeting_app = graph.compile()
    greeting_app.invoke({"name": "ana"})

    Until now you called it from OUTSIDE (at the end of the file or in main()).
    What's new: you can also call it INSIDE a node of another graph.
    There is no new syntax: it is just one more function call.

Boss and worker
    OUTER GRAPH (boss)    → has a node that, for each task, calls the inner one
    INNER GRAPH (worker)  → does its job and returns its final state

    def delegate_greetings(state):
        for name in state["names"]:
            result = greeting_app.invoke({"name": name})   # ← the boss calls the worker
            ...result["greeting"]...

Each graph has ITS OWN state: the boss's node translates
    1. it builds the inner graph's INPUT from what it has:        {"name": name}
    2. from the inner graph's RESULT it reads only what it needs:  result["greeting"]

    You already did this in agent_phase2_part3.py (main() passed the Phase 1 brief
    to Phase 2). Here the same idea lives inside a node.

Why it matters
    The Phase 4 supervisor will do exactly this: for each ConductResearch,
    researcher_app.invoke({...}) and store result["notes"].

EXERCISE (solved example)

    BOSS:          START → delegate_greetings → END
                              │ for each name
    WORKER:        START → capitalize → greet → END

    The worker receives ONE name and returns ONE greeting.
    The boss receives a LIST of names and returns a LIST of greetings.

Expected output:
    Worker receives: ana   -> returns: Hello, Ana!
    Worker receives: luis  -> returns: Hello, Luis!
    Worker receives: marta -> returns: Hello, Marta!
    {'names': ['ana', 'luis', 'marta'], 'greetings': ['Hello, Ana!', 'Hello, Luis!', 'Hello, Marta!']}

Think about it:
    - The boss's state has no "name" or "greeting", and the worker's has no
      "names" or "greetings". Who does the translation between the two?
    - What would happen if in delegate_greetings you wrote greeting_app.invoke(state)
      directly, without building the dict?

Run:  uv run python en/07_deep_research/Exercise_14_0_subgraph.py
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# ============ 1. INNER GRAPH: the worker (a normal graph, like the ones in 01_graphs) ============
class GreetingState(TypedDict):
    name: str
    greeting: str


def capitalize(state: GreetingState) -> dict:
    return {"name": state["name"].capitalize()}      # "ana" -> "Ana"


def greet(state: GreetingState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


greeting_graph = StateGraph(GreetingState)
greeting_graph.add_node("capitalize", capitalize)
greeting_graph.add_node("greet", greet)
greeting_graph.add_edge(START, "capitalize")
greeting_graph.add_edge("capitalize", "greet")
greeting_graph.add_edge("greet", END)

greeting_app = greeting_graph.compile()        # ← the worker, ready to be called


# ============ 2. OUTER GRAPH: the boss ============
class BossState(TypedDict):
    names: list[str]         # the tasks
    greetings: list[str]     # the results


# 3. WHAT'S NEW: a node that calls ANOTHER graph inside
def delegate_greetings(state: BossState) -> dict:
    greetings = []
    for name in state["names"]:
        # a) build the worker's input from what the boss has
        result = greeting_app.invoke({"name": name})
        # b) read from the result only what the boss needs
        greetings.append(result["greeting"])
        print(f"Worker receives: {name:<5} -> returns: {result['greeting']}")
    return {"greetings": greetings}


boss_graph = StateGraph(BossState)
boss_graph.add_node("delegate_greetings", delegate_greetings)
boss_graph.add_edge(START, "delegate_greetings")
boss_graph.add_edge("delegate_greetings", END)

boss_app = boss_graph.compile()


# 4. Try it: only the BOSS is invoked; it takes care of calling the worker
print(boss_app.invoke({"names": ["ana", "luis", "marta"]}))
