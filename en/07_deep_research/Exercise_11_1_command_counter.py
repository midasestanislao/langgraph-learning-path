"""
EXERCISE 11.1 — Your Exercise_1, but with a Command loop               (no LLM, free)
=====================================================================================
CONCEPTS

The same ones as in Exercise 11:
    - A node does goto to ITSELF to repeat.
    - At some point the goto points somewhere else (another node or END) to exit.
    - The Literal in the signature includes the node's own name:
          def counter(state) -> Command[Literal["counter", "__end__"]]:

EXERCISE

Rewrite your 01_graphs/Exercise_1.py (count from 0 up to a limit) WITHOUT
should_continue and WITHOUT add_conditional_edges.

    START → greeting → counter ──(count < limit)──► counter   (goes back to itself)
                               └─(count = limit)──► END

  State:  count: int,  message: str,  limit: int   (same as in Exercise_1)

  - greeting: sets the welcome message and count = 0. It's a normal node:
              it returns a dict (without modifying `state` directly).
  - counter:  adds 1 to count and prints "The count is now X".
              If it hasn't reached the limit yet -> goes back to "counter"
              If it has                          -> prints "We reach the limit: X" and goes to END
              In both cases, the update saves the new count.

  Hint: compare with Exercise 11. There the loop went to another node on exit;
        here it exits straight to END (like in your 10.3).

Expected output with limit = 3:
    The count is now 1
    The count is now 2
    The count is now 3
    We reach the limit: 3
    {'count': 3, 'message': 'Hello user, this program will count from 0 to 3', 'limit': 3}

Think about it: what happens if you call app.invoke({"limit": 30}, {"recursion_limit": 10})?
How far does it count and why does it stop?
(Hint: read what Exercise 11 says about the "emergency brake".)

Run:  uv run python en/07_deep_research/Exercise_11_1_command_counter.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. The state
class RocketState(TypedDict):
    number: int
    message: str
    limit: int

# 1. The greeting node. Here we initialize the variable
def begin(state: RocketState) -> Command[Literal["count"]]:
    return Command(goto="count", update={"number":0})


# 2. The LOOP node. Its Literal includes its own name: "count".
def count(state: RocketState) -> Command[Literal["count", "lift_off"]]:
    print(f"{state['number']}...")

    if state["number"] < state["limit"]:
        # Goes back to itself with one number less.
        # On the next turn, state["number"] will already be number - 1.
        return Command(goto="count", update={"number": state["number"] + 1})

    # Exiting the loop: when it reaches the limit, it goes to another node
    return Command(goto="lift_off", update={})


# 3. Normal node
def lift_off(state: RocketState) -> dict:
    return {"message": "🚀 Lift-off!"}


# 4. The graph: only 2 edges. None leaves "count" (not even the loop one).
graph = StateGraph(RocketState)
graph.add_node("begin", begin)
graph.add_node("count", count)
graph.add_node("lift_off", lift_off)

graph.add_edge(START, "begin")
graph.add_edge("lift_off", END)

app = graph.compile()


# 5. Try it
print(app.invoke({"limit": 3}))
