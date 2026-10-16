"""
EXERCISE 11 — A loop with Command                                      (no LLM, free)
=====================================================================================
CONCEPTS

Loop with Command -> a node can goto ITSELF.
    def count(state) -> Command[Literal["count", "lift_off"]]:
        ...
        return Command(goto="count", update={...})    # it runs again

    Each time it comes back, the node receives the state ALREADY updated by the previous update.

Exiting the loop -> at some point the goto must point SOMEWHERE ELSE (another node or END).
    If it never exits, LangGraph cuts it off with a GraphRecursionError.
    That is its "emergency brake": the recursion_limit (maximum number of steps).
    In your version the default limit is very high (10,007), but you can lower it:

        app.invoke({"number": 3}, {"recursion_limit": 10})

Before vs. now
    Before (Exercise_1):  node + should_continue + add_conditional_edges
    Now:                  the node decides on its own whether to repeat or exit

    This is how the deep research supervisor does its loop.

EXERCISE (solved example)

A countdown for a rocket: it counts from a number down to 0 and then lifts off.

    START → count ──(number > 0)──► count   (goes back to itself)
                  └─(number = 0)──► lift_off → END

  State:  number: int   and   message: str

  - count:    prints the number.
              If it's greater than 0 -> goes back to "count" with number - 1
              If it's 0              -> goes to "lift_off"
  - lift_off: sets message = "🚀 Lift-off!"

Expected output with number = 3:
    3...
    2...
    1...
    0...
    {'number': 0, 'message': '🚀 Lift-off!'}

Run:  uv run python en/07_deep_research/Exercise_11_0_command_loop.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. The state
class RocketState(TypedDict):
    number: int
    message: str


# 2. The LOOP node. Its Literal includes its own name: "count".
def count(state: RocketState) -> Command[Literal["count", "lift_off"]]:
    print(f"{state['number']}...")

    if state["number"] > 0:
        # Goes back to itself with one number less.
        # On the next turn, state["number"] will already be number - 1.
        return Command(goto="count", update={"number": state["number"] - 1})

    # Exiting the loop: when it reaches 0, it goes to another node
    return Command(goto="lift_off", update={})


# 3. Normal node
def lift_off(state: RocketState) -> dict:
    return {"message": "🚀 Lift-off!"}


# 4. The graph: only 2 edges. None leaves "count" (not even the loop one).
graph = StateGraph(RocketState)
graph.add_node("count", count)
graph.add_node("lift_off", lift_off)

graph.add_edge(START, "count")
graph.add_edge("lift_off", END)

app = graph.compile()


# 5. Try it
print(app.invoke({"number": 3}))
