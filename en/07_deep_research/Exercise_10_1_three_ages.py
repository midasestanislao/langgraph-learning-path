"""
EXERCISE 10.1 — Command with three paths: adult / teenager / child     (no LLM, free)
=====================================================================================
CONCEPTS

Command -> a node returns TWO things at once:
    from langgraph.types import Command

    return Command(goto="other_node", update={"field": value})

    goto   -> which node to go to next
    update -> what changes in the state (the same as your usual `return {...}`)

Literal in the signature -> the list of nodes it CAN go to
    from typing import Literal

    def my_node(state) -> Command[Literal["node_a", "node_b"]]:

The important part: NO add_edge leaves the node that uses Command.
The Command already says where to go.

EXERCISE

A program that checks an age and greets an adult and a minor differently.

    START → check_age ──(age >= 18)──► adult → END
                      └─(age < 18)───► minor → END

  State:  age: int   and   message: str

  - check_age: if the age is 18 or more, goes to "adult"; otherwise, goes to "minor".
               It doesn't change anything in the state (update={}).
  - adult:     sets message = "Hello, you are an adult"
  - minor:     sets message = "Hello, you are a minor"

  adult and minor are normal nodes: they return a dict, as always.

Try it with two ages:
    app.invoke({"age": 25})  →  {'age': 25, 'message': 'Hello, you are an adult'}
    app.invoke({"age": 12})  →  {'age': 12, 'message': 'Hello, you are a minor'}

Run:  uv run python en/07_deep_research/Exercise_10_1_three_ages.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. The state: the "boxes" that all the nodes share
class AgeState(TypedDict):
    age: int
    message: str


# 2. The node that DECIDES. It uses Command instead of returning a dict.
#    The Literal says which nodes it can go to: "adult" or "minor".
def check_age(state: AgeState) -> Command[Literal["adult", "teenager", "child"]]:
    if state["age"] >= 18:
        return Command(goto="adult", update={})   # goes to "adult", changes nothing
    elif state["age"] >= 13:
        return Command(goto="teenager", update={})        # goes to "teenager", changes nothing
    return Command(goto="child", update={})        # goes to "minor", changes nothing



# 3. Normal nodes: they do their job and return a dict, as always
def adult(state: AgeState) -> dict:
    return {"message": "Hello, you are an adult"}

def teenager(state: AgeState) -> dict:
    return {"message": "Hello, you are a teenager (13 to 17)"}

def child(state: AgeState) -> dict:
    return {"message": "Hello, you are a child (under 13)"}


# 4. The graph
graph = StateGraph(AgeState)
graph.add_node("check_age", check_age)
graph.add_node("adult", adult)
graph.add_node("teenager", teenager)
graph.add_node("child", child)

graph.add_edge(START, "check_age")
# ⚠️ There is NO add_edge leaving "check_age" here: the Command decides
graph.add_edge("adult", END)
graph.add_edge("teenager", END)
graph.add_edge("child", END)

app = graph.compile()


# 5. Try it with two ages
print(app.invoke({"age": 25}))
print(app.invoke({"age": 12}))
print(app.invoke({"age": 15}))
