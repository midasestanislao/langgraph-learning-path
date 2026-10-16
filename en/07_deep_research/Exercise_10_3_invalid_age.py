"""
EXERCISE 10.3 — Command going straight to END (invalid age)            (no LLM, free)
=====================================================================================
CONCEPTS

update -> the part of the Command that CHANGES the state.
    return Command(goto="adult", update={"category": "adult"})

    It's exactly the same as a normal node's `return {"category": "adult"}`,
    but together with the goto.

When is the update applied?
    BEFORE the goto node runs. That's why the next node can already
    read what the Command saved:

    check_age  →  update={"category": "adult"}  →  adult reads state["category"]

Empty update vs. update with data
    update={}                         -> only picks the path, changes nothing
    update={"category": "adult"}      -> picks the path AND saves a value
    update={"a": 1, "b": 2}           -> can change several fields at once

EXERCISE

Same as 10.1, but now check_age also SAVES the category in the state,
and the other nodes use it to build their message.

    START → check_age ──(>= 18)────► adult    → END
                      ├─(13 to 17)─► teenager → END
                      └─(< 13)─────► child    → END

  State:  age: int,  category: str  (NEW),  message: str

Expected output:
    {'age': 25, 'category': 'adult', 'message': 'Hello, adult! You can already vote.'}
    {'age': 15, 'category': 'teenager', 'message': 'Hello, teenager! You have 3 years to go before you can vote.'}
    {'age': 8, 'category': 'child', 'message': 'Hello, child! You have 10 years to go before you can vote.'}

Run:  uv run python en/07_deep_research/Exercise_10_3_invalid_age.py
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


# 1. The state: now it has a NEW field, `category`
class AgeState(TypedDict):
    age: int
    category: str     # check_age fills it in with its Command
    message: str


# 2. The node that DECIDES and also SAVES the category.
#    Before: update={}   →   Now: update={"category": ...}
def check_age(state: AgeState) -> Command[Literal["adult", "teenager", "child", "__end__"]]:
    if state["age"] >= 18:
        return Command(goto="adult", update={"category": "adult"})
    elif state["age"] >= 13:
        return Command(goto="teenager", update={"category": "teenager"})
    elif state["age"] >= 0 and state["age"] <= 12:
        return Command(goto="child", update={"category": "child"})
    else:
        return Command(goto=END, update={"message": "Invalid age"})



# 3. Normal nodes that READ what the Command saved.
#    By the time they get here, state["category"] already has a value.
def adult(state: AgeState) -> dict:
    return {"message": f"Hello, {state['category']}! You can already vote."}


def teenager(state: AgeState) -> dict:
    remaining = 18 - state["age"]
    return {"message": f"Hello, {state['category']}! You have {remaining} years to go before you can vote."}


def child(state: AgeState) -> dict:
    remaining = 18 - state["age"]
    return {"message": f"Hello, {state['category']}! You have {remaining} years to go before you can vote."}


# 4. The graph: same as in 10.1
graph = StateGraph(AgeState)
graph.add_node("check_age", check_age)
graph.add_node("adult", adult)
graph.add_node("teenager", teenager)
graph.add_node("child", child)

graph.add_edge(START, "check_age")
# No add_edge leaves "check_age": the Command decides
graph.add_edge("adult", END)
graph.add_edge("teenager", END)
graph.add_edge("child", END)

app = graph.compile()


# 5. Try it. Notice that `category` shows up in the final result:
#    the Command's update put it there.
print(app.invoke({"age": 25}))
print(app.invoke({"age": 15}))
print(app.invoke({"age": 8}))
print(app.invoke({"age": -8}))
