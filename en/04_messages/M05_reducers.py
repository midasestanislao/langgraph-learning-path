"""
EXERCISE M05 — Reducers: why messages ACCUMULATE in the state             (no LLM, free)
==========================================================================================
In your exercises 7-9 you wrote:
    messages: Annotated[Sequence[BaseMessage], add_messages]

and your nodes returned {"messages": [response]}, just ONE message. Why wasn't the rest
of the history wiped out? Because `add_messages` is a REDUCER: when a node returns a
value for that field, LangGraph does not REPLACE it, but instead calls:

    new_state = add_messages(previous_value, value_returned_by_the_node)

Without a reducer, the returned value REPLACES the previous one.
MessagesState (step 1) is exactly a TypedDict with that field already declared.

In the deep research you'll also see `Annotated[list[str], operator.add]` for notes.

Run:  uv run python en/04_messages/M05_reducers.py
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages

# ---------- Part A: reducers as normal functions ----------
old = [HumanMessage("hello", id="1"), AIMessage("how's it going?", id="2")]
new = [HumanMessage("good, and you?", id="3")]

# TODO 1: call add_messages(old, new) by hand. How many messages are left?
# A reducer is a normal function: it receives (what was there, what's new) and returns the result.
# The id "3" didn't exist -> it gets APPENDED at the end -> 2 + 1 = 3 messages.
result_1 = add_messages(old, new)
print(f"TODO 1: {len(result_1)} messages -> {[m.content for m in result_1]}")

# TODO 2: call add_messages(old, [AIMessage("how are you?", id="2")]).
#   Look at the result: what happens when the id ALREADY exists?
# The id "2" ALREADY existed -> add_messages REPLACES it in place instead of appending it.
# There are still 2 messages, but the second one now says "how are you?".
# (Useful to "correct" a message in the history without duplicating it.)
result_2 = add_messages(old, [AIMessage("how are you?", id="2")])
print(f"TODO 2: {len(result_2)} messages -> {[m.content for m in result_2]}")

# TODO 3: operator.add with lists of strings is simply `a + b`.
#   Compute operator.add(["note 1"], ["note 2", "note 3"])
# operator.add(a, b) is literally a + b. With lists, "+" concatenates them.
# It's simpler than add_messages: it doesn't look at ids, it just glues one list after the other.
result_3 = operator.add(["note 1"], ["note 2", "note 3"])
print(f"TODO 3: {result_3}")


# ---------- Part B: with reducer vs. without reducer in a real graph ----------
class WithReducer(MessagesState):
    pass


class WithoutReducer(TypedDict):
    messages: list[AnyMessage]   # no Annotated -> no reducer


def respond(state):
    return {"messages": [AIMessage("Hello, I'm the node")]}


# TODO 4: build and compile TWO identical graphs (START -> "respond" -> END):
#   one with WithReducer and the other with WithoutReducer. Invoke both with
#   {"messages": [HumanMessage("hello")]} and print how many messages each result has.
# Since both graphs are the same except for the state, we make a helper that receives
# the state class and returns the compiled graph. That way we don't repeat code.
def build_graph(state_class):
    graph = StateGraph(state_class)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile()


input_state = {"messages": [HumanMessage("hello")]}
result_with = build_graph(WithReducer).invoke(input_state)
result_without = build_graph(WithoutReducer).invoke(input_state)

# WITH reducer:    [hello] + [Hello, I'm the node] -> 2 messages (they accumulate)
# WITHOUT reducer: [Hello, I'm the node]           -> 1 message  (the "hello" was LOST)
print(f"\nTODO 4: with reducer = {len(result_with['messages'])} messages, "
      f"without reducer = {len(result_without['messages'])} message")


# TODO 5 (challenge): create a state with a field `notes: Annotated[list[str], operator.add]`
#   and a graph of TWO nodes in a row where each one returns {"notes": ["note from X"]}.
#   Invoke it with {"notes": []}. At the end `notes` must have the 2 notes.
#   What would happen without operator.add?
class NotesState(TypedDict):
    notes: Annotated[list[str], operator.add]   # Annotated[type, reducer]


def researcher_a(state: NotesState):
    # Returns ONLY its new note; the reducer takes care of adding it to the previous ones.
    return {"notes": ["note from A"]}


def researcher_b(state: NotesState):
    return {"notes": ["note from B"]}


graph_notes = StateGraph(NotesState)
graph_notes.add_node("researcher_a", researcher_a)
graph_notes.add_node("researcher_b", researcher_b)
graph_notes.add_edge(START, "researcher_a")
graph_notes.add_edge("researcher_a", "researcher_b")
graph_notes.add_edge("researcher_b", END)

result_notes = graph_notes.compile().invoke({"notes": []})
print(f"TODO 5: {result_notes['notes']}")
# Without operator.add, researcher_b would REPLACE A's note and only ["note from B"] would remain.
# In the deep research this is exactly what happens when several researchers work:
# each one returns ITS notes and the reducer joins them all together.


# ----------------------------- Checks (do not touch) -----------------------------
assert result_1 is not None and len(result_1) == 3, "TODO 1"
assert result_2 is not None and len(result_2) == 2 and result_2[1].content == "how are you?", \
    "TODO 2: with the same id, add_messages REPLACES instead of appending"
assert result_3 == ["note 1", "note 2", "note 3"], "TODO 3"
assert result_with is not None and len(result_with["messages"]) == 2, "TODO 4: with reducer there should be 2"
assert result_without is not None and len(result_without["messages"]) == 1, "TODO 4: without reducer there should be 1"
assert result_notes is not None and len(result_notes["notes"]) == 2, "TODO 5"
print("\n✅ M05 completed")
