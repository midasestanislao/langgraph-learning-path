"""
EXERCISE M04 — Filtering and extracting information from a history   (no LLM, free)
=====================================================================================
After researching, an agent builds up a long history: system, human, ai with tool_calls,
tool (search results), more ai... In the deep research you need to pull out parts:
  - only the tool results               -> to save them as "raw_notes"
  - the agent's final answer            -> what the researcher returns
  - everything except the system prompt -> to summarize

filter_messages(list, include_types=[...] / exclude_types=[...]) does it for you.

Run:  uv run python en/04_messages/M04_filter_messages.py
"""
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, filter_messages,
)

history = [
    SystemMessage("You are a researcher."),
    HumanMessage("Research the price of specialty coffee in Madrid."),
    AIMessage("", tool_calls=[{"name": "search", "args": {"query": "specialty coffee price Madrid"}, "id": "c1"}]),
    ToolMessage("A flat white costs between 3 and 4.5 euros.", name="search", tool_call_id="c1"),
    AIMessage("", tool_calls=[{"name": "think_tool", "args": {"reflection": "I need more sources."}, "id": "c2"}]),
    ToolMessage("Reflection recorded: I need more sources.", name="think_tool", tool_call_id="c2"),
    AIMessage("", tool_calls=[{"name": "search", "args": {"query": "espresso price Madrid 2026"}, "id": "c3"}]),
    ToolMessage("An espresso costs between 1.5 and 2.5 euros.", name="search", tool_call_id="c3"),
    AIMessage("In Madrid, an espresso costs 1.5-2.5 € and a flat white 3-4.5 €."),
]

# TODO 1: with filter_messages, keep only the messages of type "tool"
# include_types = "give me ONLY these types". It takes a LIST of types (the .type values from M01).
only_tools = filter_messages(history, include_types=["tool"])
print("TODO 1 - only tools:")
for m in only_tools:
    print(f"  {m.type}: {m.content}")

# TODO 2: with filter_messages, everything EXCEPT the system
# exclude_types = "give me EVERYTHING except these types".
without_system = filter_messages(history, exclude_types=["system"])
print(f"\nTODO 2 - without system: {len(without_system)} messages (the history has {len(history)})")


# TODO 3: write your own version of the filter with a list comprehension over m.type
# It reads: "give me each m in messages, IF its type is in the list of types".
# Equivalent to:
#     result = []
#     for m in messages:
#         if m.type in types:
#             result.append(m)
#     return result
def filter_by_type(messages, types: list[str]):
    return [m for m in messages if m.type in types]


# TODO 4: only the results of the "search" tool (ignore think_tool).
#   Hint: filter_messages also has include_names=[...]
# include_names filters by the `name` field of the ToolMessage (the one you set in M03).
# That way we discard think_tool's "reflections", which are not data from the internet.
only_search = filter_messages(history, include_names=["search"])
print("\nTODO 4 - only search results:")
for m in only_search:
    print(f"  [{m.tool_call_id}] {m.content}")

# TODO 5: "raw notes" as in the deep research: join with "\n" the .content of all
#   the tool and ai messages (the ai ones with empty content add empty lines, that's fine)
# Two steps:
#   1) filter tool + ai
#   2) "\n".join(...) glues all the texts into ONE single string, separated by line breaks
tool_and_ai_messages = filter_messages(history, include_types=["tool", "ai"])
raw_notes = "\n".join(m.content for m in tool_and_ai_messages)
print(f"\nTODO 5 - raw notes:\n{raw_notes}")


# TODO 6: helper that returns the LAST AI response that has NO tool_calls
#   (the researcher's final answer). If there is none, return None.
# reversed() walks the list from back to front, so the FIRST one that meets
# the condition is the LAST one in the conversation. `return` exits as soon as it finds it.
def final_answer(messages):
    for m in reversed(messages):
        if m.type == "ai" and not m.tool_calls:   # it's AI and did NOT ask for tools
            return m
    return None                                   # we walked through everything and found none


print(f"\nTODO 6 - final answer: {final_answer(history).content}")


# TODO 7: helper that counts messages by type -> {"system": 1, "human": 1, "ai": 4, "tool": 3}
# dict.get(key, 0) returns the current value or 0 if the key doesn't exist yet.
# So the first time we see a type, it starts at 0 and we add 1.
def count_by_type(messages) -> dict:
    counts = {}
    for m in messages:
        counts[m.type] = counts.get(m.type, 0) + 1
    return counts


print(f"\nTODO 7 - counts: {count_by_type(history)}")


# ----------------------------- Checks (do not touch) -----------------------------
assert only_tools is not None and len(only_tools) == 3, "TODO 1: there should be 3 ToolMessages"
assert without_system is not None and len(without_system) == 8 and without_system[0].type == "human", "TODO 2"
assert filter_by_type(history, ["tool"]) == only_tools, "TODO 3: your filter does not match filter_messages"
assert filter_by_type(history, ["human", "ai"]) == filter_messages(history, include_types=["human", "ai"]), "TODO 3"
assert only_search is not None and [m.tool_call_id for m in only_search] == ["c1", "c3"], "TODO 4"
assert raw_notes and "flat white" in raw_notes and "espresso costs" in raw_notes and "You are a" not in raw_notes, "TODO 5"
assert final_answer(history) is history[-1], "TODO 6"
assert final_answer(history[:-1]) is None, "TODO 6: without a final answer it should return None"
assert count_by_type(history) == {"system": 1, "human": 1, "ai": 4, "tool": 3}, "TODO 7"
print("\n✅ M04 completed")
