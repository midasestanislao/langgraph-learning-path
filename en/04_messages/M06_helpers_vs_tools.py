"""
EXERCISE M06 — Helpers vs. tools                                    (no LLM, free)
=====================================================================================
Two things that look the same (both are Python functions) but are NOT:

  HELPER  -> a normal function that YOUR code calls. The LLM doesn't even know it exists.
             E.g.: get_today_str() to put the date into a prompt, formatting
             results, removing duplicates...

  TOOL    -> a function with @tool that the LLM DECIDES to call (via tool_calls).
             The LLM doesn't see your code: it only sees the NAME, the DOCSTRING (as
             the description) and the TYPE HINTS (as the argument schema).
             That's why @tool requires a docstring: that was your error in 07_deep_research/agent.py.

Rule: if the LLM doesn't need to decide when to use it -> it's a helper.

Run:  uv run python en/04_messages/M06_helpers_vs_tools.py
"""
from datetime import datetime

from langchain_core.tools import tool


# ---------- Part A: helpers ----------

# TODO 1: return today's date as text, e.g. "Thu Oct 08, 2026".
#   Use datetime.now().strftime(...). Remember: "%-d" fails on Windows; use "%d".
# datetime.now() returns a datetime OBJECT; .strftime() turns it into TEXT.
#   %a = abbreviated weekday (Thu)   %b = abbreviated month (Oct)
#   %d = 2-digit day (08)            %Y = 4-digit year (2026)
def get_today_str() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


print(f"TODO 1: {get_today_str()}")

# TODO 2: fill in this template with .format(...) using `request` and get_today_str()
PROMPT = """The user asked:
<Request>
{request}
</Request>

Today's date is {date}.
Decide if you need to ask a clarifying question."""

request = "I want to research the best coffee shops in Madrid."
# .format(name=value) replaces each {name} in the template with its value.
# The names must match the placeholders EXACTLY: {request} and {date}.
final_prompt = PROMPT.format(request=request, date=get_today_str())
print(f"\nTODO 2:\n{final_prompt}")


# Sample search results (this is how a search engine like DDGS returns them):
results = [
    {"title": "Madrid coffee shops", "url": "https://a.com", "body": "The 10 best coffee shops..."},
    {"title": "Coffee guide",        "url": "https://b.com", "body": "Where to drink coffee..."},
    {"title": "Madrid coffee shops", "url": "https://a.com", "body": "The 10 best coffee shops..."},  # duplicate
]


# TODO 3: remove duplicates by "url", keeping the FIRST of each url and the order.
# We store the urls already seen in a set (fast lookup: "have I seen this?").
# If the url is new, we keep the result; if we've already seen it, we skip it.
def remove_duplicates(results: list[dict]) -> list[dict]:
    seen_urls = set()
    unique = []
    for r in results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)
    return unique


# TODO 4: turn the list into a text for the LLM with EXACTLY this format
#   (notice the blank line between sources):
#
#   --- SOURCE 1: Madrid coffee shops ---
#   URL: https://a.com
#   The 10 best coffee shops...
#
#   --- SOURCE 2: Coffee guide ---
#   URL: https://b.com
#   Where to drink coffee...
#
#   Hint: enumerate(results, 1) and "\n\n".join(...)
# enumerate(list, 1) gives (number, element) pairs starting at 1: (1, r1), (2, r2)...
# We build one text block per source and at the end glue them with "\n\n"
# (= one blank line between blocks).
def format_sources(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        block = f"--- SOURCE {i}: {r['title']} ---\nURL: {r['url']}\n{r['body']}"
        blocks.append(block)
    return "\n\n".join(blocks)


unique = remove_duplicates(results)
print(f"\nTODO 3: {len(results)} results -> {len(unique)} unique")
print(f"\nTODO 4:\n{format_sources(unique)}")


# ---------- Part B: tools ----------

def add(a: int, b: int) -> int:
    return a + b


# TODO 5: create `add_tool` by applying @tool to a function just like `add`,
#   BUT with a docstring. Then print add_tool.name, add_tool.description
#   and add_tool.args. That is ALL the LLM sees of your tool.
# @tool turns the function into a tool OBJECT. Notice what it extracts:
#   .name        <- the function name
#   .description <- the docstring  (that's why it's mandatory: it's what the LLM reads)
#   .args        <- the parameters and their types, taken from the type hints
@tool
def add_tool(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b


print(f"\nTODO 5:")
print(f"  name:        {add_tool.name}")
print(f"  description: {add_tool.description}")
print(f"  args:        {add_tool.args}")

# TODO 6: run the tool the way a tool_node would: .invoke({"a": 2, "b": 3})
# A tool is NOT called like add_tool(2, 3): it's called with .invoke(args_dict),
# because that's how the args arrive from tool_call["args"] (you saw it in M03).
tool_result = add_tool.invoke({"a": 2, "b": 3})
print(f"\nTODO 6: {tool_result}")

# TODO 7 (question): try putting @tool on `add` (no docstring). What error do you get?
#   Why does it make sense for LangChain to require it?
# Answer: it raises "ValueError: Function must have a docstring if description not provided."
# (the same error you got in 07_deep_research/agent.py). It makes sense because the LLM can't
# read your code: the docstring is the ONLY explanation it gets of what the tool does
# and when to use it. Without it, it wouldn't know what the tool is for.
try:
    tool(add)
except ValueError as e:
    print(f"\nTODO 7: error when using @tool without a docstring -> {e}")


# ----------------------------- Checks (do not touch) -----------------------------
today = get_today_str()
assert isinstance(today, str) and "%" not in today and str(datetime.now().year) in today, "TODO 1"
assert final_prompt and "{" not in final_prompt, "TODO 2: there are still {placeholders} left unfilled"
assert request in final_prompt and today in final_prompt, "TODO 2: the request or the date is missing"
unique = remove_duplicates(results)
assert [r["url"] for r in unique] == ["https://a.com", "https://b.com"], "TODO 3"
assert format_sources(unique) == (
    "--- SOURCE 1: Madrid coffee shops ---\nURL: https://a.com\nThe 10 best coffee shops...\n\n"
    "--- SOURCE 2: Coffee guide ---\nURL: https://b.com\nWhere to drink coffee..."
), "TODO 4: the format does not match exactly"
assert add_tool is not None and add_tool.description, "TODO 5: the tool needs a docstring"
assert set(add_tool.args) == {"a", "b"}, "TODO 5"
assert tool_result == 5, "TODO 6"
print("\n✅ M06 completed")
