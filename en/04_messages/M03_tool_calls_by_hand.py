"""
EXERCISE M03 — tool_calls and ToolMessage by hand                   (no LLM, free)
=====================================================================================
What happens under the hood when an agent uses a tool? (in Exercise_7 ToolNode did it)

1. The LLM replies with an AIMessage whose .content is usually EMPTY, but with
   .tool_calls = [{"name": "add", "args": {"a": 40, "b": 75}, "id": "call_1"}, ...]
   -> "I'm not answering yet: run these functions for me."
2. YOUR code runs each tool.
3. For each tool_call you return a ToolMessage(content=result, tool_call_id=<same id>).
   The id is what links "question" and "answer". If it doesn't match, the API errors out.
4. You call the LLM again with the whole history, and now it does reply with text.

Here we SIMULATE step 1 (we create the AIMessage ourselves) and you do 2 and 3.
This is exactly the `tool_node` you'll write in step 2 of the deep research.

Run:  uv run python en/04_messages/M03_tool_calls_by_hand.py
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


tools = [add, multiply]

# This is what the LLM would return (simulated):
ai_msg = AIMessage(
    content="",
    tool_calls=[
        {"name": "add", "args": {"a": 40, "b": 75}, "id": "call_1"},
        {"name": "multiply", "args": {"a": 7, "b": 8}, "id": "call_2"},
    ],
)

# TODO 1: loop over ai_msg.tool_calls and print name, args and id of each one.
#   Ask yourself: is each tool_call an object or a dict? How do you access the name?
for tool_call in ai_msg.tool_calls:
    print(f"{tool_call['name']}, {tool_call['args']}, {tool_call['id']}")

# TODO 2: build a dict {tool_name: tool} from `tools`
#   Hint: each tool has the .name attribute
tools_by_name = {"add": add, "multiply": multiply}


# TODO 3: for each tool_call:
#   - look up the tool in tools_by_name
#   - run it with .invoke(args)
#   - create a ToolMessage(content=str(result), name=..., tool_call_id=...)
#   and store them all in the list `tool_messages`
tool_messages = []
for tool_call in ai_msg.tool_calls:
    tool = tools_by_name[tool_call['name']]
    result = tool.invoke(tool_call["args"])
    message = ToolMessage(
        content=str(result),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )
    tool_messages.append(message)


# TODO 4: write the helper that decides whether to run tools or finish.
#   This is the `should_continue` from your exercises 7-9.
def has_tool_calls(message) -> bool:
    # .tool_calls is a list: filled if the LLM asked for tools, empty [] if not.
    # bool([]) is False and bool([something]) is True.
    return bool(message.tool_calls)


# TODO 5: assemble the full conversation:
#   [HumanMessage("Add 40+75 and multiply 7x8"), ai_msg, *tool_messages,
#    AIMessage("40+75 = 115 and 7x8 = 56")]
#   and call .pretty_print() on each message. Notice how the AIMessage with empty
#   content looks and how each ToolMessage shows its tool_call_id.
conversation = [
    HumanMessage("Add 40+75 and multiply 7x8"),    # 1. the user asks
    ai_msg,                                        # 2. the LLM asks for tools
    *tool_messages,                                # 3. the results (the * "unpacks" the list)
    AIMessage("40+75 = 115 and 7x8 = 56"),         # 4. the LLM replies with the results
]

for m in conversation:
    m.pretty_print()


# ----------------------------- Checks (do not touch) -----------------------------
assert tools_by_name is not None and set(tools_by_name) == {"add", "multiply"}, "TODO 2"
assert len(tool_messages) == 2 and all(isinstance(m, ToolMessage) for m in tool_messages), "TODO 3"
assert [m.content for m in tool_messages] == ["115", "56"], "TODO 3: incorrect results"
assert [m.tool_call_id for m in tool_messages] == ["call_1", "call_2"], "TODO 3: the tool_call_ids do not match"
assert has_tool_calls(ai_msg) is True, "TODO 4: ai_msg DOES have tool_calls"
assert has_tool_calls(AIMessage("Hello")) is False, "TODO 4: a normal AIMessage does NOT have tool_calls"
assert conversation is not None and [m.type for m in conversation] == ["human", "ai", "tool", "tool", "ai"], "TODO 5"
print("\n✅ M03 completed")
