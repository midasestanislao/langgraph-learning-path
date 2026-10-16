"""
EXERCISE M01 — Anatomy of a message                         (no LLM, free)
=========================================================================
A chat LLM does NOT receive "a text": it receives a LIST of messages.
Each message is an object with, at minimum:
    .type     -> who is speaking: "system", "human", "ai" or "tool"
    .content  -> the text of the message

The 4 types:
    SystemMessage -> instructions / role of the model (usually goes first)
    HumanMessage  -> what the user says
    AIMessage     -> what the model replied
    ToolMessage   -> the result of running a tool (you'll see it in M03)

Shortcut: in your Exercise_7 you wrote ("user", "Add 40 + 75...").
That tuple is a shortcut that LangChain converts into a HumanMessage for you.

Run:  uv run python en/04_messages/M01_message_anatomy.py
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, convert_to_messages

# TODO 1: create a SystemMessage with the text "You are an assistant that answers in English."
system = SystemMessage(content="You are an assistant that answers in English.")


# TODO 2: create a HumanMessage with the text "What is the capital of France?"
question = HumanMessage(content="What is the capital of France?")

# TODO 3: create an AIMessage with the text "The capital of France is Paris."
answer = AIMessage(content="The capital of France is Paris.")

# TODO 4: put the 3 messages together in a list, in the order they happen in a chat
conversation = [system, question, answer]

# TODO 5: loop over `conversation` and print "<type>: <content>" for each message.
#   Expected output:
#     system: You are an assistant that answers in English.
#     human: What is the capital of France?
#     ai: The capital of France is Paris.
for message in conversation:
    print(f"{message.type}: {message.content}")


# TODO 6: call .pretty_print() on `answer` and notice how it displays it.
answer.pretty_print()

# TODO 7: print type(answer) and type(answer.content).
#   Ask yourself: which of the two would you put in an f-string to show it to the user?
print(type(answer))
print(type(answer.content))

# TODO 8: create the SAME conversation but with tuples ("system", ...), ("user", ...),
#   ("assistant", ...) and convert it with convert_to_messages(...)

conversation_with_tuples = convert_to_messages([
    ("system", "You are an assistant that answers in English."),
    ("human", "What is the capital of France?"),
    ("assistant", "The capital of France is Paris."),
])




# ----------------------------- Checks (do not touch) -----------------------------
assert isinstance(system, SystemMessage), "TODO 1: `system` must be a SystemMessage"
assert isinstance(question, HumanMessage), "TODO 2: `question` must be a HumanMessage"
assert isinstance(answer, AIMessage), "TODO 3: `answer` must be an AIMessage"
assert isinstance(conversation, list) and len(conversation) == 3, "TODO 4: list of 3 messages"
assert [m.type for m in conversation] == ["system", "human", "ai"], "TODO 4: check the order"
assert conversation_with_tuples is not None, "TODO 8: the tuples still need to be converted"
assert [type(m) for m in conversation_with_tuples] == [type(m) for m in conversation], \
    "TODO 8: the types do not match `conversation`"
assert [m.content for m in conversation_with_tuples] == [m.content for m in conversation], \
    "TODO 8: the texts do not match `conversation`"
print("\n✅ M01 completed")
