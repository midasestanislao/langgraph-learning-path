"""
EXERCISE M02 — The LLM has no memory: the memory is the list        (uses LLM, ~4 calls)
=========================================================================================
Each call to model.invoke(...) is independent. The model does NOT remember anything.
If it "seems" to remember, it's because you re-send the whole conversation every time.

In your Exercise_9 the InMemorySaver (checkpointer) seemed to "give memory". What it
actually did was SAVE the list of messages per thread_id and re-send it. Here you'll
do it by hand to see there's no magic.

Run:  uv run python en/04_messages/M02_history_with_llm.py
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

NAME = "Gustavo"  # change it if you want

# ---------- Part A: without history ----------
# TODO 1: invoke the model with a list that only has HumanMessage(f"Hi, my name is {NAME}")
#   Save the result in `r1` and print r1.content
r1 = model.invoke([HumanMessage(f"Hi, my name is {NAME}")])
print(r1.content)

# TODO 2: invoke the model with a NEW list that only has HumanMessage("What is my name?")
#   Save in `r2` and print r2.content.  -> Does it know your name? Why?
r2 = model.invoke([HumanMessage("What is my name?")])
print(r2.content)

# ---------- Part B: with manual history ----------
# TODO 3: create `history` = [SystemMessage("Answer in a single sentence.")]
history = [SystemMessage("Answer in a single sentence.")]

# TODO 4:
#   a) append HumanMessage(f"Hi, my name is {NAME}") to the history
#   b) invoke the model with the history and APPEND the response to the history
#      (hint: what invoke returns is already an AIMessage, no need to create one)
#   c) append HumanMessage("What is my name?") to the history
#   d) invoke again with the history, save in `r4` and append it too

# a)
history.append(HumanMessage(f"Hi, my name is {NAME}"))

# b)
response = model.invoke(history)
history.append(response)

# c)
history.append(HumanMessage("What is my name?"))

# d)
r4 = model.invoke(history)
history.append(r4)

# TODO 5: print each message in the history as "<type>: <content>"
for message in history:
    print(f"{message.type}: {message.content}")

# ---------- Part C: what else an AIMessage carries ----------
# TODO 6: print r4.usage_metadata  (input/output tokens: what you pay for)
#         and print r4.response_metadata.get("model_name")
print(r4.usage_metadata)
print(r4.response_metadata.get("model_name"))

# ----------------------------- Checks (do not touch) -----------------------------
assert isinstance(r1, AIMessage) and isinstance(r2, AIMessage), "TODO 1/2: save what invoke returns"
assert isinstance(history, list) and isinstance(history[0], SystemMessage), "TODO 3"
assert [m.type for m in history] == ["system", "human", "ai", "human", "ai"], \
    f"TODO 4: the history should be system,human,ai,human,ai and it is {[m.type for m in history]}"
assert history[-1] is r4, "TODO 4d: append r4 to the end of the history"
assert NAME.lower() in r4.content.lower(), "TODO 4: with history, the model should know your name"
print("\n✅ M02 completed")
