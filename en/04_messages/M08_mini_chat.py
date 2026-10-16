"""
EXERCISE M08 — Mini-project: chat with history and commands         (uses LLM)
=====================================================================================
Put everything above together in a console chat, WITHOUT LangGraph, just messages and helpers.
(Your 02_chatbots/chatbot.py and Agent_Bot.py forgot everything each turn: this one doesn't.)

Requirements:
  1. The history starts with a SystemMessage that includes today's date
     (use your get_today_str from M06).
  2. Each turn: append the HumanMessage, invoke the model with the WHOLE history,
     append the response and print it.
  3. Commands (not sent to the LLM):
       /history -> prints each message as "<type>: <content>" (numbered)
       /stats   -> prints how many messages there are of each type (your count_by_type from M04)
                   and the total tokens spent (sum of usage_metadata["total_tokens"]
                   of each AIMessage)
       /reset   -> clears the conversation but KEEPS the SystemMessage
       /quit    -> exits
  4. Put each command in its own helper function (e.g. show_history(history)).

There are no automatic checks: test it yourself.
  - Tell it your name, ask it for it, do /reset and ask again.
  - Ask it "what day is it today?" (does it know thanks to the SystemMessage?)
  - Use /stats before and after several questions and watch the tokens grow:
    each turn re-sends the WHOLE history. That's why the deep research compresses.

Run:  uv run python en/04_messages/M08_mini_chat.py
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# ===================== HELPERS (reused from M04 and M06) =====================

def get_today_str() -> str:
    """From M06: today's date as text."""
    return datetime.now().strftime("%a %b %d, %Y")


def count_by_type(messages) -> dict:
    """From M04: how many messages there are of each type."""
    counts = {}
    for m in messages:
        counts[m.type] = counts.get(m.type, 0) + 1
    return counts


def create_system_message() -> SystemMessage:
    """The initial SystemMessage. It's a function because /reset needs to create it again."""
    return SystemMessage(
        f"You are a friendly assistant that answers in English. Today is {get_today_str()}."
    )


# ===================== COMMAND HELPERS =====================

def show_history(history):
    """/history -> each message numbered as '<n>. <type>: <content>'."""
    print()
    for i, m in enumerate(history, 1):
        print(f"  {i}. {m.type}: {m.content}")
    print()


def show_stats(history):
    """/stats -> messages by type and total tokens spent."""
    # Only AIMessages carry usage_metadata (the tokens of that LLM call).
    # We add up total_tokens from each AI response.
    total_tokens = 0
    for m in history:
        if isinstance(m, AIMessage) and m.usage_metadata:
            total_tokens += m.usage_metadata["total_tokens"]

    print(f"\n  Messages by type: {count_by_type(history)}")
    print(f"  Tokens spent:     {total_tokens}\n")


def reset(history):
    """/reset -> keeps only the SystemMessage (the first one in the list)."""
    # history[:1] = a list with only the first element (the SystemMessage).
    return history[:1]


def respond(history, user_text: str):
    """A normal chat turn: append the question, call the LLM and append the response."""
    history.append(HumanMessage(user_text))
    response = model.invoke(history)      # the WHOLE history is sent (memory)
    history.append(response)              # the response is already an AIMessage
    print(f"\nAI: {response.content}\n")


# ===================== MAIN LOOP =====================

def main():
    history = [create_system_message()]
    print("Mini chat. Commands: /history  /stats  /reset  /quit\n")

    while True:
        text = input("User: ").strip()

        if not text:                  # the user pressed Enter without typing anything
            continue
        elif text == "/quit":
            print("See you later!")
            break
        elif text == "/history":
            show_history(history)
        elif text == "/stats":
            show_stats(history)
        elif text == "/reset":
            history = reset(history)
            print("\n  Conversation cleared (the SystemMessage is kept).\n")
        else:
            respond(history, text)


if __name__ == "__main__":
    main()
