"""
PHASE 1 — Scoping: clarifying the request before researching          (uses LLM)
=====================================================================================
OVERVIEW (what you will have when the phase is done)

    START → clarify ──(information missing)──► END             (it asks a QUESTION)
                    └─(all clear)────────────► write_brief → END

    Step 1 (NOW): the clarify node decides with the LLM and Command
    Step 2:       write_brief turns the conversation into a brief with the LLM
    Step 3:       multi-turn conversation with checkpointer and input()

CONCEPTS (you have practiced all of them already)

Structured output (M07) -> the LLM returns an object, not text
    decision_model = model.with_structured_output(ClarifyWithUser)
    decision = decision_model.invoke([HumanMessage(content=prompt)])
    decision.need_clarification   # True or False

Command (Exercise 10.3) -> choose the path and store a message at the same time
    return Command(goto=END, update={"messages": [AIMessage(content="...")]})

MessagesState (M05) -> a state that already brings `messages` with the add_messages reducer
    class ScopeState(MessagesState):
        research_brief: str

Messages to text (M01) -> a prompt only accepts text, the history is a list
    text = "\\n".join(f"{m.type}: {m.content}" for m in state["messages"])

The date in the prompt (M06) -> the LLM does not know what day it is
    datetime.now().strftime("%d/%m/%Y")   # returns a str, e.g. "12/10/2026"

EXERCISE — STEP 1

  1. Fix get_today (below): it must RETURN a str with the date (not the time).

  2. Create the schema ClarifyWithUser(BaseModel) with 3 fields, each with its
     Field(description=...):
        need_clarification: bool   -> use EXACTLY this description:
            "True ONLY if the topic, the place or the goal is missing. If the request
             already says what to research, False. When in doubt, False."
        question: str              -> the question ("" if not needed)
        verification: str          -> an "I'm starting to research" message ("" if a question is needed)

     (The description of need_clarification matters: with a simple description,
      GPT-4o asks almost every time, even with clear requests.)

  3. Create the state ScopeState(MessagesState) with research_brief: str

  4. Use this prompt (copy it as is):

        CLARIFY_PROMPT = '''These are the messages of the conversation with the user:
        <Messages>
        {messages}
        </Messages>

        Today is {date}.

        Decide whether you need to ask them a question before starting to research.
        - Ask ONLY if the topic, the place or the goal of the research is missing.
        - If the request already says what to research, do NOT ask: the researcher
          will decide the missing details.
        - Do not ask for facts that can be looked up on the internet.
        - If you ALREADY asked them a question before, do not ask again.
        - When in doubt, do NOT ask.
        Reply in the language the user used in their messages (English → English, Spanish → Spanish).'''

  5. Node clarify(state) -> Command[Literal["write_brief", "__end__"]]:
        - convert state["messages"] to text
        - fill in CLARIFY_PROMPT with .format(messages=..., date=...)
        - call the model with structured output (ClarifyWithUser)
        - if need_clarification -> goto END with an AIMessage with the question
        - otherwise              -> goto "write_brief" with an AIMessage with the verification

  6. PROVISIONAL write_brief node (in Step 2 the LLM will do it):
        return {"research_brief": "(pending)"}

  7. Graph: START → clarify, write_brief → END. No edges leave clarify.

Test with two requests (each one in its own invoke):
    vague = "Research coffee shops"
        → last message: a QUESTION (which city? which criterion?...)
    clear = "Compare price and quality of the 5 best-rated specialty coffee shops
             on Google Maps in Madrid in 2026"
        → last message: the verification, and research_brief = "(pending)"

    Print result["messages"][-1].content and result.get("research_brief").

Think about it: once it works, change the description of need_clarification to something
simple ("If you need to ask the user something") and run the clear request
3 times. What happens? Remember what you saw in M07.

Run:  uv run python en/07_deep_research/agent_phase1_part1.py
"""

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

def get_today() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y")

class ClarifyWithUser(BaseModel):
    """Decides whether the user must be asked a question before researching."""
    need_clarification: bool = Field(description="True ONLY if the topic, the place or the goal is missing. If the request already says what to research, False. When in doubt, False")
    question: str = Field(description="The question for the user, brief and in their language ('' if no question is needed)")
    verification: str = Field(description="Message saying you are starting to research, summarizing what you understood ('' if a question is needed)")

class ScopeState(MessagesState):
    research_brief: str

CLARIFY_PROMPT = '''These are the messages of the conversation with the user:
<Messages>
{messages}
</Messages>

Today is {date}.

Decide whether you need to ask them a question before starting to research.
- Ask ONLY if the topic, the place or the goal of the research is missing.
- If the request already says what to research, do NOT ask: the researcher
  will decide the missing details.
- Do not ask for facts that can be looked up on the internet.
- If you ALREADY asked them a question before, do not ask again.
- When in doubt, do NOT ask.
Reply in the language the user used in their messages (English → English, Spanish → Spanish).'''

def clarify(state: ScopeState) -> Command[Literal["write_brief", "__end__"]]:
    text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = CLARIFY_PROMPT.format(messages=text, date=get_today())
    decision_model = model.with_structured_output(ClarifyWithUser)
    decision = decision_model.invoke([HumanMessage(content=prompt)])
    if decision.need_clarification:
        return Command(goto=END, update={"messages":[AIMessage(content=decision.question)]})
    else:
        return Command(goto="write_brief", update={"messages":[AIMessage(content=decision.verification)]})

def write_brief(state: ScopeState) -> dict:
    return {"research_brief": "(pending)"}

graph = StateGraph(ScopeState)
graph.add_node("clarifyIntent", clarify)
graph.add_node("write_brief", write_brief)

graph.add_edge(START, "clarifyIntent")
graph.add_edge("write_brief", END)

app = graph.compile()

result = app.invoke({"messages": [HumanMessage('Research writers')]})
print(result["messages"][-1].content)
print(result.get("research_brief"))
result = app.invoke({"messages": [HumanMessage('Research Latin American writers who have won a Nobel Prize')]})
print(result["messages"][-1].content)
print(result.get("research_brief"))
