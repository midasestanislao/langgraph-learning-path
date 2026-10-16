"""
PHASE 1 — STEP 2: the LLM writes the research brief                    (uses LLM)
=====================================================================================
WHERE YOU ARE

    START → clarify ──(information missing)──► END
                    └─(all clear)────────────► write_brief → END
                                               ▲
                                               └── NOW: this node uses the LLM

    Step 1 ✅  clarify decides with the LLM and Command      (agent_phase1_part1.py)
    Step 2 ⬅️  write_brief turns the conversation into a brief
    Step 3     multi-turn conversation with checkpointer and input()

WHAT IS THE RESEARCH BRIEF?

    It is the conversation with the user turned into ONE detailed research
    question. In Phase 2 the researcher will only receive this text, not the
    conversation, so it has to include EVERYTHING the user asked for.

    User:   "Research Latin American writers who have won a Nobel"
    Brief:  "I want to know the Latin American writers who have won the Nobel
             Prize in Literature: who they are, in which year they won it and for
             which works. I have not indicated a period, so consider all of them."

CONCEPTS (none new)

Structured output (M07) -> same as in clarify, but with another class
    class ResearchQuestion(BaseModel):
        research_brief: str = Field(description="...")

    brief_model = model.with_structured_output(ResearchQuestion)
    answer = brief_model.invoke([HumanMessage(content=prompt)])
    answer.research_brief      # the brief's text

Normal node vs. node with Command
    clarify      -> chooses the path    -> returns Command(goto=..., update=...)
    write_brief  -> always goes to END  -> returns a dict: {"research_brief": ...}

EXERCISE — STEP 2

  1. Create the schema ResearchQuestion(BaseModel) with a single field:
        research_brief: str  -> describe that it is ONE detailed research question,
                                in first person and in the user's language

  2. Use this prompt (copy it as is):

        BRIEF_PROMPT = '''These are the messages of the conversation with the user:
        <Messages>
        {messages}
        </Messages>

        Today is {date}.

        Turn this conversation into ONE detailed research question.
        - Include ALL the details and preferences the user gave.
        - Do not invent preferences the user did not state. If a detail is missing
          (dates, budget, sources...), say it is left open.
        - Write it in first person, as if the user were asking it.
        - Write it in the language the user used in their messages (English → English, Spanish → Spanish).'''

  3. Change the write_brief node so that it uses the LLM:
        - convert state["messages"] to text (the same line as in clarify)
        - fill in BRIEF_PROMPT with .format(messages=..., date=...)
        - call the model with structured output (ResearchQuestion)
        - return {"research_brief": answer.research_brief}

  4. The graph does NOT change.

Test with the Nobel request and print result.get("research_brief").
Check the brief:
    - Does it include everything you asked for?
    - Did it invent any preference you did not state?

Think about it: the messages that write_brief reads include the verification that
clarify stored ("I'm going to research..."). Why? (Hint: the Command's update is
applied BEFORE the next node runs — Exercise 10.2.)

Run:  uv run python en/07_deep_research/agent_phase1_part2.py
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

class ResearchQuestion(BaseModel):
    """Detailed research question that will guide the researcher."""
    research_brief: str = Field(description="One detailed research question, in first person and in the user's language")

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

BRIEF_PROMPT = '''These are the messages of the conversation with the user:
<Messages>
{messages}
</Messages>

Today is {date}.

Turn this conversation into ONE detailed research question.
- Include ALL the details and preferences the user gave.
- Do not invent preferences the user did not state. If a detail is missing
    (dates, budget, sources...), say it is left open.
- Write it in first person, as if the user were asking it.
- Write it in the language the user used in their messages (English → English, Spanish → Spanish).'''

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
    text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = BRIEF_PROMPT.format(messages=text, date=get_today())
    brief_model = model.with_structured_output(ResearchQuestion)
    answer = brief_model.invoke([HumanMessage(content=prompt)])
    return {"research_brief": answer.research_brief}

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
