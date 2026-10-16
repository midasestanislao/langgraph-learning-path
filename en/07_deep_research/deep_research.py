"""
DEEP RESEARCH — the complete agent                                     (uses LLM)
=====================================================================================
WHERE EACH PIECE COMES FROM

    PHASE 1  conversation → brief           agent_phase1_part3.py   (import scope_app)
    PHASE 2  topic → notes with sources     agent_phase2_part2.py   (import researcher_app)
    PHASE 4  supervisor that delegates      Exercise_16_0 + 16_1    (copied here, with think_tool)
    PHASE 5  notes → final report           NEW: the write_report node

THE COMPLETE FLOW

    main()
      │  PHASE 1: loop with input() until we have the brief   (as in agent_phase2_part3.py)
      ▼
    brief ──► RESEARCH GRAPH (a single app):
                START → supervisor ⇄ supervisor_tools ──► write_report → END
                                          │ for each ConductResearch:
                                          └─► researcher_app (the REAL one, Phase 2) in parallel

    Difference from 16.0: when the supervisor finishes, supervisor_tools no longer goes
    to END but to "write_report", which turns all the notes into a report.

THE ONLY NEW THING: the write_report node
    It receives the brief and ALL the notes (with their [n] and their Sources) and asks
    the LLM for a Markdown report that answers the brief citing the sources. It is free
    text (a person reads it), so model without tools and .content, as in compress.
    It saves it to outputs/report_<date>.md, as you did in the newsletter.

This file keeps the explanations with which each part was built. The clean English
version of the finished agent lives in its own repository: deep-research-agent
(https://github.com/midasestanislao/deep-research-agent).

Run:  uv run python en/07_deep_research/deep_research.py
    User: Compare the education systems of Finland, Japan and Chile: academic results and teacher training
"""
import asyncio
import operator
import os
import time
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

# ============ 1. THE PHASES YOU ALREADY BUILT ============
from agent_phase1_part3 import app as scope_app              # Phase 1: conversation → brief
from agent_phase2_part2 import app as researcher_app         # Phase 2: topic → notes with sources

load_dotenv()

MAX_ROUNDS = 6              # the supervisor's hard brake (the thinking rounds count too)
MAX_RESEARCHERS = 3         # soft brake: we tell the LLM in the prompt


def get_today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ============ 2. THE SUPERVISOR'S TOOLS (Exercises 13 and 16.1) ============
class ConductResearch(BaseModel):
    """Delegates the research of ONE topic to a researcher. Use it once per distinct topic."""
    research_topic: str = Field(description="The topic to research, described in detail in one sentence, in the language of the brief.")


class ResearchComplete(BaseModel):
    """Use it when the notes received already cover the brief and no more research is needed."""


@tool
def think_tool(reflection: str) -> str:
    """Use it to think BEFORE delegating (which topics do I need?) and AFTER receiving
    notes (do they cover the brief? what is missing?). Write your reasoning in reflection."""
    return f"Reflection recorded: {reflection}"


supervisor_model = llm.bind_tools([ConductResearch, ResearchComplete, think_tool])

SUPERVISOR_PROMPT = """You are a research supervisor. Today is {date}.
You receive a brief. You do NOT research: you DELEGATE with ConductResearch, one form per topic.

How to work:
- Use think_tool before delegating and after receiving notes, to reason about your decision.
- Split the brief into FEW independent topics. In a comparison, one topic per compared
  element (for example, a country), and that topic must include ALL the aspects
  the brief asks for. Do not make one topic per aspect.
- Simple questions: 1 or 2 topics. At most {max_researchers} topics per round.
- When you receive the researchers' notes, check whether they cover the brief.
  If something important is missing, delegate ONLY what is missing. If it is already covered, use ResearchComplete.
- Do not repeat a topic that already has notes."""

REPORT_PROMPT = """You are a writer. Today is {date}.
Write a report in Markdown that answers this brief:

{brief}

Use ONLY the information in these notes (each note brings its numbered sources):

{notes}

Rules:
- Structure: title, a brief introduction, one section per topic or compared element,
  and a conclusion that answers the brief directly.
- Cite the sources with [n] after each fact, and end with a "Sources" section
  with the list of URLs. If two notes repeat the same URL, include it only once.
- If the notes do not cover something the brief asks for, say so explicitly. Do not invent.
- Write in the language of the brief (English → English, Spanish → Spanish)."""


# ============ 3. THE STATE ============
class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # brief + forms + notes (ToolMessages)
    notes: Annotated[list[str], operator.add]                 # accumulate round after round
    rounds: int
    report: str
    file: str


# ============ 4. supervisor NODE (Exercise 16) ============
async def supervisor(state: ResearchState) -> Command[Literal["supervisor_tools"]]:
    prompt = SUPERVISOR_PROMPT.format(date=get_today(), max_researchers=MAX_RESEARCHERS)
    response = await supervisor_model.ainvoke([SystemMessage(content=prompt)] + state["messages"])

    print(f"\n--- Round {state['rounds'] + 1}: the supervisor asks for {[c['name'] for c in response.tool_calls]}")
    return Command(
        goto="supervisor_tools",
        update={"messages": [response], "rounds": state["rounds"] + 1},
    )


# ============ 5. supervisor_tools NODE (Exercise 16.1, with REAL researchers) ============
async def supervisor_tools(state: ResearchState) -> Command[Literal["supervisor", "write_report"]]:
    last = state["messages"][-1]
    requests = [c for c in last.tool_calls if c["name"] == "ConductResearch"]
    reflections = [c for c in last.tool_calls if c["name"] == "think_tool"]
    finished = any(c["name"] == "ResearchComplete" for c in last.tool_calls)

    # a) Should we stop? If so, we no longer go to END: we go write the report
    if finished:
        print("    Done: the supervisor declared the research finished")
        return Command(goto="write_report", update={})
    if not last.tool_calls:
        print("    Done: the supervisor asked for nothing")
        return Command(goto="write_report", update={})
    if state["rounds"] >= MAX_ROUNDS:
        print(f"    Done: reached the limit of {MAX_ROUNDS} rounds (hard brake)")
        return Command(goto="write_report", update={})

    # b) The reflections are run by hand (M03) and come back as ToolMessages
    tool_messages = []
    for r in reflections:
        print(f"    💭 {r['args']['reflection'][:120]}")
        tool_messages.append(ToolMessage(content=think_tool.invoke(r["args"]), tool_call_id=r["id"]))
    if not requests:                                  # it only thought: another round without researching
        return Command(goto="supervisor", update={"messages": tool_messages})

    # c) One REAL researcher (Phase 2) per form, all at once (Exercise 15)
    print(f"    Launching {len(requests)} researchers at once...")
    start = time.perf_counter()
    tasks = [
        researcher_app.ainvoke({"turns": 0, "messages": [HumanMessage(content=c["args"]["research_topic"])]})
        for c in requests
    ]
    results = await asyncio.gather(*tasks)

    # d) Their notes go back to the supervisor as ToolMessages and accumulate in the state
    for result, request in zip(results, requests):
        print(f"    ✓ {request['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=result["notes"], tool_call_id=request["id"]))
    print(f"    ({time.perf_counter() - start:.0f} s)")

    return Command(
        goto="supervisor",
        update={"messages": tool_messages, "notes": [r["notes"] for r in results]},
    )


# ============ 6. write_report NODE (NEW: Phase 5) ============
def write_report(state: ResearchState) -> dict:
    brief = state["messages"][0].content                 # the first message is the brief
    prompt = REPORT_PROMPT.format(
        date=get_today(),
        brief=brief,
        notes="\n\n---\n\n".join(state["notes"]),
    )
    print("\n📝 Writing the report...")
    report = llm.invoke([HumanMessage(content=prompt)]).content

    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/report_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return {"report": report, "file": path}


# ============ 7. THE RESEARCH GRAPH ============
graph = StateGraph(ResearchState)
graph.add_node("supervisor", supervisor)
graph.add_node("supervisor_tools", supervisor_tools)
graph.add_node("write_report", write_report)
graph.add_edge(START, "supervisor")
# supervisor ⇄ supervisor_tools and supervisor_tools → write_report are decided by the Commands
graph.add_edge("write_report", END)
research_app = graph.compile()


# ============ 8. main(): PHASE 1 with input(), and then the research graph ============
async def main():
    config = {"configurable": {"thread_id": "1"}}
    brief = None

    while True:
        text = input("User: ").strip()
        if not text:
            continue
        elif text.lower() == "exit":
            break

        result = scope_app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"AI: {result['messages'][-1].content}")

        if "research_brief" in result:
            brief = result["research_brief"]
            print(f"\nBRIEF:\n{brief}")
            break

    if brief is None:
        print("No brief, nothing to research. See you!")
        return

    print("\n🔎 Researching (this takes a few minutes)...")
    start = time.perf_counter()
    result = await research_app.ainvoke(
        {"messages": [HumanMessage(content=brief)], "notes": [], "rounds": 0}
    )

    print(f"\n{'=' * 70}\n{result['report']}\n{'=' * 70}")
    print(f"\n💾 Report saved to {result['file']}  ({time.perf_counter() - start:.0f} s in total)")


if __name__ == "__main__":
    asyncio.run(main())
