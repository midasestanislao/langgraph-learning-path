"""
EXERCISE 16 — The toy supervisor: all the pieces together             (uses LLM: only the supervisor)
=====================================================================================
CONCEPTS (none new: this is an ASSEMBLY of what you already know)

    13   ConductResearch / ResearchComplete -> the forms the supervisor fills in
    10/11 Command -> the supervisor ⇄ supervisor_tools loop, without add_conditional_edges
    14/15 gather of researcher_app.ainvoke -> one researcher per form, all at once
    M03  ToolMessage(content=..., tool_call_id=...) -> handing the notes back to the supervisor
    M05  notes: Annotated[list[str], operator.add] -> the notes ACCUMULATE round after round

The loop

    START → supervisor ──► supervisor_tools ──(there is a ConductResearch)──► supervisor   (another round)
                                             └─(ResearchComplete, no forms
                                                or no rounds left)──────────────────► END

    supervisor:        the LLM reads the brief (and the notes so far) and fills in forms
    supervisor_tools:  your code reads the forms and launches one researcher per form

How the supervisor "sees" what the researchers did
    Each ConductResearch is a tool_call with its "id". When the researcher finishes,
    its notes come back as ToolMessage(content=notes, tool_call_id=id), just like in M03.
    In the next round the LLM reads those notes and decides: is something missing? or is it done?

Two brakes (as in Phase 2)
    Soft: the prompt says how many topics to delegate and when to stop.
    Hard: MAX_ROUNDS. In round MAX_ROUNDS, supervisor_tools finishes no matter what.

EXERCISE (solved example)

Brief: compare the education systems of Finland, Japan and Chile.
The supervisor should delegate 3 topics (one per country) in the first round, receive
the 3 notes at once (1 second, not 3) and in the second round decide whether to finish.

Expected output (the LLM may vary in the texts and in asking for one more round):
    --- Round 1: the supervisor asks for ['ConductResearch', 'ConductResearch', 'ConductResearch']
        Launching 3 researchers at once...
        ✓ Education system of Finland: results, class hours and teacher training
        ✓ Education system of Japan: ...
        ✓ Education system of Chile: ...
        (1.0 s)
    --- Round 2: the supervisor asks for ['ResearchComplete']
        Done: the supervisor declared the research finished

    ACCUMULATED NOTES (3):
    - Notes on Education system of Finland...: fact A..., fact B...
    - ...

Think about it:
    - Set MAX_ROUNDS to 1. What happens? Who cuts the loop, the LLM or your code?
    - Change the brief to a simple one ("who invented espresso?"). How many
      ConductResearch does it ask for? Does launching researchers in parallel still make sense?
    - Real lesson of this exercise: the first version of the prompt said
      "one topic per compared thing". The LLM read it as one topic per ASPECT
      (results, hours, teachers) × country = 9 topics, 3 per round, and the hard brake
      cut it off in round 3 without finishing. With "one topic per country, with ALL the
      aspects" it makes 3 topics and closes in round 2. The supervisor's prompt decides
      how much work (and money) it generates.

Run:  uv run python en/07_deep_research/Exercise_16_0_toy_supervisor.py
"""
import asyncio
import operator
import os
import time
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

MAX_ROUNDS = 3              # hard brake: maximum number of calls to the supervisor
MAX_RESEARCHERS = 3         # soft brake: we tell the LLM in the prompt


def get_today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


# ============ 1. TOY RESEARCHER (your 14.1, unchanged) ============
class ResearcherState(TypedDict):
    topic: str
    results: list[str]
    notes: str


def fake_search(state: ResearcherState) -> dict:
    time.sleep(1)       # simulates the search
    return {"results": [f"fact A about {state['topic']}", f"fact B about {state['topic']}"]}


def fake_compress(state: ResearcherState) -> dict:
    return {"notes": f"Notes on {state['topic']}: " + ", ".join(state["results"])}


researcher_graph = StateGraph(ResearcherState)
researcher_graph.add_node("fake_search", fake_search)
researcher_graph.add_node("fake_compress", fake_compress)
researcher_graph.add_edge(START, "fake_search")
researcher_graph.add_edge("fake_search", "fake_compress")
researcher_graph.add_edge("fake_compress", END)
researcher_app = researcher_graph.compile()


# ============ 2. THE SUPERVISOR'S FORMS (Exercise 13) ============
class ConductResearch(BaseModel):
    """Delegates the research of ONE topic to a researcher. Use it once per distinct topic."""
    research_topic: str = Field(description="The topic to research, described in detail in one sentence.")


class ResearchComplete(BaseModel):
    """Use it when the notes received already cover the brief and no more research is needed."""


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
supervisor_model = llm.bind_tools([ConductResearch, ResearchComplete])

SUPERVISOR_PROMPT = """You are a research supervisor. Today is {date}.
You receive a brief. You do NOT research: you DELEGATE with ConductResearch, one form per topic.

How to work:
- Split the brief into FEW independent topics. In a comparison, one topic per compared
  element (for example, a country), and that topic must include ALL the aspects
  the brief asks for. Do not make one topic per aspect.
- Simple questions: 1 or 2 topics. At most {max_researchers} topics per round.
- When you receive the researchers' notes, check whether they cover the brief.
  If something important is missing, delegate ONLY what is missing. If it is already covered, use ResearchComplete.
- Do not repeat a topic that already has notes."""


# ============ 3. THE SUPERVISOR'S STATE ============
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   # brief + forms + notes (ToolMessages)
    notes: Annotated[list[str], operator.add]                 # accumulate round after round (M05)
    rounds: int


# ============ 4. supervisor NODE: the LLM fills in forms ============
async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    prompt = SUPERVISOR_PROMPT.format(date=get_today(), max_researchers=MAX_RESEARCHERS)
    response = await supervisor_model.ainvoke([SystemMessage(content=prompt)] + state["messages"])

    print(f"--- Round {state['rounds'] + 1}: the supervisor asks for {[c['name'] for c in response.tool_calls]}")
    return Command(
        goto="supervisor_tools",
        update={"messages": [response], "rounds": state["rounds"] + 1},
    )


# ============ 5. supervisor_tools NODE: your code reads the forms and acts ============
async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    last = state["messages"][-1]                          # the AIMessage with the forms
    requests = [c for c in last.tool_calls if c["name"] == "ConductResearch"]
    finished = any(c["name"] == "ResearchComplete" for c in last.tool_calls)

    # a) Should we stop? (the two brakes + the "asked for nothing" case)
    if finished:
        print("    Done: the supervisor declared the research finished")
        return Command(goto=END, update={})
    if not requests:
        print("    Done: the supervisor did not ask for more research")
        return Command(goto=END, update={})
    if state["rounds"] >= MAX_ROUNDS:
        print(f"    Done: reached the limit of {MAX_ROUNDS} rounds (hard brake)")
        return Command(goto=END, update={})

    # b) One researcher per form, all at once (Exercise 15)
    print(f"    Launching {len(requests)} researchers at once...")
    start = time.perf_counter()
    tasks = [researcher_app.ainvoke({"topic": c["args"]["research_topic"]}) for c in requests]
    results = await asyncio.gather(*tasks)

    # c) Hand the notes back to the supervisor as ToolMessages (M03): one ToolMessage per form,
    #    linked by tool_call_id. zip pairs each result with its form (same order).
    tool_messages = []
    for result, request in zip(results, requests):
        print(f"    ✓ {request['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=result["notes"], tool_call_id=request["id"]))
    print(f"    ({time.perf_counter() - start:.1f} s)")

    return Command(
        goto="supervisor",                                            # another round: the LLM will read the notes
        update={"messages": tool_messages, "notes": [r["notes"] for r in results]},
    )


# ============ 6. THE GRAPH: a single edge; the Commands make the loop (Exercise 11) ============
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("supervisor_tools", supervisor_tools)
graph.add_edge(START, "supervisor")
supervisor_app = graph.compile()


# ============ 7. Try it ============
BRIEF = """I want to compare the education systems of Finland, Japan and Chile: academic
results, weekly class hours and teacher training."""


async def main():
    result = await supervisor_app.ainvoke(
        {"messages": [HumanMessage(content=BRIEF)], "notes": [], "rounds": 0}
    )
    print(f"\nACCUMULATED NOTES ({len(result['notes'])}):")
    for note in result["notes"]:
        print(f"- {note}")


if __name__ == "__main__":
    asyncio.run(main())
