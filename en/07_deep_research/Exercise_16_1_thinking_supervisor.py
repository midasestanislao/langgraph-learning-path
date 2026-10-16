"""
EXERCISE 16.1 — A supervisor that thinks before delegating             (uses LLM: only the supervisor)
=====================================================================================
CONCEPTS

Everything is from Exercise 16. What's new is giving the supervisor a think_tool (Exercise 12)
and making supervisor_tools know what to do with it.

Three kinds of form in the same response
    Now a tool_call can be ConductResearch, ResearchComplete or think_tool.
    Separate them by name (as in 13.2):
        requests    = [c for c in last.tool_calls if c["name"] == "ConductResearch"]
        reflections = [c for c in last.tool_calls if c["name"] == "think_tool"]

Running a tool by hand (M03 / M06)
    think_tool DOES have code (it returns "Reflection recorded: ..."). Since there is
    no ToolNode here, you run it yourself:
        observation = think_tool.invoke(r["args"])                  # r is the tool_call
        ToolMessage(content=observation, tool_call_id=r["id"])      # and you hand it back to the LLM

A round can be ONLY for thinking
    If the supervisor only asked for think_tool (no ConductResearch), there is nothing
    to research: you return the reflections' ToolMessages and go back to "supervisor"
    so it can decide with a cool head. Watch out for the "if not requests" brake from
    16.0! Now "no requests" does NOT mean "finish".

EXERCISE

Copy your Exercise_16_0 and make these changes:

  1. Import tool (from langchain_core.tools import tool) and define think_tool as in
     Exercise 12, but with a docstring for a SUPERVISOR: use it BEFORE delegating
     (which topics do I need?) and AFTER receiving notes (do they cover the brief?).

  2. Add it to bind_tools: [ConductResearch, ResearchComplete, think_tool].
     And in SUPERVISOR_PROMPT, one more line: "Use think_tool before delegating and
     after receiving notes, to reason about your decision."

  3. Raise MAX_ROUNDS to 6: the thinking rounds count too.

  4. In supervisor_tools:
        a) separate the tool_calls into requests (ConductResearch) and reflections (think_tool)
        b) change the "if not requests" brake to "if not last.tool_calls"
           (finish only if it asked for NOTHING)
        c) run each reflection with think_tool.invoke(...), print the reflection's
           text and create its ToolMessage
        d) if there are no requests (it only thought): go back to "supervisor" with those ToolMessages
        e) if there are requests: launch the researchers as in 16.0 and add their
           ToolMessages to the SAME list, before the return

Expected output (the reflection texts vary):
    --- Round 1: the supervisor asks for ['think_tool']
        💭 The brief asks for a comparison of the education systems of Finland, Japan and Chile...
    --- Round 2: the supervisor asks for ['ConductResearch', 'ConductResearch', 'ConductResearch']
        Launching 3 researchers at once...
        ✓ Education system of Finland: academic results, weekly class hours and teacher training...
        ✓ Education system of Japan: ...
        ✓ Education system of Chile: ...
        (1.0 s)
    --- Round 3: the supervisor asks for ['think_tool']
        💭 I have received the notes on the education systems of Finland, Japan and Chile, covering...
    --- Round 4: the supervisor asks for ['ResearchComplete']
        Done: the supervisor declared the research finished

    ACCUMULATED NOTES (3): ...

Think about it:
    - Now there are 4 LLM calls instead of 2. What do you gain in exchange? Read the two
      reflections: would the supervisor notice if the notes were missing a country?
    - What would happen if in step 4b you kept the 16.0 brake ("if not requests")?
      Trace round 1 by hand.

Run:  uv run python en/07_deep_research/Exercise_16_1_thinking_supervisor.py
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

load_dotenv()

MAX_ROUNDS = 6              # hard brake: maximum number of calls to the supervisor
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

@tool
def think_tool(reflection: str) -> str:
    """Use it to think BEFORE delegating (which topics do I need?) and AFTER receiving
    notes (do they cover the brief? what is missing?). Write your reasoning in reflection."""
    return f"Reflection recorded: {reflection}"


llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

supervisor_model = llm.bind_tools([ConductResearch, ResearchComplete, think_tool])

SUPERVISOR_PROMPT = """You are a research supervisor. Today is {date}.
You receive a brief. You do NOT research: you DELEGATE with ConductResearch, one form per topic.

How to work:
- Split the brief into FEW independent topics. In a comparison, one topic per compared
  element (for example, a country), and that topic must include ALL the aspects
  the brief asks for. Do not make one topic per aspect.
- Simple questions: 1 or 2 topics. At most {max_researchers} topics per round.
- When you receive the researchers' notes, check whether they cover the brief.
  If something important is missing, delegate ONLY what is missing. If it is already covered, use ResearchComplete.
- Do not repeat a topic that already has notes.
- Use think_tool before delegating and after receiving notes, to reason about your decision."""



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
    last = state["messages"][-1]
    # 4a. Three kinds of form, separated by name
    requests = [c for c in last.tool_calls if c["name"] == "ConductResearch"]
    reflections = [c for c in last.tool_calls if c["name"] == "think_tool"]
    finished = any(c["name"] == "ResearchComplete" for c in last.tool_calls)

    # a) Should we stop?
    if finished:
        print("    Done: the supervisor declared the research finished")
        return Command(goto=END, update={})
    if not last.tool_calls:                          # 4b. finish only if it asked for NOTHING
        print("    Done: the supervisor asked for nothing")
        return Command(goto=END, update={})
    if state["rounds"] >= MAX_ROUNDS:
        print(f"    Done: reached the limit of {MAX_ROUNDS} rounds (hard brake)")
        return Command(goto=END, update={})

    # 4c. The reflections are run by hand (M03/M06) and come back as ToolMessages
    tool_messages = []
    for r in reflections:
        observation = think_tool.invoke(r["args"])
        print(f"    💭 {r['args']['reflection'][:110]}")
        tool_messages.append(ToolMessage(content=observation, tool_call_id=r["id"]))

    # 4d. If it only thought, there is nothing to research: another round
    if not requests:
        return Command(goto="supervisor", update={"messages": tool_messages})

    # b) One researcher per form, all at once (Exercise 15)
    print(f"    Launching {len(requests)} researchers at once...")
    start = time.perf_counter()
    tasks = [researcher_app.ainvoke({"topic": c["args"]["research_topic"]}) for c in requests]
    results = await asyncio.gather(*tasks)

    # c) The notes are added to the SAME list as the reflections (4e: no tool_messages = [] here)
    for result, request in zip(results, requests):
        print(f"    ✓ {request['args']['research_topic']}")
        tool_messages.append(ToolMessage(content=result["notes"], tool_call_id=request["id"]))
    print(f"    ({time.perf_counter() - start:.1f} s)")

    return Command(
        goto="supervisor",
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
