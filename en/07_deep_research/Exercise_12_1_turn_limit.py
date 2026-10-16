"""
EXERCISE 12.1 — Putting a limit on the agent                               (uses LLM)
=====================================================================================
CONCEPTS

The problem
    In Exercise 12 the agent decides ON ITS OWN when to stop. It may search 2 times...
    or 10. Each turn is a call to the LLM (money and time). A research agent
    always needs a brake.

Counter in the state -> a field that adds 1 on each turn
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        turns: int

    def model_call(state):
        ...
        return {"messages": [response], "turns": state["turns"] + 1}

    (It's the same as the count in your Exercise_1, but next to messages.)

One model WITH tools and another WITHOUT tools
    llm = ChatOpenAI(...)                     # the "normal" model
    model_with_tools = llm.bind_tools(tools)
    model_without_tools = llm                 # no bind_tools: it CAN'T ask for tools

    If you call model_without_tools, the LLM has no choice but to ANSWER with what
    it already found. That way the agent always ends with an answer, not halfway.

EXERCISE

Copy your Exercise 12 and add a turn limit to it.

    START → our_agent ──(asked for tools)───► tools ──► our_agent
                      └─(already answering)─► END

    The graph does NOT change. Only what happens INSIDE our_agent changes:

        turns < MAX_TURNS   → uses model_with_tools      (can search and think)
        turns >= MAX_TURNS  → uses model_without_tools   (has to answer)

  1. Add turns: int to the state.
  2. Create llm, model_with_tools and model_without_tools (see CONCEPTS).
  3. Create the constant MAX_TURNS = 2.
  4. In model_call:
        - pick the model according to state["turns"]
        - print which turn it's on and whether it uses tools or not
        - return the message AND turns + 1
  5. should_continue and the graph do NOT change.
  6. In the invoke, start with the counter at 0:
        app.invoke({"turns": 0, "messages": [HumanMessage("...")]})
     At the end, print result["turns"] and the answer.

Expected output with MAX_TURNS = 2 (the tools may vary):
    --- turn 1 (with tools) -> asks for: ['search']
    --- turn 2 (with tools) -> asks for: ['think_tool']
    --- turn 3 (NO tools) -> asks for: []
    Turns: 3
    The Latin American writers who have won the Nobel Prize are: ...

    Hint for the print: response.tool_calls is a list of dicts (M03);
    [c["name"] for c in response.tool_calls] gives you the names.

Think about it: try MAX_TURNS = 0, 2 and 6 with the same question.
    - With 0, does it still answer? Where does it get the information from?
    - With 6, does it use all 6 turns, or does it finish earlier on its own?

Run:  uv run python en/07_deep_research/Exercise_12_1_turn_limit.py
"""
import os
from typing import Annotated, Sequence, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

MAX_TURNS = 2

# 1. The state: same as in Exercise 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turns: int


# 2. Search tool: your Exercise 8, with 3 results and error handling
@tool
def search(query: str) -> str:
    """Searches the internet for current information. Returns up to 3 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"The search failed: {e}"

    if not results:
        return "No results were found."

    return "\n\n".join(f"{r['title']}\n{r['body']}\n({r['href']})" for r in results)


# 3. THE NEW PART: think_tool. It doesn't search anything, doesn't compute anything: it only "records" what it thought.
@tool
def think_tool(reflection: str) -> str:
    """Use it AFTER each search to reflect before continuing:
    what information you found, what you are still missing, and whether you need
    to search more or can already answer."""
    return f"Reflection recorded: {reflection}"


tools = [search, think_tool]

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

model_with_tools = model.bind_tools(tools)
model_without_tools = model


# 4. The system prompt reinforces WHEN to use each tool
SYSTEM_PROMPT = """You are a researcher. Answer the user's question by searching the internet.
- After EACH search, use think_tool to reflect before continuing.
- When you have enough information, answer without calling any more tools.
Reply in English."""


def model_call(state: AgentState) -> dict:
    if state["turns"] >= MAX_TURNS:
        chosen_model = model_without_tools
    else:
        chosen_model = model_with_tools

    response = chosen_model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    print(f"--- turn {state['turns'] + 1} -> asks for: {[c['name'] for c in response.tool_calls]}")
    return {"messages": [response], "turns": state['turns'] + 1}


def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# 5. The graph: identical to the one in Exercise 8
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "our_agent")

app = graph.compile()


# 6. Try it: prints each message as it is generated
def print_stream(stream):
    for event in stream:
        event["messages"][-1].pretty_print()


inputs = {"turns": 0, "messages": [HumanMessage("Which Latin American writers have won the Nobel Prize in Literature?")]}
print_stream(app.stream(inputs, stream_mode="values"))

result = app.invoke(inputs)
print("Turns:", result["turns"])

