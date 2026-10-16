"""
EXERCISE 12 — think_tool: let the agent think between searches             (uses LLM)
=====================================================================================
CONCEPTS

think_tool -> a tool that DOES NOTHING: it returns the same thing it receives.
    @tool
    def think_tool(reflection: str) -> str:
        \"\"\"Use it AFTER each search to reflect...\"\"\"
        return f"Reflection recorded: {reflection}"

What is a tool that does nothing good for?
    To FORCE the LLM to write down its reasoning between searches:
        what did I find?  what am I missing?  do I search more or can I answer already?
    That reasoning is stored in the history (as a ToolMessage), so on the
    next turn the LLM reads it again and decides better.

    Without think_tool:  search → search → search → answer      (blindly)
    With think_tool:     search → think → search → think → answer

The docstring is the instruction (M06)
    The LLM doesn't see think_tool's code, only its name and its docstring.
    That's why the docstring says WHEN to use it ("after each search").
    The system prompt reinforces it.

Everything else is your Exercise 8: @tool, bind_tools, ToolNode, should_continue.

EXERCISE (solved example)

An agent that answers a question by searching the internet, and that reflects with
think_tool after each search.

    START → our_agent ──(asked for tools)───► tools ──► our_agent   (loop)
                      └─(already answering)─► END

    Tools: search (DDGS, like in your Exercise 8) and think_tool (NEW)

When you run it, look at the "Tool Message — think_tool" messages: that's where
what the agent thought before deciding the next step is.

Run:  uv run python en/07_deep_research/Exercise_12_0_think_tool.py
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


# 1. The state: same as in Exercise 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


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
).bind_tools(tools)


# 4. The system prompt reinforces WHEN to use each tool
SYSTEM_PROMPT = """You are a researcher. Answer the user's question by searching the internet.
- After EACH search, use think_tool to reflect before continuing.
- When you have enough information, answer without calling any more tools.
Reply in English."""


def model_call(state: AgentState) -> dict:
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


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


inputs = {"messages": [HumanMessage("Which Latin American writers have won the Nobel Prize in Literature?")]}
print_stream(app.stream(inputs, stream_mode="values"))
