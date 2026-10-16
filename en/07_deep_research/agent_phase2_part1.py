"""
PHASE 2 — STEP 1: the researcher                                       (uses LLM)
=====================================================================================
WHERE YOU ARE

    PHASE 1 ✅  conversation → brief                   (agent_phase1_part1/2/3.py)
    PHASE 2     brief → research ⇄ tools → compress → notes with sources
                        ▲
                        └── STEP 1 (NOW): the agent that researches from the brief

    The code below is a COPY of your Exercise 12.1. It already has almost everything:
    search, think_tool, the loop with should_continue and the turn limit.

CONCEPTS

Two brakes: a soft one and a hard one
    Soft (prompt):      "simple questions: 2-3 searches; complex: up to 5"
                        The LLM respects it almost always, but it is not guaranteed.
    Hard (MAX_TURNS):   your counter from 12.1. Whatever happens, on turn
                        MAX_TURNS the agent answers without tools.

    The reference project uses both: the prompt guides and the code guarantees.

The brief is the input
    In 12.1 the agent received a short question. Now it receives the brief from
    Phase 1: a long text with everything the user wants to know.

        inputs = {"turns": 0, "messages": [HumanMessage(content=BRIEF)]}

The researcher system prompt
    It explains HOW to research: read the brief, search from the general to the
    specific, think after each search, when to stop and how to deliver the answer.

EXERCISE — STEP 1

  1. Add the get_today() helper from Phase 1 (returns "13/10/2026").

  2. Replace SYSTEM_PROMPT with this RESEARCHER_PROMPT (copy it as is):

        RESEARCHER_PROMPT = '''You are a research assistant. Today is {date}.
        You will receive a brief with what the user wants to research.
        Use search to look for information and think_tool to think between searches.

        How to research:
        1. Read the brief: what specific information do you need?
        2. Start with broad searches and then make them more specific.
        3. After EACH search, use think_tool: what did I find? what am I missing?
           do I keep searching or can I answer already?
        4. Stop when you can answer the brief well, or when your last
           2 searches add nothing new.

        Limits:
        - Simple questions: 2-3 searches at most.
        - Complex questions: up to 5 searches.

        Final answer: in the language of the brief (English → English, Spanish → Spanish),
        with the information found and the URLs of the sources you used.'''

     In model_call, fill it in with .format(date=get_today()) before creating the
     SystemMessage.

  3. Raise MAX_TURNS to 10 (5 searches + 5 reflections at most).

  4. Create a BRIEF constant with a brief like the one your Phase 1 generates:

        BRIEF = '''I want to research the Latin American writers who have won
        the Nobel Prize in Literature: who they are, from which country, in which year
        they won it and for which works. I have no preference of sources.'''

     and use it in inputs instead of the Nobel question.

  5. When it finishes, print how many turns it used and the final answer
     (result["turns"] and result["messages"][-1].content).

  The graph, search, think_tool and should_continue do NOT change.

What to check in the output:
    - How many searches did it make? Did it respect the prompt's soft limit?
    - Does the final answer bring the URLs of the sources?
    - Does it answer EVERYTHING in the brief (who, country, year, works)?

Think about it: change the brief to a complex one (for example, "compare the education
systems of Finland and Japan: results, class hours and teacher training").
Does it make more searches than with the Nobel one?

Run:  uv run python en/07_deep_research/agent_phase2_part1.py
"""
import os
from datetime import datetime
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

MAX_TURNS = 10

# 1. The state: same as in Exercise 8
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turns: int

# Helpers
def get_today() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y")


# 2. Search tool: your Exercise 8, with 3 results and error handling

@tool
def search(query: str) -> str:
    """Searches for current information on the internet. Returns up to 3 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return f"The search failed: {e}"

    if not results:
        return "No results found."

    return "\n\n".join(f"{r['title']}\n{r['body']}\n({r['href']})" for r in results)


# 3. WHAT'S NEW: think_tool. It searches nothing, computes nothing: it only "records" what it thought.
@tool
def think_tool(reflection: str) -> str:
    """Use it AFTER each search to reflect before going on:
    what information you found, what you are still missing, and whether you need
    to search more or can answer already."""
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
RESEARCHER_PROMPT = '''You are a research assistant. Today is {date}.
    You will receive a brief with what the user wants to research.
    Use search to look for information and think_tool to think between searches.

    How to research:
    1. Read the brief: what specific information do you need?
    2. Start with broad searches and then make them more specific.
    3. After EACH search, use think_tool: what did I find? what am I missing?
        do I keep searching or can I answer already?
    4. Stop when you can answer the brief well, or when your last
        2 searches add nothing new.

    Limits:
    - Simple questions: 2-3 searches at most.
    - Complex questions: up to 5 searches.

    Final answer: in the language of the brief (English → English, Spanish → Spanish),
    with the information found and the URLs of the sources you used.'''

BRIEF = '''I want to research the Latin American writers who have won
    the Nobel Prize in Literature: who they are, from which country, in which year
    they won it and for which works. I have no preference of sources.'''

def model_call(state: AgentState) -> dict:
    if state["turns"] >= MAX_TURNS:
        chosen_model = model_without_tools
    else:
        chosen_model = model_with_tools

    prompt = RESEARCHER_PROMPT.format(date=get_today())

    response = chosen_model.invoke([SystemMessage(content=prompt)] + state["messages"])
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


inputs = {"turns": 0, "messages": [HumanMessage(BRIEF)]}
print_stream(app.stream(inputs, stream_mode="values"))

# result = app.invoke(inputs)
# print("Turns:", result["turns"])
# print("Result:", result["messages"][-1].content)

