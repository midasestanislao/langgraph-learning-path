"""
PHASE 2 — STEP 2: compressing the research                             (uses LLM)
=====================================================================================
WHERE YOU ARE

    brief → our_agent ⇄ tools ──► compress ──► END
                                   ▲
                                   └── STEP 2 (NOW): a new node at the end

    Step 1 ✅  the researcher searches, thinks and answers     (agent_phase2_part1.py)
    Step 2 ⬅️  compress: turn the research into clean notes with sources
    Step 3     join Phase 1 and Phase 2

    The code below is a COPY of your Step 1.

CONCEPTS

Why compress?
    When it finishes, state["messages"] has everything: the brief, tool calls,
    repeated search results, think_tool reflections and the final answer. In
    Phase 4 a supervisor will launch SEVERAL researchers; if each one hands back
    all that history, the context explodes.
    That is why each researcher delivers only its NOTES: clean, without repetition
    and with numbered sources.

Why not use the final answer?
    The agent's final answer is a summary: it may leave out facts that it DID find
    in the searches. compress works on the RAW results of search, so nothing is lost.

filter_messages (M04) -> keep only the search results
    from langchain_core.messages import filter_messages

    results = filter_messages(state["messages"], include_names=["search"])
    # → only the ToolMessages from search (no think_tool, no AI messages)

The brief is in the first message
    state["messages"][0].content    # the HumanMessage that started it all

Free text, not structured output
    The notes will be read by another LLM (the supervisor or the report writer), not
    by your code. So model_without_tools.invoke(...).content is enough (M07, rule:
    if the data is used by YOUR CODE → structured output; if an LLM or a person
    reads it → text).

EXERCISE — STEP 2

  1. Add notes: str to the state.

  2. Use this prompt (copy it as is):

        COMPRESS_PROMPT = '''You are an assistant that cleans up the results of a research.
        Today is {date}.

        Research brief:
        {brief}

        Search results:
        {results}

        Write some CLEAN NOTES:
        - Keep ALL the facts relevant to the brief (names, dates, figures).
          Do not over-summarize.
        - Remove what is repeated and what has nothing to do with the brief.
        - After each fact, indicate its source with a number in square brackets, e.g. [1].
        - At the end, a "Sources" section with the numbered list of URLs.
        - Do not invent anything that is not in the results.'''

  3. Create the node compress(state) -> dict:
        - take the search results out with filter_messages
        - join them into a single text ("\\n\\n".join(m.content for m in results))
        - fill in COMPRESS_PROMPT with date, brief and results
        - call model_without_tools with a HumanMessage and keep .content
        - return {"notes": ...}

  4. Change the graph: when the agent finishes it no longer goes to END, it goes to "compress".
        - in add_conditional_edges, "end" points to "compress"
        - add the node and the edge compress → END

  5. At the end, print result["notes"]. (Use app.invoke; if you want to see the
     process, keep the turn print you already have in model_call.)

What to check in the output (also print result["messages"][-1].content):
    - Does each fact carry its [number] and is there a list of Sources at the end?
    - Do the notes have the 6 writers? Compare with the agent's final answer.
      If the notes are missing one that IS in the final answer, look for their
      name in the search results: does it appear? If not, where did the agent
      get that fact from? (Hint: "Do not invent anything that is not in the results".)

Think about it: print len(str(result["messages"])) and len(result["notes"]).
How many times smaller is what you would hand to the supervisor?

Run:  uv run python en/07_deep_research/agent_phase2_part2.py
"""
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, filter_messages
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
    notes: str


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

COMPRESS_PROMPT = '''You are an assistant that cleans up the results of a research.
Today is {date}.

Research brief:
{brief}

Search results:
{results}

Write some CLEAN NOTES:
- Keep ALL the facts relevant to the brief (names, dates, figures).
    Do not over-summarize.
- Remove what is repeated and what has nothing to do with the brief.
- After each fact, indicate its source with a number in square brackets, e.g. [1].
- At the end, a "Sources" section with the numbered list of URLs.
- Do not invent anything that is not in the results.'''

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

def compress(state: AgentState) -> dict:
    results = filter_messages(state["messages"], include_names=["search"])
    results_text = "\n\n".join(m.content for m in results)
    prompt = COMPRESS_PROMPT.format(
        date=get_today(),
        brief=state["messages"][0].content, # the first message is the brief
        results=results_text
    )
    answer = model_without_tools.invoke([HumanMessage(content=prompt)])
    return {"notes": answer.content}


def should_continue(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# 5. The graph: identical to the one in Exercise 8
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("compress", compress)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": "compress"})
graph.add_edge("tools", "our_agent")
graph.add_edge("compress", END)

app = graph.compile()


# 6. Try it: prints each message as it is generated
def print_stream(stream):
    for event in stream:
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    inputs = {"turns": 0, "messages": [HumanMessage(BRIEF)]}
    result = app.invoke(inputs)
    print("Turns:", result["turns"])
    print("\n\nNotes:", result["notes"])
    print("\n\nResult:", result["messages"][-1].content)

