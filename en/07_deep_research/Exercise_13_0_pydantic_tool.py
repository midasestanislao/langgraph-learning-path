"""
EXERCISE 13 — A Pydantic class as a tool                              (uses LLM, 3 calls)
=====================================================================================
CONCEPTS

The Phase 4 supervisor doesn't search anything itself: it DELEGATES. To delegate, the LLM
needs a "tool" that is really just a form to fill in.

A Pydantic class as a tool -> describes WHAT data you want, not what to do with it
    from pydantic import BaseModel, Field

    class ConductResearch(BaseModel):
        \"\"\"Delegates the research of ONE topic to a researcher.\"\"\"
        research_topic: str = Field(description="The topic to research, in detail.")

    class ResearchComplete(BaseModel):
        \"\"\"Signals that the research is finished.\"\"\"     # no fields: it's just a signal

    model_with_tools = llm.bind_tools([ConductResearch, ResearchComplete])

    Just like in M06: the LLM sees the class NAME, the DOCSTRING (when to use it)
    and the Fields (what to fill in). The class has no code: there is nothing to run.
    (The reference course puts @tool on top of the class; it works the same with or without it.)

What the LLM returns -> tool_calls, just like in M03
    response.tool_calls
    [{"name": "ConductResearch", "args": {"research_topic": "Price of coffee in Madrid"}, "id": "call_1"},
     {"name": "ConductResearch", "args": {"research_topic": "Price of coffee in Buenos Aires"}, "id": "call_2"}]

    The LLM decides HOW MANY times it uses it: 0, 1 or several in the same response.
    Each tool_call is a task for a different researcher.

Difference from with_structured_output (M07)
    with_structured_output(Class) -> ALWAYS returns exactly ONE object.
    bind_tools([Class])           -> the LLM CHOOSES whether to use it, which one, and how many times.

EXERCISE (solved example)

No graph: just the model with tools and three different questions.

    a) a comparison of two cities            -> the LLM delegates TWO research tasks
    b) a simple topic                        -> delegates ONE
    c) "I already have all the information"  -> uses ResearchComplete

Expected output (the texts may vary):
    --- Question a: 2 tool_calls
    ConductResearch {'research_topic': 'Price of coffee in Madrid, Spain'}
       Researcher 1 receives: Price of coffee in Madrid, Spain
    ConductResearch {'research_topic': 'Price of coffee in Buenos Aires, Argentina'}
       Researcher 2 receives: Price of coffee in Buenos Aires, Argentina
    --- Question b: 1 tool_calls
    ...
    --- Question c: 1 tool_calls
    ResearchComplete {}

Think about it:
    - Nobody runs ConductResearch: the class has no code. So, when your
      program sees a tool_call with name == "ConductResearch", what should it do?
      (That is the question the supervisor answers.)
    - What happens in question c if you remove ResearchComplete from the bind_tools?

Run:  uv run python en/07_deep_research/Exercise_13_0_pydantic_tool.py
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# 1. The model, as always
llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# 2. THE NEW PART: two classes that are "forms" for the LLM.
#    The docstring says WHEN to use it; the Field says WHAT to write in each field.
class ConductResearch(BaseModel):
    """Delegates the research of ONE topic to a researcher. Use it once for each distinct topic."""
    research_topic: str = Field(description="The topic to research, described in detail.")


class ResearchComplete(BaseModel):
    """Use it when nothing else needs to be researched."""


# 3. bind_tools accepts the classes directly (no need for @tool)
model_with_tools = llm.bind_tools([ConductResearch, ResearchComplete])


# 4. Three questions to see how many times it decides to use each tool
questions = {
    "a": "Compare the price of coffee in Madrid and in Buenos Aires. Delegate each city to a different researcher.",
    "b": "Research who invented espresso.",
    "c": "I already have all the information I need about espresso. There is no need to research anything else.",
}

for letter, question in questions.items():
    response = model_with_tools.invoke(question)
    print(f"--- Question {letter}: {len(response.tool_calls)} tool_calls")

    researcher = 0
    for call in response.tool_calls:
        print(call["name"], call["args"])

        # 5. Read the form: the topic is in args["research_topic"]
        if call["name"] == "ConductResearch":
            researcher += 1
            print(f"   Researcher {researcher} receives: {call['args']['research_topic']}")
