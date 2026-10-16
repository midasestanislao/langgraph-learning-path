"""
EXERCISE M07 — Structured output: the LLM returns an object, not a message   (uses LLM, ~14 calls)
=====================================================================================================
model.invoke(...)                                    -> AIMessage (free text in .content)
model.with_structured_output(MyClass).invoke(...)    -> a MyClass object with typed fields

It's the foundation of step 1 (scoping): instead of reading "I think I need more info..."
and guessing, you receive `need_clarification=True` and use it in an `if`.

The LLM fills in the fields guided by the class docstring and by each Field's
`description`: they are instructions, write them well.

Run:  uv run python en/04_messages/M07_structured_output.py
"""
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ---------- Part A: extracting data ----------

# TODO 1: complete the class with these fields (all with Field(description=...)):
#   title: str, year: int, director: str, genres: list[str]
# Each field is:   name: type = Field(description="instruction for the LLM")
# The TYPE matters: if you put `int`, Pydantic guarantees you get a number, not "2014".
class Movie(BaseModel):
    """Basic data about a movie."""
    title: str = Field(description="Original title of the movie")
    year: int = Field(description="Release year")
    director: str = Field(description="Name of the director")
    genres: list[str] = Field(description="List of the movie's genres")


# TODO 2: create the structured model and invoke it with [HumanMessage("Tell me about Interstellar")]
#   Print type(movie) and movie. Then access movie.year + 1 (it's a real int!)
# with_structured_output(Class) creates a "wrapper model": it's invoked just like the normal one,
# but returns a Movie object instead of an AIMessage.
movie_model = model.with_structured_output(Movie)
movie = movie_model.invoke([HumanMessage("Tell me about Interstellar")])

print(f"TODO 2: {type(movie)}")
print(f"  {movie}")
print(f"  title: {movie.title} | director: {movie.director}")
print(f"  year + 1 = {movie.year + 1}   <- you can add to it because it's an int, not text")


# ---------- Part B: making decisions (preview of step 1) ----------

# TODO 3: create the Decision class with:
#   needs_clarification: bool  -> whether the request is too vague to research
#   question: str              -> the question for the user ("" if not needed)
class Decision(BaseModel):
    """Decides whether a research request needs clarification."""
    needs_clarification: bool = Field(description="True if the request is too vague to research")
    question: str = Field(description="Question for the user, or \"\" if no clarification is needed")


# TODO 4: invoke the structured model with Decision for these two requests
#   and save the results. Print both.
vague = "Research coffee shops"
clear = "Compare price and quality of the 5 best-rated specialty coffee shops in Madrid in 2026"

# We create the structured model ONCE and reuse it for both requests.
decision_model = model.with_structured_output(Decision)
decision_vague = decision_model.invoke([HumanMessage(vague)])
decision_clear = decision_model.invoke([HumanMessage(clear)])

print(f"\nTODO 4:")
print(f"  vague -> {decision_vague}")
print(f"  clear -> {decision_clear}")


# TODO 5: this is how the clarify_with_user node does it: if decision_vague.needs_clarification,
#   turn the question into an AIMessage to add it to the conversation.
# The Decision object is NOT a message: it can't go into the history.
# That's why we take the text (.question) and wrap it in an AIMessage,
# because from the user's point of view, it's the AI that is asking them.
if decision_vague.needs_clarification:
    message_for_user = AIMessage(content=decision_vague.question)
    print(f"\nTODO 5:")
    message_for_user.pretty_print()


# TODO 6 (experiment — the most important one in this exercise):
#   Look at decision_clear. With a simple description, GPT-4o often asks for clarification
#   EVEN for the clear request (e.g. "which coffee shops do you want to compare?"), because
#   it doesn't know that it will be able to search the internet itself.
#
#   a) Invoke the clear request 5 times in a loop and count how many times it comes out True.
#   b) Improve the Decision docstring and the description of `needs_clarification`
#      (e.g. explain that it's an agent that SEARCHES THE INTERNET and that data that
#      can be looked up is not asked of the user). Repeat a) and compare.
#
#   Lesson: even with temperature=0, the decision is not 100% stable. That's why in
#   step 1 the prompt says "if you ALREADY asked once, almost never ask again":
#   the flow design compensates for the model being imperfect.

# b) Same structure, but with much more precise instructions in the docstring and Field.
class ImprovedDecision(BaseModel):
    """You are the first step of a research agent that SEARCHES THE INTERNET on its own.
    Decide whether the user's request needs clarification before starting to research."""
    needs_clarification: bool = Field(description=(
        "True ONLY if the request is so vague that it's unclear what to research "
        "(the topic, the place or the goal is missing). False if we can already start: "
        "data that can be found by searching the internet (names, rankings, "
        "prices) is NOT asked of the user."
    ))
    question: str = Field(description="Concise question for the user, or \"\" if no clarification is needed")


# Helper so we don't repeat the loop for each class.
def count_clarifications(schema_class, request: str, times: int = 5) -> int:
    structured_model = model.with_structured_output(schema_class)
    total = 0
    for _ in range(times):              # "_" = variable we don't use
        if structured_model.invoke([HumanMessage(request)]).needs_clarification:
            total += 1
    return total


print("\nTODO 6 - CLEAR request, how many times out of 5 does it ask for clarification?")
print(f"  a) Decision (simple description):   {count_clarifications(Decision, clear)}/5")
print(f"  b) ImprovedDecision:                {count_clarifications(ImprovedDecision, clear)}/5")
print("  (Ideally it would be 0/5. If it doesn't reach 0, that's the lesson: the model is not 100% reliable.)")


# ----------------------------- Checks (do not touch) -----------------------------
assert isinstance(movie, Movie), "TODO 2: it must be a Movie object, not an AIMessage"
assert isinstance(movie.year, int) and movie.year == 2014, "TODO 1/2: Interstellar is from 2014"
assert isinstance(movie.genres, list), "TODO 1: genres must be list[str]"
assert isinstance(decision_vague, Decision) and isinstance(decision_clear, Decision), "TODO 4"
assert decision_vague.needs_clarification is True, "TODO 4: the vague request should need clarification"
# (decision_clear is deliberately not checked: see TODO 6)
assert isinstance(message_for_user, AIMessage) and message_for_user.content == decision_vague.question, "TODO 5"
print("\n✅ M07 completed")
