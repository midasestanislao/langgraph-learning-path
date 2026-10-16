"""
EXERCISE 13.2 — A travel planner that delegates                       (uses LLM, 3 calls)
=====================================================================================
CONCEPTS

Everything is from Exercise 13. The only new thing is this:

Forms with SEVERAL fields of different types
    class BookHotel(BaseModel):
        \"\"\"Books a hotel in a city.\"\"\"
        city: str = Field(description="City of the hotel")
        nights: int = Field(description="Number of nights")

    The type matters: if the field is int, call["args"]["nights"] gives you a 3,
    not a "3". The LLM fills in each field guided by its description.

Dispatching each tool_call by its name -> one if/elif per tool
    for call in response.tool_calls:
        if call["name"] == "BookHotel":
            ...
        elif call["name"] == ...:
            ...

    This is exactly what the Phase 4 supervisor_tools node will do:
    look at the name of each tool_call and decide what to do with it.

EXERCISE

A travel assistant that does NOT book anything itself: it only fills in forms.

  1. Create the llm with temperature=0 (like in 13).
  2. Define three classes:
        BookFlight   -> origin (str), destination (str), date (str)
                        In the description of date ask for the DD/MM format and to write
                        'no date' if the user doesn't say it.
        BookHotel    -> city (str), nights (int)
        PlanComplete -> no fields: the user doesn't need anything else
     Write a docstring in each one saying WHEN to use it.
  3. Create model_with_tools with the three of them.
  4. Invoke the model with these three requests:
        a) "I want to go from Madrid to Rome on November 5th and stay 3 nights in a
            hotel. Then I come back to Madrid on November 8th."
        b) "I need a flight from Lima to Cusco."
        c) "Thanks, that's all, I have everything booked now."
  5. For each response print how many tool_calls there are and dispatch each one with if/elif:
        BookFlight   -> "Flight: Madrid -> Rome (05/11)"
        BookHotel    -> "Hotel: 3 nights in Rome"
        PlanComplete -> "Plan complete, nothing else to book"
  6. If a response has 0 tool_calls, print response.content
     (there the LLM talks to you with text instead of filling in a form).

Expected output (may vary):
    --- Request a: 3 tool_calls
    Flight: Madrid -> Rome (05/11)
    Hotel: 3 nights in Rome
    Flight: Rome -> Madrid (08/11)
    --- Request b: 0 tool_calls
    Could you tell me the date on which you wish to travel from Lima to Cusco? ...
    --- Request c: 1 tool_calls
    Plan complete, nothing else to book

Think about it:
    - In request b the Field said "write 'no date'", but the LLM preferred not to
      use the tool and ask you instead. What would you change in the BookFlight
      docstring so that it books anyway, even if the date is missing?
    - In request a, how many BookFlight are there? Who decided there were two?

Run:  uv run python en/07_deep_research/Exercise_13_2_travel_planner.py
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
    temperature=0
)

class BookFlight(BaseModel):
    """Use it when the user wants a flight"""
    origin: str = Field(description="The departure city of the flight")
    date: str = Field(description="Date of the flight in DD/MM format, or 'no date' if the user doesn't say it.")
    destination: str = Field(description="The destination city of the flight")

class BookHotel(BaseModel):
    """Use it when the user wants to book a hotel in a city."""
    city: str = Field(description="City of the hotel")
    nights: int = Field(description="Number of nights")

class PlanComplete(BaseModel):
    """Use it when nothing else needs to be booked."""

model_with_tools = llm.bind_tools([BookFlight, BookHotel, PlanComplete])

# 4. Three requests to see how many times it decides to use each tool
requests = {
    "a": "I want to go from Madrid to Rome on November 5th and stay 3 nights in a hotel. Then I come back to Madrid on November 8th.",
    "b": "I need a flight from Lima to Cusco.",
    "c": "Thanks, that's all, I have everything booked now.",
}

for letter, request in requests.items():
    response = model_with_tools.invoke(request)
    print(f"--- Request {letter}: {len(response.tool_calls)} tool_calls")
    if len(response.tool_calls) == 0:
        print(response.content)
    else:
        for call in response.tool_calls:
            args = call["args"]
            if call["name"] == "BookFlight":
                print(f"Flight: {args['origin']} -> {args['destination']} ({args['date']})")

            elif call["name"] == "BookHotel":
                print(f"Hotel: {args['nights']} nights in {args['city']}")

            elif call["name"] == "PlanComplete":
                print("Plan complete, nothing else to book")
