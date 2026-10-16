"""
EXERCISE 13.1 — A coffee shop: the LLM writes the order ticket        (uses LLM, 2 calls)
=====================================================================================
CONCEPTS

The idea in one sentence
    The LLM is the WAITER: it listens to the customer and writes ORDER TICKETS.
    Your code is the KITCHEN: it reads each ticket and prepares it.

The ticket is a Pydantic class (a "form" with no code)
    from pydantic import BaseModel, Field

    class OrderCoffee(BaseModel):
        \"\"\"Use it when the customer orders a coffee. Once per coffee.\"\"\"
        kind: str = Field(description="Kind of coffee, for example 'cappuccino'")

    - The docstring tells the LLM WHEN to use it.
    - The Field tells it WHAT to write in it.

Handing the tickets to the waiter
    model_with_tools = llm.bind_tools([OrderCoffee])

The tickets it wrote -> response.tool_calls (a LIST of dicts)
    response = model_with_tools.invoke("I'd like a cappuccino and a latte")

    response.tool_calls
    [{"name": "OrderCoffee", "args": {"kind": "cappuccino"}, "id": "..."},
     {"name": "OrderCoffee", "args": {"kind": "latte"},      "id": "..."}]

    To read the kind of coffee from a ticket:  call["args"]["kind"]

    If the customer doesn't order anything, the list is EMPTY and the waiter replies with
    normal text in response.content.

EXERCISE

  1. Create the llm (OpenRouter, gpt-4o, temperature=0), like in 13.0.

  2. Create the OrderCoffee class exactly as it is in CONCEPTS.

  3. Create model_with_tools with bind_tools([OrderCoffee]).

  4. Request 1: invoke model_with_tools with "I'd like a cappuccino and a latte, please".
        - print how many tickets there are:  len(response.tool_calls)
        - go through response.tool_calls with a for and, for each ticket, print:
              Preparing a cappuccino...

  5. Request 2: invoke with "Hi, what time do you open?".
        - print how many tickets there are (there should be 0)
        - since there are no tickets, print what it replied: response.content

Expected output:
    Tickets: 2
    Preparing a cappuccino...
    Preparing a latte...
    Tickets: 0
    Reply: Hi, the opening hours may vary depending on the location...

Think about it:
    - Who decided there were 2 tickets and not 1? You didn't write any if.
    - Try the request "I'd like three espressos". How many tickets does it write?
      Does it write 1 ticket with "three espressos" or 3 tickets with "espresso"?
      (Look at the docstring: "Once per coffee".)

Run:  uv run python en/07_deep_research/Exercise_13_1_coffee_order.py
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

class OrderCoffee(BaseModel):
    """Use it when the customer orders a coffee. Once per coffee"""
    kind: str = Field(description="Kind of coffee, for example 'cappuccino'")

model_with_tools = llm.bind_tools([OrderCoffee])

response = model_with_tools.invoke("I'd like a cappuccino and a latte")

print(f"Tickets: {len(response.tool_calls)} tool_calls")

for call in response.tool_calls:
    print(f"Preparing: {call["args"]["kind"]}")

response = model_with_tools.invoke("Hi, what time do you open?")

print(f"Tickets: {len(response.tool_calls)} tool_calls")
print(f"Reply {response.content}")
