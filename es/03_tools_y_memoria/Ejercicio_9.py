from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an additional function that adds 2 numbers together"""
    return a + b

@tool
def multiply(a:int, b: int):
    """This is a multiplication function that multiplies 2 numbers together"""
    return a * b 

tools = [add, multiply]


model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You're my AI assistant, please answer my query to the best of your ability. "
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")
checkpointer = InMemorySaver()

app = graph.compile(checkpointer=checkpointer)

def print_stream(stream):
    for event in stream:
        message = event["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else: 
            message.pretty_print()

# --- Turno 1: le decimos quienes somos y le pedimos una operacion ---
config = {"configurable": {"thread_id": "1"}}

print("\n===== Turno 1 (thread_id=1) =====")
inputs = {"messages": [("user", "Hola, me llamo Gustavo. Suma 40 + 75 y multiplica el resultado por 12.")]}
print_stream(app.stream(inputs, config, stream_mode="values"))

# --- Turno 2: mismo thread_id -> debe recordar el nombre y el resultado anterior ---
print("\n===== Turno 2 (thread_id=1, mismo chat) =====")
inputs = {"messages": [("user", "¿Cómo me llamo y cuál fue el ultimo resultado que calculaste?")]}
print_stream(app.stream(inputs, config, stream_mode="values"))

# --- Turno 3: thread_id distinto -> no debe recordar nada ---
config_otro = {"configurable": {"thread_id": "2"}}

print("\n===== Turno 3 (thread_id=2, chat nuevo) =====")
inputs = {"messages": [("user", "¿Cómo me llamo?")]}
print_stream(app.stream(inputs, config_otro, stream_mode="values"))