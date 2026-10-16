import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# OpenRouter speaks the OpenAI API, so ChatOpenAI works once it points at OpenRouter's URL
llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("agent", process)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
agent = graph.compile()

user_message = input("User: ")

while user_message.strip().lower() not in ("quit", "exit"):
    agent.invoke({"messages": [HumanMessage(content=user_message)]})
    user_message = input("\nUser: ")
