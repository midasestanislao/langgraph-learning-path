import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You answer only in rhymes."



class AgentState(TypedDict):
    messages: List[BaseMessage]


llm = ChatOpenAI(
    model=os.getenv("BONSAI_MODEL", "prism-ml/bonsai-27b"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    temperature=0.7,
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    state["messages"].append(response)
    return state


graph = StateGraph(AgentState)
graph.add_node("agent", process)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
agent = graph.compile()


def chat():
    history: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    print(f"Bonsai 27B agent ready. System prompt: {SYSTEM_PROMPT!r}")
    print("Type 'quit' or 'exit' to stop.\n")

    user_message = input("User: ")
    while user_message.strip().lower() not in ("quit", "exit"):
        history.append(HumanMessage(content=user_message))
        result = agent.invoke({"messages": history})
        history = result["messages"]
        user_message = input("\nUser: ")


if __name__ == "__main__":
    chat()
