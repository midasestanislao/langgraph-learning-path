import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

SHOW_THINKING = os.getenv("SHOW_THINKING", "false").lower() == "true"
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() == "true"

extra_body = {}
if DISABLE_THINKING:
    extra_body["enable_thinking"] = False


class AgentState(TypedDict):
    messages: List[BaseMessage]


llm = ChatOpenAI(
    model=os.getenv("LMSTUDIO_MODEL", "qwen3.5-9b"),
    api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    temperature=0.7,
    max_tokens=2048,
    extra_body=extra_body or None,
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    reasoning = response.additional_kwargs.get("reasoning_content")

    if SHOW_THINKING and reasoning:
        print(f"\n[thinking]\n{reasoning}\n[/thinking]")

    print(f"\nAI: {response.content}")
    state["messages"].append(response)
    return state


graph = StateGraph(AgentState)
graph.add_node("agent", process)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
agent = graph.compile()


if __name__ == "__main__":
    history: List[BaseMessage] = []
    print(f"Model: {os.getenv('LMSTUDIO_MODEL', 'qwen3.5-9b')}")
    print(f"Thinking: {'disabled' if DISABLE_THINKING else ('visible' if SHOW_THINKING else 'hidden')}")
    print("Type 'quit' or 'exit' to stop.\n")

    user_message = input("User: ")
    while user_message.strip().lower() not in ("quit", "exit"):
        history.append(HumanMessage(content=user_message))
        result = agent.invoke({"messages": history})
        history = result["messages"]
        user_message = input("\nUser: ")
