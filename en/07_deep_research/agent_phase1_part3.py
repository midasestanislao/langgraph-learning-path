"""
PHASE 1 — STEP 3: a multi-turn conversation                            (uses LLM)
=====================================================================================
WHERE YOU ARE

    Step 1 ✅  clarify decides with the LLM and Command           (agent_phase1_part1.py)
    Step 2 ✅  write_brief writes the brief with the LLM          (agent_phase1_part2.py)
    Step 3 ⬅️  the user can ANSWER the question and continue

    Until now, if the agent asked something, the program ended and you could not
    answer. When this step is done it will work like this:

        User:  Research writers
        AI:    Which specific aspect would you like me to research?
        User:  Latin American ones who won the Nobel Prize
        AI:    I'm going to research...
        BRIEF: I want to research the Latin American writers...

CONCEPTS (none new)

Checkpointer (Exercise 9) -> saves the state between one invocation and the next
    from langgraph.checkpoint.memory import InMemorySaver

    app = graph.compile(checkpointer=InMemorySaver())

thread_id (Exercise 9) -> the "conversation number". Same thread_id = same
conversation; the checkpointer remembers all its messages.
    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"messages": [HumanMessage("...")]}, config)

You only send the NEW message
    In your M08 you re-sent the whole history by hand. Not here: the checkpointer
    already has the previous messages and add_messages appends the new one at the end.

Loop with input() (M08) -> ask the user for text until some condition is met
    while True:
        text = input("You: ")
        ...
        if <condition>:
            break

EXERCISE — STEP 3

  1. Compile the graph WITH a checkpointer (InMemorySaver).

  2. Create the config with a fixed thread_id, for example "1".

  3. Replace the tests at the end with a while True loop:
        - ask the user for text with input()
        - invoke the graph with ONLY the new message and the config
        - print the AI's last message
        - if the result already has "research_brief" -> print it and leave the loop
          (hint: "research_brief" in result)

  4. The graph and the nodes do NOT change.

Test with this conversation:
    You:  Research writers
          → it must ASK (there is no brief yet, the loop continues)
    You:  Latin American ones who won the Nobel Prize
          → it must say "I'm going to research..." and print the brief, and the program ends

Check that the brief includes what was in your FIRST message (writers) and in your
SECOND message (Latin American, Nobel). If so, the checkpointer works.

Think about it: what would happen if you used a different thread_id on each loop turn?
(Hint: your Exercise 9, turn 3.)

Run:  uv run python en/07_deep_research/agent_phase1_part3.py
"""

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

def get_today() -> str:
    now = datetime.now()
    return now.strftime("%d/%m/%Y")

class ClarifyWithUser(BaseModel):
    """Decides whether the user must be asked a question before researching."""
    need_clarification: bool = Field(description="True ONLY if the topic, the place or the goal is missing. If the request already says what to research, False. When in doubt, False")
    question: str = Field(description="The question for the user, brief and in their language ('' if no question is needed)")
    verification: str = Field(description="Message saying you are starting to research, summarizing what you understood ('' if a question is needed)")

class ResearchQuestion(BaseModel):
    """Detailed research question that will guide the researcher."""
    research_brief: str = Field(description="One detailed research question, in first person and in the user's language")

class ScopeState(MessagesState):
    research_brief: str

CLARIFY_PROMPT = '''These are the messages of the conversation with the user:
<Messages>
{messages}
</Messages>

Today is {date}.

Decide whether you need to ask them a question before starting to research.
- Ask ONLY if the topic, the place or the goal of the research is missing.
- If the request already says what to research, do NOT ask: the researcher
  will decide the missing details.
- Do not ask for facts that can be looked up on the internet.
- If you ALREADY asked them a question before, do not ask again.
- When in doubt, do NOT ask.
Reply in the language the user used in their messages (English → English, Spanish → Spanish).'''

BRIEF_PROMPT = '''These are the messages of the conversation with the user:
<Messages>
{messages}
</Messages>

Today is {date}.

Turn this conversation into ONE detailed research question.
- Include ALL the details and preferences the user gave.
- Do not invent preferences the user did not state. If a detail is missing
  (dates, budget, sources...), say it is left open.
- Do not add goals the user did not ask for (for example "the impact" or
  "the influence" if they did not mention them).
- Write it in first person, as if the user were asking it.
- Write it in the language the user used in their messages (English → English, Spanish → Spanish).'''

def clarify(state: ScopeState) -> Command[Literal["write_brief", "__end__"]]:
    text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = CLARIFY_PROMPT.format(messages=text, date=get_today())
    decision_model = model.with_structured_output(ClarifyWithUser)
    decision = decision_model.invoke([HumanMessage(content=prompt)])
    if decision.need_clarification:
        return Command(goto=END, update={"messages":[AIMessage(content=decision.question)]})
    else:
        return Command(goto="write_brief", update={"messages":[AIMessage(content=decision.verification)]})

def write_brief(state: ScopeState) -> dict:
    text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"])
    prompt = BRIEF_PROMPT.format(messages=text, date=get_today())
    brief_model = model.with_structured_output(ResearchQuestion)
    answer = brief_model.invoke([HumanMessage(content=prompt)])
    return {"research_brief": answer.research_brief}

graph = StateGraph(ScopeState)
graph.add_node("clarifyIntent", clarify)
graph.add_node("write_brief", write_brief)

graph.add_edge(START, "clarifyIntent")
graph.add_edge("write_brief", END)

app = graph.compile(checkpointer=InMemorySaver())

def main():
    config = {"configurable": {"thread_id": "1"}}

    while True:
        text = input("User: ").strip()
        if not text:
            continue
        elif text.lower() == "exit":
            break

        result = app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"AI: {result['messages'][-1].content}")

        if "research_brief" in result:
            print(f"\nBRIEF:\n{result['research_brief']}")
            break






if __name__ == "__main__":
    main()

# result = app.invoke({"messages": [HumanMessage('Research writers')]})
# print(result["messages"][-1].content)
# print(result.get("research_brief"))
# result = app.invoke({"messages": [HumanMessage('Research Latin American writers who have won a Nobel Prize')]})
# print(result["messages"][-1].content)
# print(result.get("research_brief"))
