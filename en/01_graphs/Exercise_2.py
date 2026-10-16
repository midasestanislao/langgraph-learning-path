from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
import random

class AgentState(TypedDict):
    limit: int
    max_attempts: int
    secret: int
    guess: int
    pastChoices: List[int]
    attempts: int

def greeting_node(state: AgentState) -> AgentState:
    """This node adds a greeting message to the state and sets the secret number to guess and the limit of attempts"""
    state["secret"] = random.randint(0,state['limit'])
    print(f"Hello user, this program will guess a secret number generated randomly from 0 to {state['limit']} in at most {state['max_attempts']} attempts")
    state['pastChoices'] = []
    state['attempts'] = 0
    return state

def guess_node(state: AgentState) -> AgentState:
    "This node generates a random number that is going to be compared with the secret number"
    state['guess'] = random.randint(0, state['limit'])

    while state['guess'] in state['pastChoices']:
        state['guess'] = random.randint(0, state['limit'])

    # 3. Save the new attempt in the history
    state['pastChoices'].append(state['guess'])

    state['attempts'] += 1
    print(f"Guess number: {state["guess"]}")
    return state

def should_continue(state: AgentState) -> str:
    "This node decides if the counter should continue or not"
    if state['attempts'] < state['max_attempts'] and state['guess'] != state['secret']:
        return "loop"
    else:
        if state['guess'] == state['secret']:
            print(f"Success! We guessed the secret number {state['secret']} in {state['attempts']} attempts.")
        else:
            print(f"We reached the maximum attempts! Secret was: {state['secret']}")
        return "exit"


graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("guesser", guess_node)
graph.add_edge("greeting", "guesser")

graph.add_conditional_edges(
    "guesser",
    should_continue,
    {
        "loop": "guesser",
        "exit": END
    }
)

graph.set_entry_point("greeting")

app = graph.compile()

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"limit": 20, "max_attempts": 8})
