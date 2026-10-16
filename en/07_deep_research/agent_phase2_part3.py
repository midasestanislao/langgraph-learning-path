"""
PHASE 2 — STEP 3: joining Phase 1 and Phase 2                          (uses LLM)
=====================================================================================
WHERE YOU ARE

    PHASE 1 (agent_phase1_part3.py)            PHASE 2 (agent_phase2_part2.py)
    conversation ──► brief  ───────────────►   brief ──► research ──► notes
                       └──────── STEP 3 (NOW): connect the two ─────────┘

    When done you will have a program that: asks you what is needed, writes the
    brief, researches on the internet and hands you notes with sources.

CONCEPTS

Importing from your own files -> reuse without copy and paste
    from agent_phase1_part3 import app as scope_app
    from agent_phase2_part2 import app as researcher_app

    - "agent_phase1_part3" is the file name WITHOUT the .py
    - "import app" brings the app variable from that file
    - "as scope_app" gives it another name here, because BOTH files call their
      graph "app" and otherwise one would hide the other

    It works because the 3 files are in the same folder (07_deep_research).

Careful! Importing a file RUNS all of its code
    When you do "from agent_phase2_part2 import app", Python runs that whole
    file, top to bottom. If it has a loose app.invoke(...) at the end,
    it will launch a complete research just by importing it!

if __name__ == "__main__": -> code that ONLY runs if you execute THAT file
    if __name__ == "__main__":
        result = app.invoke(inputs)     # runs with: uv run python en/07_deep_research/agent_phase2_part2.py
                                        # does NOT run if another file imports it

    Your agent_phase1_part3.py already uses it (main() is inside that if).
    Your agent_phase2_part2.py does not yet.

The output of one graph is the input of the other
    phase1_result["research_brief"]   →   HumanMessage(content=brief) for Phase 2

EXERCISE — STEP 3

  1. In agent_phase2_part2.py: put the test at the end (inputs, app.invoke and the
     prints) inside an  if __name__ == "__main__":
     Check that the file still works on its own:
        uv run python en/07_deep_research/agent_phase2_part2.py

  2. In THIS file, import the two graphs with different names
     (scope_app and researcher_app) and HumanMessage.

  3. Write main() in two parts:

     PHASE 1 — the loop from agent_phase1_part3.py, using scope_app
        (with its config and thread_id). When there is a research_brief, store it in
        a brief variable and leave the loop with break.

     PHASE 2 — outside the loop:
        - print "Researching..." (it takes a while)
        - invoke researcher_app with {"turns": 0, "messages": [HumanMessage(content=brief)]}
        - print the notes: result["notes"]

  4. At the end of the file:
        if __name__ == "__main__":
            main()

Test with this conversation:
    User: Research writers
    AI:   Which aspect...?                              ← Phase 1 asks
    User: Latin American ones who won the Nobel Prize
    AI:   I'm going to research...                      ← Phase 1 writes the brief
    BRIEF: ...
    Researching...
    --- turn 1 -> asks for: ['search']                  ← Phase 2 researches
    ...
    NOTES: ... with [1], [2]... and the list of Sources  ← final result

Think about it: if you type "exit" in Phase 1, the loop ends without a brief.
What would happen then in Phase 2? How would you avoid it?

Run:  uv run python en/07_deep_research/agent_phase2_part3.py
"""
from agent_phase1_part3 import app as scope_app
from agent_phase2_part2 import app as researcher_app

from langchain_core.messages import HumanMessage

def main():
    config = {"configurable": {"thread_id": "1"}}
    brief = None
    while True:
        text = input("User: ").strip()
        if not text:
            continue
        elif text.lower() == "exit":
            break

        result = scope_app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"AI: {result['messages'][-1].content}")

        if "research_brief" in result:
            brief = result["research_brief"]
            print(f"\nBRIEF:\n{result['research_brief']}")
            break

    # If the user left with "exit", there is no brief: there is nothing to research
    if brief is None:
        print("No brief, nothing to research. See you!")
        return

    print("\nResearching...")
    result = researcher_app.invoke({"turns": 0, "messages": [HumanMessage(content=brief)]})
    print("\nNOTES:\n", result["notes"])

if __name__ == "__main__":
    main()
