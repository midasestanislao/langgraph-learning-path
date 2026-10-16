"""
EXERCISE 15 — async / await / gather: doing several things at once    (no LLM, free)
=====================================================================================
CONCEPTS

The problem (you saw it in 14.1)
    3 researchers that take 1 s each, ONE AFTER ANOTHER  -> 3 s.
    Almost all of that time is WAITING (for the internet, for the LLM). While one
    waits, the others could be working.

async def -> a function that can PAUSE while it waits
    async def research(topic):
        ...
    Calling it does NOT run it: research("Chile") only creates the "pending task".
    It runs when someone does await on it.

await -> "wait here, but let the others move forward in the meantime"
    notes = await research("Chile")

asyncio.sleep(1) vs time.sleep(1)
    time.sleep(1)           -> freezes the WHOLE program for 1 second
    await asyncio.sleep(1)  -> this task waits 1 s, but the others keep going

asyncio.gather(*tasks) -> launches ALL of them at once and waits for ALL to finish
    results = await asyncio.gather(*[research(t) for t in topics])
    Returns a list with the results IN THE SAME ORDER as the tasks.
    The * unpacks the list (you saw it in M03 with *tool_messages).

asyncio.run(main()) -> starts the async "engine" from normal code
    It is called only ONCE, at the end of the file. Inside main() you can already use await.

The connection with LangGraph (you will use it in 15.1)
    app.invoke(...)          ->  await app.ainvoke(...)
    Every compiled graph has both versions. The second one can go into gather.

EXERCISE (solved example)

The same simulation as 14.1 (3 topics, 1 s each), but WITHOUT LangGraph, done
both ways: one after another, and all at once. Compare the times and pay attention
to the ORDER of the prints.

Expected output:
    === One after another ===
      starts: Finland
      finishes: Finland
      starts: Japan
      finishes: Japan
      starts: Chile
      finishes: Chile
    Time: 3.0 s

    === All at once (gather) ===
      starts: Finland
      starts: Japan
      starts: Chile
      finishes: Finland
      finishes: Japan
      finishes: Chile
    Time: 1.0 s
    ['Notes on Finland', 'Notes on Japan', 'Notes on Chile']

Think about it:
    - In the gather version, the three "starts" come out BEFORE any "finishes".
      What was the program doing during that second?
    - Change asyncio.sleep(1) to time.sleep(1) in research(). How long does the
      gather version take now? Why?

Run:  uv run python en/07_deep_research/Exercise_15_0_async_gather.py
"""
import asyncio
import time


# 1. WHAT'S NEW: an async function. It simulates a researcher that takes 1 second.
async def research(topic: str) -> str:
    print(f"  starts: {topic}")
    await asyncio.sleep(1)          # waits 1 s WITHOUT freezing the others
    print(f"  finishes: {topic}")
    return f"Notes on {topic}"


# 2. Version A: one after another (like your 14.1)
async def one_after_another(topics: list[str]) -> list[str]:
    notes = []
    for topic in topics:
        notes.append(await research(topic))      # waits for it to finish before going on
    return notes


# 3. Version B: all at once with gather
async def all_at_once(topics: list[str]) -> list[str]:
    tasks = [research(topic) for topic in topics]    # creates the 3 tasks (they don't run yet)
    return await asyncio.gather(*tasks)              # launches them together and waits for all 3


# 4. main() is async too, so we can use await inside
async def main():
    topics = ["Finland", "Japan", "Chile"]

    print("=== One after another ===")
    start = time.perf_counter()
    await one_after_another(topics)
    print(f"Time: {time.perf_counter() - start:.1f} s\n")

    print("=== All at once (gather) ===")
    start = time.perf_counter()
    notes = await all_at_once(topics)
    print(f"Time: {time.perf_counter() - start:.1f} s")
    print(notes)


# 5. Start the async engine (only once, from normal code)
asyncio.run(main())
