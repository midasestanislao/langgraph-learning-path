"""
ASYNC 3 — Idea 4: launching several tasks at once with gather             (no LLM)
=====================================================================================
IDEA 4: asyncio.gather(task1, task2) starts ALL of them and waits for ALL of them to finish

    results = await asyncio.gather(research("Chile"), research("Japan"))

    - research("Chile") and research("Japan") only create two pending tasks.
    - gather starts them together; the await waits for BOTH to finish.
    - It returns a LIST with the results, in the same order:
          ['Notes on Chile', 'Notes on Japan']

Why it takes 1 second and not 2 (timeline)

    t = 0 s   gather starts Chile → "starts: Chile"
                                  → reaches await asyncio.sleep(1) → Chile PAUSES
              since Chile is paused, Python starts Japan → "starts: Japan"
                                  → reaches its sleep → Japan pauses
              (both wait out their second AT THE SAME TIME)
    t = 1 s   Chile's second is over → "finishes: Chile"
              Japan's too            → "finishes: Japan"

    The key: when a task reaches an await, it says "I'm waiting, let someone
    else go". In async_2 there was no "someone else" to let through; with gather, there is.

asyncio.sleep vs time.sleep
    await asyncio.sleep(1) → this task waits, but lets the others go
    time.sleep(1)          → FREEZES the whole program: nobody else advances
    Try it: change the sleep line to time.sleep(1) and gather will take 2.0 s.
    That's why inside an async function you always use asyncio.sleep.

Expected output:
    starts: Chile
    starts: Japan           ← Japan starts BEFORE Chile finishes
    finishes: Chile
    finishes: Japan
    Time: 1.0 s
    ['Notes on Chile', 'Notes on Japan']

What's next (idea 5): passing gather a LIST of tasks with *, instead of
writing them one by one. That's what Exercise_15_0 does.

Run:  uv run python en/07_deep_research/async_3_gather_all_at_once.py
"""
import asyncio
import time


# 1. Same as in async_2, but now it RETURNS a result.
async def research(topic):
    print(f"starts: {topic}")
    await asyncio.sleep(1)               # when it gets here, the task pauses and lets another one go
    print(f"finishes: {topic}")
    return f"Notes on {topic}"


# 2. THE NEW PART: gather starts both tasks at once and waits for both to finish.
async def main():
    start = time.perf_counter()
    results = await asyncio.gather(research("Chile"), research("Japan"))
    print(f"Time: {time.perf_counter() - start:.1f} s")   # 1.0 s, not 2.0
    print(results)                                        # list, in the same order


asyncio.run(main())
