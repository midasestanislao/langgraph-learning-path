"""
ASYNC 2 — Idea 3: a task that has to wait                                 (no LLM)
=====================================================================================
IDEA 3: await asyncio.sleep(1) = "wait 1 second here"

    A real researcher WAITS: for the internet, for the LLM. To simulate that wait:

        await asyncio.sleep(1)

    asyncio.sleep(1) is also a pending task ("wait 1 second"), which is why
    it takes await just like await greet().

Two awaits in a row = one after the other

    await research("Chile")    # main() waits for Chile to finish COMPLETELY...
    await research("Japan")    # ...and only then does Japan start

    That's why it takes 2 seconds: 1 + 1. Having async has NOT made anything faster yet.
    This is exactly what your Exercise_14_1 does (delegating with a for).
    The fast part comes in async_3, with a single change in main().

Expected output:
    starts: Chile
    finishes: Chile
    starts: Japan
    finishes: Japan
    Time: 2.0 s

Run:  uv run python en/07_deep_research/async_2_wait_with_sleep.py
"""
import asyncio
import time


# 1. A toy researcher: prints, WAITS 1 second, prints.
async def research(topic):
    print(f"starts: {topic}")
    await asyncio.sleep(1)  # ← waits 1 second (simulates a search)
    print(f"finishes: {topic}")


# 2. Two awaits in a row: the second one doesn't start until the first one finishes.
async def main():
    start =  time.perf_counter()         # the stopwatch starts
    await research("Chile")              # 1 second
    await research("Japan")              # another second
    print(f"Time: {time.perf_counter() - start:.1f} s")   # 2.0 s


asyncio.run(main())
