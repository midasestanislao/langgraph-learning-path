"""
ASYNC 1 — Ideas 1 and 2: calling doesn't run it; await does               (no LLM)
=====================================================================================
IDEA 1: an async function does NOT run when you call it

    async def greet():
        print("Hello!")

    greet()          # ← prints NOTHING. It only creates a "pending task" (a post-it note).
                     #   Python warns: "coroutine 'greet' was never awaited"

    With a normal function (no async), greet() WOULD print "Hello!".
    With async, calling it only writes the post-it: "pending: greet".

IDEA 2: await is what runs the task

    await greet()    # ← picks up the post-it, does the work and waits for it to finish

    Rule: await can ONLY be written INSIDE an async def function.
    So, who starts the first async function? asyncio.run(...):
    it's the "power button". You press it ONCE, at the end of the file.

Expected output:
    Hello!

To test idea 1: comment out the main() and asyncio.run() lines, and leave only
a bare greet(). You'll see it prints nothing and the warning shows up.

Run:  uv run python en/07_deep_research/async_1_call_vs_await.py
"""
import asyncio


# 1. An async function. Calling it (greet()) does NOT run it: it creates a pending task.
async def greet():
    print("Hello!")


# 2. To use await you need to be inside another async function.
async def main():
    await greet()  # <- picks up the post-it and does the work


# 3. The power button: starts main() from normal code. Only once.
asyncio.run(main())
