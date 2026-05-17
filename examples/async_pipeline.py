"""Async producer/consumer using asyncio.Queue, gather, timeout."""

import asyncio


async def producer(q, name, n):
    for i in range(n):
        await asyncio.sleep(0)
        await q.put((name, i))
    await q.put(None)


async def consumer(q, expected_sentinels):
    out = []
    seen = 0
    while seen < expected_sentinels:
        item = await q.get()
        if item is None:
            seen += 1
            continue
        out.append(item)
    return out


async def with_timeout():
    async def slow():
        await asyncio.sleep(10)
        return "never"

    try:
        return await asyncio.wait_for(slow(), timeout=0.05)
    except asyncio.TimeoutError:
        return "timed-out"


async def main():
    q = asyncio.Queue()
    producers = [
        asyncio.create_task(producer(q, "A", 3)),
        asyncio.create_task(producer(q, "B", 2)),
    ]
    consumer_task = asyncio.create_task(consumer(q, len(producers)))

    await asyncio.gather(*producers)
    items = await consumer_task

    # gather demo
    gathered = await asyncio.gather(
        asyncio.sleep(0, result="x"),
        asyncio.sleep(0, result="y"),
    )

    timeout_result = await with_timeout()

    return items, gathered, timeout_result


items, gathered, timeout_result = asyncio.run(main())
print("items:", sorted(items))
print("gathered:", gathered)
print("timeout:", timeout_result)
