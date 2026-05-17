import asyncio


async def simple_coro():
    return 42


print(asyncio.run(simple_coro()))


async def double(x):
    return x * 2


async def call_other():
    return await double(21)


print(asyncio.run(call_other()))


async def gather_demo():
    a = await asyncio.gather(
        asyncio.sleep(0, result=1),
        asyncio.sleep(0, result=2),
    )
    return a


print(asyncio.run(gather_demo()))


async def boom():
    raise ValueError('boom')


async def catch_across_await():
    try:
        await boom()
    except ValueError as e:
        return ('caught', str(e))


print(asyncio.run(catch_across_await()))


async def plain_no_await():
    return [x * 2 for x in range(3)]


print(asyncio.run(plain_no_await()))


# async for over an async generator (without intermediate awaits)
async def producer():
    for i in range(3):
        yield i


async def consume_async_gen():
    out = []
    async for x in producer():
        out.append(x)
    return out


print(asyncio.run(consume_async_gen()))


# async with
class AC:
    def __init__(self, name):
        self.name = name

    async def __aenter__(self):
        return self.name

    async def __aexit__(self, *a):
        return False


async def use_async_with():
    async with AC('hello') as v:
        return v


print(asyncio.run(use_async_with()))


# Multi-step await
async def step(x):
    return x + 1


async def chain():
    a = await step(1)
    b = await step(a)
    c = await step(b)
    return c


print(asyncio.run(chain()))


# asyncio.sleep is awaitable
async def sleeper():
    await asyncio.sleep(0)
    return 'slept'


print(asyncio.run(sleeper()))


# async function called from another via gather
async def add(a, b):
    return a + b


async def parallel():
    return await asyncio.gather(add(1, 2), add(3, 4), add(5, 6))


print(asyncio.run(parallel()))
