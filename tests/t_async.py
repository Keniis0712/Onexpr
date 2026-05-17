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


# Async generator with await between yields
async def producer_with_await():
    yield 'a'
    await asyncio.sleep(0)
    yield 'b'
    await asyncio.sleep(0)
    yield 'c'


async def consume_with_await():
    out = []
    async for x in producer_with_await():
        out.append(x)
    return out


print(asyncio.run(consume_with_await()))


# Async list comprehension
async def producer_3():
    for i in range(3):
        yield i


async def list_comp():
    return [x * 10 async for x in producer_3()]


print(asyncio.run(list_comp()))


async def list_comp_filter():
    return [x async for x in producer_3() if x > 0]


print(asyncio.run(list_comp_filter()))


# Async dict comprehension
async def keygen():
    for k in ('x', 'y', 'z'):
        yield k


async def dict_comp():
    return {k: ord(k) async for k in keygen()}


print(asyncio.run(dict_comp()))


# Async set comprehension
async def set_comp():
    return {x * x async for x in producer_3()}


print(asyncio.run(set_comp()))


# Async generator: await result then yield
async def helper_compute():
    await asyncio.sleep(0)
    return 100


async def gen_with_await_result():
    val = await helper_compute()
    yield val
    yield val + 1


async def consume_with_result():
    out = []
    async for x in gen_with_await_result():
        out.append(x)
    return out


print(asyncio.run(consume_with_result()))



# Regression: bare `try: return X; finally: ...` in an async function.
# The fast path used to keep the try at the lambda level, where the
# return short-circuits send() and the value gets yielded to asyncio.
async def coro_try_return_finally():
    try:
        return "inner-try"
    finally:
        pass


print(asyncio.run(coro_try_return_finally()))


# Regression: try with await + return in body, plus finally. The
# nonlocal pre-pass was boxing the awaited result onto the outer
# func helper, so the state machine could not find it.
async def coro_try_await_return_finally():
    try:
        x = await asyncio.sleep(0, result="awaited")
        return x
    finally:
        pass


print(asyncio.run(coro_try_await_return_finally()))


# Regression: `as e` in a coroutine try crossing await. The except
# handler block needs its dispatcher seen by the body before the body
# is emitted.
async def coro_try_except_await():
    try:
        await asyncio.sleep(0)
        raise ValueError("v")
    except ValueError as e:
        return ("caught", str(e))


print(asyncio.run(coro_try_except_await()))
