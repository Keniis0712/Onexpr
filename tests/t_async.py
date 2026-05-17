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


# Regression: PEP 525 async generator athrow / aclose / asend.
# _AsyncGenWrapper used to only implement __aiter__/__anext__.
async def ag_protocol():
    out = []

    async def ag():
        try:
            yield 1
            yield 2
        except ValueError:
            yield 'caught'

    a = ag()
    out.append(await a.__anext__())
    try:
        v = await a.athrow(ValueError())
        out.append(v)
    except StopAsyncIteration:
        pass

    async def ag2():
        x = yield 'first'
        yield ('got', x)

    b = ag2()
    out.append(await b.__anext__())
    out.append(await b.asend('SENT'))

    return out


print(asyncio.run(ag_protocol()))


# Regression: asyncio.wait_for / asyncio.timeout used to fail with
# "Timeout should be used inside a task". The _OnexprLoop's _run_once
# now restores the prior running loop while a callback runs so user
# code that queries asyncio.get_running_loop() inside a try-region
# callback sees the loop their Task actually belongs to.
async def aio_wait_for():
    async def slow():
        await asyncio.sleep(0.01)
        return 'wf-done'
    return await asyncio.wait_for(slow(), timeout=1.0)


print(asyncio.run(aio_wait_for()))


async def aio_timeout_async_with():
    async with asyncio.timeout(1.0):
        await asyncio.sleep(0)
        return 'ato-done'


print(asyncio.run(aio_timeout_async_with()))


async def aio_wait_for_timeout():
    async def really_slow():
        await asyncio.sleep(10)
        return 'never'
    try:
        return await asyncio.wait_for(really_slow(), timeout=0.05)
    except asyncio.TimeoutError:
        return 'caught-timeout'


print(asyncio.run(aio_wait_for_timeout()))


# Regression: coroutine inside coroutine with `nonlocal x`. The outer
# coroutine is also a state machine so plain helper-attribute boxing
# can't reach it; instead the nonlocal pre-pass marks the outer
# generator with a self-alias (`_gen_self_alias`) that emit_state_machine
# binds at the start of send(), and the inner's Name references are
# rewritten to <alias>.x.
async def coro_in_coro_nonlocal():
    x = 0

    async def inner():
        nonlocal x
        x += 5
        return x

    r = await inner()
    return (r, x)


print(asyncio.run(coro_in_coro_nonlocal()))


# Three-deep: outer -> inner1 -> inner2 with nonlocal at the outermost.
async def coro_three_deep_nonlocal():
    x = 'orig'

    async def inner1():
        nonlocal x

        async def inner2():
            nonlocal x
            x = 'changed'
            return x

        return await inner2()

    r = await inner1()
    return (r, x)


print(asyncio.run(coro_three_deep_nonlocal()))


# Regression: __annotations__ on async-def forwarders. FastAPI etc.
# read inspect.signature(endpoint) at decoration time to figure out
# request parsing — without annotations on the forwarder, every
# parameter looks like a query parameter.
async def annotated_coro(x: int, y: str = "hi") -> tuple[int, str]:
    return (x, y)


print(annotated_coro.__annotations__)
