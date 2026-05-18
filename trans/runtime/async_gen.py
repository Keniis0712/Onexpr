"""Async generator support: _UserYield marker, _AsyncGenWrapper, protocol
coroutines (_async_gen_anext/asend/athrow/aclose)."""


class _UserYield:
    def __init__(self, v):
        self.v = v


class _AsyncGenWrapper:
    def __init__(self, gen):
        self._g = gen

    def __aiter__(self):
        return self

    def __anext__(self):
        return _async_gen_anext(self._g)

    def asend(self, value):
        return _async_gen_asend(self._g, value)

    def athrow(self, typ, val=None, tb=None):
        return _async_gen_athrow(self._g, typ, val, tb)

    def aclose(self):
        return _async_gen_aclose(self._g)


def _async_gen_anext(g):
    while True:
        try:
            v = next(g)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(v, _UserYield):
            return v.v
        yield v


def _async_gen_asend(g, value):
    first = True
    while True:
        try:
            if first:
                v = g.send(value)
                first = False
            else:
                v = next(g)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(v, _UserYield):
            return v.v
        yield v


def _async_gen_athrow(g, typ, val=None, tb=None):
    first = True
    while True:
        try:
            if first:
                v = g.throw(typ, val, tb)
                first = False
            else:
                v = next(g)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(v, _UserYield):
            return v.v
        yield v


def _async_gen_aclose(g):
    try:
        v = g.throw(GeneratorExit)
    except (GeneratorExit, StopIteration, StopAsyncIteration):
        return
    except BaseException:
        raise
    if isinstance(v, _UserYield):
        raise RuntimeError('async generator ignored GeneratorExit')
    while True:
        yield v
        try:
            v = next(g)
        except (GeneratorExit, StopIteration, StopAsyncIteration):
            return
        if isinstance(v, _UserYield):
            raise RuntimeError('async generator ignored GeneratorExit')


_async_gen_anext = __import__('types').coroutine(_async_gen_anext)
_async_gen_asend = __import__('types').coroutine(_async_gen_asend)
_async_gen_athrow = __import__('types').coroutine(_async_gen_athrow)
_async_gen_aclose = __import__('types').coroutine(_async_gen_aclose)
