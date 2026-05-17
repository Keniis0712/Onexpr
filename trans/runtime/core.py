"""Core runtime helpers: _FuncHelper, _ForHelper, _WhileHelper.

These three classes are injected into the user's tree at transform
time and go through the regular onexpr pipeline along with user code,
so the final output stays a single expression with no `exec` needed.

Self-reference subtlety: a fresh function/class transformed in the
default way produces a lambda body that begins with
`(helper := _FuncHelper())`. When the function being transformed is a
method of _FuncHelper (or _ForHelper / _WhileHelper, which run before
_FuncHelper is bound during class construction) that name doesn't
exist yet — chicken-and-egg.

helpers.py marks every method here with `_use_legacy_return = True`,
which switches gen_func / parse_return / parse_class_def to the older
`(value, True)` tuple-and-[0] convention. That convention has no
external dependency: a `return X` becomes `(X, True)`, and the outer
lambda picks `[0]` out at exit. Limitation: the older convention can
not surface a `return` from inside a loop. Methods here don't do that.
"""


class _FuncHelper:
    def __init__(self):
        self.returned = False
        self.value = None

    def do_return(self, v):
        self.returned = True
        self.value = v
        return True


class _ForHelper:
    def __init__(self, iterable, func_helper):
        self.iterable = iter(iterable)
        self.stopped = False
        self.func_helper = func_helper
        self.last_yielded = None
        self.was_iterated = False
        self.pending_continue = False

    def __iter__(self):
        return self

    def __next__(self):
        # Reset per-iteration flags before checking termination.
        self.pending_continue = False
        if self.func_helper.returned:
            self.stopped = True
        if self.stopped:
            raise StopIteration
        v = next(self.iterable)
        self.last_yielded = v
        self.was_iterated = True
        return v

    def stop(self):
        self.stopped = True
        return True

    def do_continue(self):
        self.pending_continue = True
        return True


class _WhileHelper:
    def __init__(self, func_helper):
        self.stopped = False
        self.ended = False
        self.func_helper = func_helper
        self.pending_continue = False

    def __iter__(self):
        return self

    def __next__(self):
        self.pending_continue = False
        if self.func_helper.returned:
            self.stopped = True
        if self.stopped or self.ended:
            raise StopIteration
        return None

    def stop(self):
        self.stopped = True
        return True

    def do_continue(self):
        self.pending_continue = True
        return True

    def cond(self, condition):
        if condition:
            return False
        self.ended = True
        return True


# Best-effort `del x` for an optimized (function/lambda) frame.
#
# CPython does not expose any way to *unbind* a fast local from Python
# code — PEP 667 explicitly disallows del/pop on the f_locals proxy of
# an optimized frame. The closest we can do is set the slot to None,
# which (a) drops the original object reference so it can be GC'd, and
# (b) makes subsequent reads return None instead of the original value.
# `del x` does NOT raise NameError on later access, which is a known
# divergence from CPython.
#
# Requires Python 3.13+ for the function-local case (PEP 667 made
# f_locals on optimized frames a write-through proxy). On earlier
# versions the assignment is a silent no-op.
def _del_local(name):
    import sys
    frame = sys._getframe(1)
    if frame.f_locals is frame.f_globals:
        # Module-level frame — locals IS globals. Real removal is fine.
        frame.f_globals.pop(name, None)
        return None
    # Function/lambda frame. PEP 667 (Python 3.13+) makes assignment to
    # f_locals on optimized frames write through to the actual fast
    # local slot; setting the slot to None drops the original object
    # reference (so it can be GC'd) and any subsequent read returns
    # None. Earlier Python versions: the assignment is a silent no-op.
    frame.f_locals[name] = None
    return None


# Build a class the same way `class Foo(...)` does at the source
# level. The user code's "lambda body returning a dict" path skips
# the metaclass protocol entirely; this helper restores it:
#   1. Pick the right metaclass — the user's explicit `metaclass=`
#      wins; otherwise, take the most-derived metaclass among the
#      bases (Python's standard rule).
#   2. Call metaclass.__prepare__(name, bases) to obtain the
#      namespace dict the metaclass wants. For Enum that returns an
#      _EnumDict whose __setitem__ registers each member as it's set.
#   3. Move the body's bindings into that namespace via update.
#   4. Instantiate the class via the chosen metaclass.
def _make_class(metaclass, name, bases, body_dict, **kw):
    # PEP 560: any base that is not itself a class but defines
    # `__mro_entries__` is replaced by what that method returns. This
    # is what `__build_class__` does automatically for typing
    # constructs like `NamedTuple`, `Generic[T]`, `Protocol`,
    # `TypedDict`, etc. The original bases are stashed in
    # `__orig_bases__` (PEP 560) so the metaclass / __init_subclass__
    # can see the subscripted forms.
    orig_bases = bases
    new_bases = []
    used_mro_entries = False
    for b in bases:
        if not isinstance(b, type) and hasattr(b, '__mro_entries__'):
            used_mro_entries = True
            new_bases.extend(b.__mro_entries__(bases))
        else:
            new_bases.append(b)
    bases = tuple(new_bases)

    if metaclass is type and bases:
        # If the user didn't write metaclass=..., derive it from the
        # bases. Pick the most-derived; raise on incompatible siblings.
        for b in bases:
            mc = type(b)
            if issubclass(mc, metaclass):
                metaclass = mc
            elif not issubclass(metaclass, mc):
                raise TypeError(
                    'metaclass conflict: the metaclass of a derived '
                    'class must be a (non-strict) subclass of the '
                    'metaclasses of all its bases'
                )
    ns = metaclass.__prepare__(name, bases, **kw)
    # CPython injects __module__ automatically into the class
    # namespace before running the body. Some metaclasses (e.g.
    # typing.NamedTupleMeta, EnumMeta) read it during __new__.
    if '__module__' not in body_dict:
        import sys as _sys
        # _getframe(1) is _make_class's caller — i.e. the frame that
        # holds the user's class statement. Fall back to '__main__'
        # when sys is somehow not available (it always is in
        # CPython, so this branch is just defensive).
        f = _sys._getframe(1) if hasattr(_sys, '_getframe') else None
        ns['__module__'] = (
            f.f_globals.get('__name__', '__main__') if f is not None
            else '__main__'
        )
    if used_mro_entries:
        ns['__orig_bases__'] = orig_bases
    ns.update(body_dict)
    return metaclass(name, bases, ns, **kw)


# Coroutine plumbing: an `await x` expression in user code lowers to
# `yield from _await_iter(x)`. _await_iter takes either a real
# coroutine (objects implementing __await__) or one of our fake
# generator-as-coroutines (created by `async def` lowering — they're
# plain generators registered as collections.abc.Coroutine via
# helpers.py's add_helper) and returns the iterator that `yield from`
# can drive.
def _await_iter(x):
    if hasattr(x, '__await__'):
        return x.__await__()
    return iter(x)


# Async generator wrapper: an `async def f(): yield V` body becomes
# a plain generator state machine; we wrap its instance in this
# class so it satisfies the async iterator protocol (__aiter__ /
# __anext__). __anext__ is itself a coroutine that drives the
# underlying generator until it sees a user-level yield (wrapped in
# _UserYield), passing through any intermediate yields (from awaits
# inside the body) to the surrounding scheduler.
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
        # Not a user yield — this is an intermediate yield from an
        # `await` somewhere in the async generator's body. Propagate
        # it to the surrounding scheduler so it can resume us when
        # the awaited operation completes.
        yield v


def _async_gen_asend(g, value):
    # First step: send `value` into the body to be received by the
    # most-recent `yield`. Subsequent steps drive the generator with
    # plain next() until the next user-level yield.
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
    # Inject the exception at the suspended yield. If the body
    # catches it and yields again (user-level), surface that value;
    # otherwise StopIteration → StopAsyncIteration.
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
    # Inject GeneratorExit. The body's finally clauses run; if the
    # body swallows GeneratorExit and yields again, that's a
    # protocol violation (RuntimeError per PEP 525).
    try:
        v = g.throw(GeneratorExit)
    except (GeneratorExit, StopIteration, StopAsyncIteration):
        return
    except BaseException:
        raise
    if isinstance(v, _UserYield):
        raise RuntimeError('async generator ignored GeneratorExit')
    # Intermediate yield from a finally-clause await — keep driving.
    while True:
        yield v
        try:
            v = next(g)
        except (GeneratorExit, StopIteration, StopAsyncIteration):
            return
        if isinstance(v, _UserYield):
            raise RuntimeError('async generator ignored GeneratorExit')


# Mark generator-protocol forwarders as coroutines so `await` accepts them.
_async_gen_anext = __import__('types').coroutine(_async_gen_anext)
_async_gen_asend = __import__('types').coroutine(_async_gen_asend)
_async_gen_athrow = __import__('types').coroutine(_async_gen_athrow)
_async_gen_aclose = __import__('types').coroutine(_async_gen_aclose)

