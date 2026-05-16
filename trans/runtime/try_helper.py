"""Runtime helper for `try`/`except`/`finally`.

This file is read at transform time (NOT imported), parsed into AST,
and injected into the user's tree as top-level statements. It then
goes through the full onexpr transformation along with user code, so
the final output is still a single expression.

The technique — making the asyncio event loop reentrant by replacing
run_forever / _run_once / _check_running and inlining manage_run —
is distilled from nest_asyncio (BSD-2-Clause, (c) Ewald de Wit,
https://github.com/erdewit/nest_asyncio). We only need nested
`run_forever()`, so we skip the Task/Future patches and the
run_until_complete replacement, and we use SelectorEventLoop
unconditionally so the proactor branches can be dropped.

Importantly, we patch a LOCAL SUBCLASS (_OnexprLoop) of
SelectorEventLoop instead of the framework class itself. Otherwise
importing an onexpr-obfuscated module from a normal asyncio program
would silently re-route every loop's run_forever / _check_running,
which would be very surprising.

This module deliberately uses no `try` keyword. The two try/finally
blocks in the upstream code are handled by:
- run_forever: setup/teardown inlined (the loop body doesn't raise,
  so finally cleanup always runs anyway)
- _run_once: handle._run() catches its own exceptions and reports them
  via call_exception_handler, so it never propagates to its caller
"""
import asyncio
import asyncio.events as _events
import sys
import threading
from heapq import heappop


class _OnexprLoop(asyncio.SelectorEventLoop):
    pass


def _onexpr_patch_loop_cls(cls):
    if getattr(cls, '_nest_patched', False):
        return

    def run_forever(self):
        self._check_closed()
        old_thread_id = self._thread_id
        old_running_loop = _events._get_running_loop()
        self._thread_id = threading.get_ident()
        _events._set_running_loop(self)
        self._num_runs_pending += 1

        while True:
            self._run_once()
            if self._stopping:
                break

        self._thread_id = old_thread_id
        _events._set_running_loop(old_running_loop)
        self._num_runs_pending -= 1
        self._stopping = False

    def _run_once(self):
        ready = self._ready
        scheduled = self._scheduled
        while scheduled and scheduled[0]._cancelled:
            heappop(scheduled)

        timeout = (
            0 if ready or self._stopping
            else min(max(scheduled[0]._when - self.time(), 0), 86400)
            if scheduled else None
        )
        event_list = self._selector.select(timeout)
        self._process_events(event_list)

        end_time = self.time() + self._clock_resolution
        while scheduled and scheduled[0]._when < end_time:
            handle = heappop(scheduled)
            ready.append(handle)

        for _ in range(len(ready)):
            if not ready:
                break
            handle = ready.popleft()
            if not handle._cancelled:
                curr_task = curr_tasks.pop(self, None)
                handle._run()
                if curr_task is not None:
                    curr_tasks[self] = curr_task

    def _check_running(self):
        pass

    cls.run_forever = run_forever
    cls._run_once = _run_once
    cls._check_running = _check_running
    cls._check_runnung = _check_running
    cls._num_runs_pending = 0
    curr_tasks = (
        asyncio.tasks._current_tasks
        if sys.version_info >= (3, 7) else asyncio.Task._current_tasks
    )
    cls._nest_patched = True


_onexpr_patch_loop_cls(_OnexprLoop)


class _TryHelper:
    loop = _OnexprLoop()

    @staticmethod
    def guarded(fn):
        loop = _TryHelper.loop
        captured = [None]
        prev_handler = loop.get_exception_handler()

        def h(l, ctx):
            captured[0] = ctx.get('exception')
            l.stop()

        loop.set_exception_handler(h)

        def wrapped():
            fn()
            loop.stop()

        loop.call_soon(wrapped)
        loop.run_forever()
        loop.set_exception_handler(prev_handler)
        return captured[0]

    @staticmethod
    def dispatch(body_fn, handlers, else_fn, func_helper, loop_helper):
        """Run the try body, route through handlers / else, and return
        the exception that should propagate (or None).

        - body_fn: zero-arg callable for the try body
        - handlers: list of (exc_types_or_None, handler_callable) — the
          handler callable takes one argument (the caught exception)
        - else_fn: zero-arg callable for the `else` clause, or None
        - func_helper: the current function's _FuncHelper, used to
          detect whether body did `return`
        - loop_helper: the innermost enclosing loop's helper, used to
          detect whether body did `break`; None if not in a loop
        """
        e1 = _TryHelper.guarded(body_fn)
        if e1 is None:
            terminated = func_helper.returned
            if not terminated and loop_helper is not None:
                terminated = loop_helper.stopped
            if not terminated and else_fn is not None:
                e_else = _TryHelper.guarded(else_fn)
                # `else` ran with no in-flight exception, no context to chain.
                return e_else
            return None
        # body raised
        for exc_types, handler in handlers:
            if exc_types is None or isinstance(e1, exc_types):
                e2 = _TryHelper.guarded(lambda: handler(e1))
                # If the handler itself raised, Python sets the new
                # exception's __context__ to the one it was handling.
                # CPython does this in RAISE_VARARGS by reading the
                # current exception state; our generator-throw trick
                # runs outside of that exception state, so we re-create
                # the chain explicitly.
                if e2 is not None and e2.__context__ is None and e2 is not e1:
                    e2.__context__ = e1
                return e2
        # unhandled — surface the body exception
        return e1
