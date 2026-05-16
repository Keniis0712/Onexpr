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

