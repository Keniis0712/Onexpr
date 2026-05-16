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

    def __iter__(self):
        return self

    def __next__(self):
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


class _WhileHelper:
    def __init__(self, func_helper):
        self.stopped = False
        self.ended = False
        self.func_helper = func_helper

    def __iter__(self):
        return self

    def __next__(self):
        if self.func_helper.returned:
            self.stopped = True
        if self.stopped or self.ended:
            raise StopIteration
        return None

    def stop(self):
        self.stopped = True
        return True

    def cond(self, condition):
        if condition:
            return False
        self.ended = True
        return True
