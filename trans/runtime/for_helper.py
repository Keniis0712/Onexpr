"""_ForHelper: drives for-loop ListComp, propagates break/continue/return."""


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
