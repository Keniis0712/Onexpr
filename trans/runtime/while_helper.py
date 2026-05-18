"""_WhileHelper: drives while-loop ListComp, propagates break/continue/return."""


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
