"""_FuncHelper: tracks return value and returned flag for each function."""


class _FuncHelper:
    def __init__(self):
        self.returned = False
        self.value = None

    def do_return(self, v):
        self.returned = True
        self.value = v
        return True
