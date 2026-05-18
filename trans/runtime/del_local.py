"""_del_local: best-effort `del x` for function-local variables."""


def _del_local(name):
    import sys
    frame = sys._getframe(1)
    if frame.f_locals is frame.f_globals:
        frame.f_globals.pop(name, None)
        return None
    # PEP 667 (Python 3.13+): assignment to f_locals on optimized frames
    # writes through to the fast-local slot.
    frame.f_locals[name] = None
    return None
