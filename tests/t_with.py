class CM:
    def __init__(self, name, swallow=False):
        self.name = name
        self.swallow = swallow

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        return self.swallow


def basic_with():
    with CM('hello') as v:
        return v


print(basic_with())


def with_no_as():
    out = []
    with CM('x'):
        out.append('inside')
    return out


print(with_no_as())


def with_assigns_in_body():
    with CM('y'):
        a = 1
        b = 2
    return (a, b)


print(with_assigns_in_body())


class Logger:
    def __init__(self):
        self.log = []

    def __enter__(self):
        self.log.append('enter')
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.append(('exit', exc_type.__name__ if exc_type else None))
        return False


def with_records():
    log = Logger()
    with log:
        pass
    return log.log


print(with_records())


def with_propagates_exception():
    log = Logger()

    def f():
        with log:
            raise ValueError('boom')

    try:
        f()
    except ValueError as e:
        return (log.log, str(e))


print(with_propagates_exception())


def with_swallows_exception():
    cm = CM('s', swallow=True)
    with cm:
        raise ValueError('eaten')
    return 'after'


print(with_swallows_exception())


def multi_with():
    out = []
    with CM('a') as a, CM('b') as b:
        out.append((a, b))
    return out


print(multi_with())


def nested_with():
    out = []
    with CM('outer') as a:
        with CM('inner') as b:
            out.append((a, b))
    return out


print(nested_with())


def return_inside_with():
    log = Logger()

    def f():
        with log:
            return 'returned'

    r = f()
    return (r, log.log)


print(return_inside_with())


def break_inside_with_in_loop():
    log = Logger()

    def f():
        out = []
        for i in range(5):
            with log:
                if i == 2:
                    break
                out.append(i)
        return out

    return (f(), log.log)


print(break_inside_with_in_loop())


def continue_inside_with_in_loop():
    log = Logger()

    def f():
        out = []
        for i in range(4):
            with log:
                if i % 2 == 0:
                    continue
                out.append(i)
        return out

    return (f(), len(log.log))


print(continue_inside_with_in_loop())


def with_inside_try():
    log = Logger()
    try:
        with log:
            raise ValueError('x')
    except ValueError as e:
        return (str(e), log.log)


print(with_inside_try())


def try_inside_with():
    log = Logger()
    with log:
        try:
            raise ValueError('x')
        except ValueError:
            pass
    return log.log


print(try_inside_with())


# Real builtin context manager
def with_open_temp():
    import io
    with io.StringIO('payload') as buf:
        contents = buf.getvalue()
    return contents


print(with_open_temp())


# contextlib.contextmanager-like generator-based isn't supported
# (we don't support yield), but a regular class-based one is.
class Counter:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *a):
        self.exited += 1
        return False


def reuse_cm():
    c = Counter()
    with c:
        pass
    with c:
        pass
    with c:
        pass
    return (c.entered, c.exited)


print(reuse_cm())


def exit_reads_exc_info():
    seen = []

    class Inspect:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            seen.append((exc_type.__name__ if exc_type else None,
                         str(exc) if exc else None))
            return True
    with Inspect():
        raise RuntimeError('peek')
    return seen


print(exit_reads_exc_info())


def with_assign_target_via_unpack():
    class CMTuple:
        def __enter__(self):
            return (1, 2, 3)

        def __exit__(self, *a):
            return False
    with CMTuple() as (a, b, c):
        pass
    return (a, b, c)


print(with_assign_target_via_unpack())
