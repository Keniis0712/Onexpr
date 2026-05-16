# Cross-construct nesting regression cases. Each one was hand-verified
# during the PEP 695 / nesting audit, so any future change that breaks
# the round-trip semantics will fail here.


# 1) try inside match case.
def try_in_case():
    out = []
    for x in range(3):
        match x:
            case 0:
                try:
                    raise ValueError('v')
                except ValueError:
                    out.append(('caught', x))
            case _:
                out.append(('plain', x))
    return out


print(try_in_case())


# 2) match inside try body, raise from a case caught by the surrounding
#    except.
def match_in_try_body():
    out = []
    try:
        for x in (1, 2, 3):
            match x:
                case 2:
                    raise RuntimeError('at 2')
                case _:
                    out.append(x)
    except RuntimeError:
        out.append('caught')
    return out


print(match_in_try_body())


# 3) match inside an except handler, dispatching on the captured value.
def match_in_except():
    out = []
    try:
        raise ValueError(('tag', 42))
    except ValueError as e:
        arg = e.args[0]
        match arg:
            case ('tag', n):
                out.append(n)
            case _:
                out.append('other')
    return out


print(match_in_except())


# 4) match inside finally.
def match_in_finally():
    out = []
    mode = 'A'
    try:
        out.append('body')
    finally:
        match mode:
            case 'A':
                out.append('A-final')
            case 'B':
                out.append('B-final')
    return out


print(match_in_finally())


# 5) deep nesting: while > try > match > for > break.
def deep_nesting():
    out = []
    i = 0
    while i < 4:
        try:
            match i:
                case 1:
                    for j in range(10):
                        if j == 3:
                            break
                        out.append(('inner', i, j))
                case 2:
                    raise ValueError('two')
                case _:
                    out.append(('default', i))
        except ValueError:
            out.append(('caught', i))
        i += 1
    return out


print(deep_nesting())


# 6) return from inside case inside try in function: the return signal
#    must propagate through the per-clause lambda boundary AND the
#    surrounding loop.
def return_from_nested():
    for x in range(5):
        try:
            match x:
                case 3:
                    return ('hit', x)
                case _:
                    pass
        finally:
            pass
    return None


print(return_from_nested())


# 7) match inside while with continue and break from cases.
def match_in_while():
    out = []
    i = 0
    while i < 5:
        match i:
            case 2:
                i += 1
                continue
            case 4:
                break
            case _:
                out.append(i)
        i += 1
    return out


print(match_in_while())


# 8) raise from inside case, surrounding except catches.
def raise_in_case():
    out = []
    for x in (1, 2, 3):
        try:
            match x:
                case 2:
                    raise KeyError(x)
                case _:
                    out.append(('ok', x))
        except KeyError as e:
            out.append(('err', e.args[0]))
    return out


print(raise_in_case())


# 9) bare `raise` inside a case inside an except handler — must rethrow
#    the active exception.
def bare_raise_in_case_in_except():
    def f():
        try:
            raise ValueError('v')
        except ValueError as e:
            match e.args[0]:
                case 'v':
                    raise
                case _:
                    return 'no-match'

    try:
        f()
        return 'unreached'
    except ValueError as e:
        return ('rethrown', e.args[0])


print(bare_raise_in_case_in_except())


# 10) finally that does break — the pending exception must be swallowed
#     and the loop must exit.
def finally_break():
    out = []
    for x in range(5):
        try:
            out.append(('body', x))
            if x == 3:
                raise ValueError()
        except ValueError:
            out.append(('caught', x))
        finally:
            if x == 2:
                break
            out.append(('fin', x))
    out.append('done')
    return out


print(finally_break())


# 11) while-else with match inside — the else must run because no break
#     fires.
def while_else_with_match():
    out = []
    i = 0
    while i < 3:
        match i:
            case 99:
                out.append('hit')
                break
        out.append(i)
        i += 1
    else:
        out.append('else')
    return out


print(while_else_with_match())


# 12) for-else where the break comes from a case body — else must NOT
#     run.
def for_else_break_from_case():
    out = []
    for x in range(5):
        match x:
            case 2:
                out.append(('break-at', x))
                break
            case _:
                out.append(x)
    else:
        out.append('else')
    return out


print(for_else_break_from_case())


# 13) try* / except* with match inside the handler.
def except_star_with_match():
    out = []
    try:
        raise ExceptionGroup('g', [ValueError('v'), TypeError('t')])
    except* ValueError as eg:
        for e in eg.exceptions:
            match str(e):
                case 'v':
                    out.append('matched-v')
                case _:
                    out.append(('other', str(e)))
    except* TypeError as eg:
        out.append('type')
    return out


print(except_star_with_match())


# 14) match guard that closes over an outer (mutable) variable.
THRESHOLD = 3


def guard_with_closure():
    def classify(x):
        match x:
            case n if n > THRESHOLD:
                return 'big'
            case n:
                return 'small'

    return [classify(i) for i in range(6)]


print(guard_with_closure())


# 15) Nested match.
def nested_match():
    def f(p):
        match p:
            case (a, b):
                match (a, b):
                    case (0, _):
                        return 'zero-x'
                    case (_, 0):
                        return 'zero-y'
                    case _:
                        return ('both', a, b)
            case _:
                return 'not-pair'

    return [f((0, 5)), f((5, 0)), f((1, 1)), f('nope')]


print(nested_match())


# 16) MatchOr alternatives sharing names + guard.
def or_with_guard():
    def f(x):
        match x:
            case [a, b] | [a, b, _] if a == b:
                return ('eq', a)
            case [a, *_]:
                return ('first', a)
            case _:
                return 'none'

    return [f([1, 1]), f([1, 1, 2]), f([1, 2]), f([3, 4, 5]), f('hi')]


print(or_with_guard())


# 17) Match guard that raises — the surrounding try must catch it.
def guard_raises():
    def f(x):
        try:
            match x:
                case n if 1 / n:
                    return 'ok'
                case _:
                    return 'default'
        except ZeroDivisionError:
            return 'zero'

    return [f(5), f(0), f(-3)]


print(guard_raises())


# 18) Module-top except handler assigns a name — the assignment must
#     reach module globals (so subsequent module-level code sees it).
try:
    raise ValueError('top')
except ValueError as e:
    TOP_LEVEL = ('top-caught', e.args[0])

print(TOP_LEVEL)


# 19) Module-top try body assigns a name.
try:
    TOP_BODY = 'set-in-body'
except Exception:
    TOP_BODY = 'set-in-handler'

print(TOP_BODY)


# 20) Module-top try-finally assigns; finalbody also assigns.
try:
    TOP_TRY = 'try-val'
finally:
    TOP_FIN = 'fin-val'

print(TOP_TRY)
print(TOP_FIN)
