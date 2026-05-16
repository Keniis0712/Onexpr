def empty_body():
    pass


print(empty_body())
print(empty_body() is None)


class EmptyClass:
    pass


e = EmptyClass()
print(type(e).__name__)


class OnlyAttrs:
    x = 1
    y = 'hi'


print(OnlyAttrs.x, OnlyAttrs.y)


def only_assign():
    x = 5
    y = x + 1


print(only_assign())


def only_print():
    print('side effect')


print(only_print())


def conditional_no_else_return(x):
    if x > 0:
        return 'pos'


print(conditional_no_else_return(1))
print(conditional_no_else_return(-1))


def returns_none_explicit():
    return None


def returns_none_implicit():
    pass


def returns_no_value():
    return


print(returns_none_explicit() is None)
print(returns_none_implicit() is None)
print(returns_no_value() is None)


def deeply(n):
    if n == 0:
        return 'bottom'
    return deeply(n - 1)


print(deeply(20))


def loops_in_loops():
    out = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 1 and j == 2:
                    return out
                out.append((i, j, k))
    return out


print(loops_in_loops())


def loop_then_return_none(xs):
    for x in xs:
        if x == 'mark':
            return
        print('saw', x)


print(loop_then_return_none([1, 2, 'mark', 4]))


def while_with_break_and_return(n):
    i = 0
    while True:
        if i == n:
            return ('reached', i)
        if i > 10:
            break
        i += 1
    return ('broke', i)


print(while_with_break_and_return(3))
print(while_with_break_and_return(20))


def returns_in_comprehension_call():
    def helper():
        return [i for i in range(3)]
    return helper()


print(returns_in_comprehension_call())


def f1():
    return 1


def f2():
    return f1() + 1


def f3():
    return f2() + f1()


print(f1(), f2(), f3())


def with_walrus(n):
    return [(y := i, y * y) for i in range(n)]


print(with_walrus(4))


def captures_loop_var():
    fns = []
    for i in range(3):
        fns.append(lambda i=i: i)
    return [f() for f in fns]


print(captures_loop_var())


def returns_class_object():
    class Local:
        def method(self):
            return 'from local'
    return Local


L = returns_class_object()
print(L().method())
print(L is returns_class_object())


def bool_short_circuit_args():
    def t():
        print('t called')
        return True

    def f():
        print('f called')
        return False
    print('---')
    r = t() or f()
    print(r)
    print('---')
    r = f() or t()
    print(r)
    print('---')
    r = t() and f()
    print(r)
    print('---')
    r = f() and t()
    print(r)


bool_short_circuit_args()
