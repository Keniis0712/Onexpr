def f():
    return 1


def g():
    return


def h(x):
    if x > 0:
        return 'pos'
    return 'neg'


def implicit():
    x = 10


def deep(x):
    if x > 0:
        if x > 10:
            return 'big'
        return 'small pos'
    return 'neg'


def early_for(xs):
    for x in xs:
        if x == 0:
            return 'zero'
    return 'no zero'


def early_while(n):
    i = 0
    while i < n:
        if i == 3:
            return i
        i = i + 1
    return -1


def nested(xs):
    for x in xs:
        for y in x:
            if y == 'stop':
                return ('found', x, y)
    return None


def loop_after_inner_return():
    for i in range(5):
        if i == 2:
            return ('inner', i)
        for j in range(5):
            if j == 4:
                return ('deep', i, j)
    return 'never'


def for_else_when_returning(xs):
    for x in xs:
        if x < 0:
            return 'neg found'
    else:
        return 'no neg'
    return 'unreachable'


def while_else_when_returning(n):
    i = 0
    while i < n:
        if i == 999:
            return 'special'
        i = i + 1
    else:
        return 'finished'
    return 'unreachable'


print(f())
print(g())
print(h(1))
print(h(-1))
print(implicit())
print(deep(20))
print(deep(5))
print(deep(-1))
print(early_for([1, 2, 0, 3]))
print(early_for([1, 2, 3]))
print(early_while(10))
print(early_while(2))
print(nested([['a', 'b'], ['c', 'stop', 'd'], ['e']]))
print(nested([['a'], ['b']]))
print(loop_after_inner_return())
print(for_else_when_returning([1, 2, 3]))
print(for_else_when_returning([1, -2, 3]))
print(while_else_when_returning(3))
