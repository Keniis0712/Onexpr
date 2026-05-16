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


print(f())
print(g())
print(h(1))
print(h(-1))
print(implicit())
print(deep(20))
print(deep(5))
print(deep(-1))
