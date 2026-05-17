def gen_basic():
    yield 1
    yield 2
    yield 3


print(list(gen_basic()))


def gen_for():
    for i in range(4):
        yield i


print(list(gen_for()))


def gen_nested_for():
    for i in range(3):
        for j in range(2):
            yield (i, j)


print(list(gen_nested_for()))


def gen_with_break():
    for i in range(10):
        if i >= 3:
            break
        yield i


print(list(gen_with_break()))


def gen_with_continue():
    for i in range(6):
        if i % 2 == 0:
            continue
        yield i


print(list(gen_with_continue()))


def gen_while():
    n = 0
    while n < 4:
        yield n
        n = n + 1


print(list(gen_while()))


def gen_while_with_break():
    n = 0
    while True:
        if n >= 5:
            break
        yield n
        n += 1


print(list(gen_while_with_break()))


def gen_if_branches(flag):
    yield 'start'
    if flag:
        yield 'pos'
    else:
        yield 'neg'
    yield 'end'


print(list(gen_if_branches(True)))
print(list(gen_if_branches(False)))


def gen_yield_from():
    yield from [1, 2, 3]
    yield 'after'


print(list(gen_yield_from()))


def gen_yield_from_two():
    yield from range(3)
    yield from range(10, 13)


print(list(gen_yield_from_two()))


def gen_with_args(*xs):
    for x in xs:
        yield x * 2


print(list(gen_with_args(1, 2, 3)))


def gen_pipeline():
    def double():
        for i in range(3):
            yield i * 2

    def squared():
        for x in double():
            yield x * x
    return squared()


print(list(gen_pipeline()))


def gen_with_assignments():
    x = 10
    for i in range(3):
        y = i * x
        yield y


print(list(gen_with_assignments()))


def gen_with_return_value():
    yield 1
    yield 2
    return 'done'


g = gen_with_return_value()
out = []
val = '_DONE'
while True:
    try:
        out.append(next(g))
    except StopIteration as e:
        val = e.value
        break
print(out, val)


def gen_loop_then_yield():
    total = 0
    for i in range(5):
        total = total + i
    yield total
    yield total * 2


print(list(gen_loop_then_yield()))


def gen_with_match():
    for x in range(5):
        match x % 3:
            case 0:
                yield ('zero', x)
            case 1:
                yield ('one', x)
            case _:
                yield ('other', x)


print(list(gen_with_match()))


def gen_with_match_capture():
    for x in [(1, 2), (3, 4), (5, 6)]:
        match x:
            case (a, b):
                yield a + b


print(list(gen_with_match_capture()))


def gen_with_match_guard():
    for x in range(10):
        match x:
            case n if n % 2 == 0:
                yield ('even', n)
            case n:
                yield ('odd', n)


print(list(gen_with_match_guard()))


def gen_with_nested_def():
    def helper(x):
        return x * 2

    for i in range(3):
        yield helper(i)


print(list(gen_with_nested_def()))


def gen_with_nested_class():
    class Pair:
        def __init__(self, a, b):
            self.a, self.b = a, b

    for i in range(3):
        p = Pair(i, i * 2)
        yield (p.a, p.b)


print(list(gen_with_nested_class()))


def gen_with_walrus():
    items = iter([1, 2, 3, None, 4])
    while (chunk := next(items)) is not None:
        yield chunk


print(list(gen_with_walrus()))


def gen_with_walrus_in_if():
    for i in range(6):
        if (sq := i * i) > 4:
            yield sq


print(list(gen_with_walrus_in_if()))
