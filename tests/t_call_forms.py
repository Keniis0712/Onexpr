def basic(a, b):
    return a + b


print(basic(1, 2))
print(basic(b=10, a=20))
print(basic(a=100, b=200))


def with_default(a, b=5):
    return a + b


print(with_default(1))
print(with_default(1, 2))
print(with_default(1, b=99))


def variadic(*args):
    return sum(args)


print(variadic())
print(variadic(1))
print(variadic(1, 2, 3, 4))


def kwvariadic(**kwargs):
    return sorted(kwargs.items())


print(kwvariadic())
print(kwvariadic(a=1, b=2))


def both(*args, **kwargs):
    return (args, sorted(kwargs.items()))


print(both(1, 2, a=3))
print(both())


def keyword_only(a, *, b, c=10):
    return a, b, c


print(keyword_only(1, b=2))
print(keyword_only(1, b=2, c=3))


def position_only(a, b, /, c):
    return a + b + c


print(position_only(1, 2, 3))
print(position_only(1, 2, c=3))


def mix(a, b=2, *args, c, d=4, **kwargs):
    return (a, b, args, c, d, sorted(kwargs.items()))


print(mix(1, c=10))
print(mix(1, 20, 30, 40, c=10, d=99, x=1, y=2))


def call_with_unpack(a, b, c):
    return a * 100 + b * 10 + c


t = (1, 2, 3)
print(call_with_unpack(*t))
d = {'a': 9, 'b': 8, 'c': 7}
print(call_with_unpack(**d))
print(call_with_unpack(*[4], **{'b': 5, 'c': 6}))
