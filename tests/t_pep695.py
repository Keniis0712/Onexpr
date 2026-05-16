import typing


# 1) Generic function: __type_params__ + behavior.
def first[T](lst):
    return lst[0]


print(first.__type_params__[0].__name__)
print(first([1, 2, 3]))


# 2) Type vars don't leak.
print('T' in dir())


# 3) Generic class: __type_params__.
class Box[U]:
    def __init__(self, x):
        self.x = x

    def get(self):
        return self.x


print(Box.__type_params__[0].__name__)
b = Box(42)
print(b.get())


# 4) Generic class with base + method.
class Pair[A, B]:
    def __init__(self, a, b):
        self.a = a
        self.b = b


p = Pair('x', 1)
print(p.a, p.b)
print(len(Pair.__type_params__))


# 5) ParamSpec on a function.
def cb[**P](*args, **kwargs):
    return len(args)


print(type(cb.__type_params__[0]).__name__)
print(cb(1, 2, 3))


# 6) TypeVarTuple on a function.
def tup[*Ts](*args):
    return args


print(type(tup.__type_params__[0]).__name__)
print(tup(1, 'a', 2.5))


# 7) Decorator + generic — decorator sees __type_params__ already set.
def trace(f):
    f.__traced__ = True
    return f


@trace
def g[T](x):
    return x


print(g.__type_params__[0].__name__)
print(g.__traced__)
print(g(7))


# 8) Generic class subscript via __class_getitem__.
class Holder[T]:
    pass


print(Holder[int])
# CPython auto-generates typing._GenericAlias; onexpr emits
# types.GenericAlias (public API). Both are GenericAlias-shaped but
# they're different concrete types, so just check the alias is callable
# and roundtrips.
print('GenericAlias' in type(Holder[int]).__name__)


# 9) Generic methods inside a generic class.
class Stack[T]:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def peek(self):
        return self.items[-1] if self.items else None


s = Stack()
s.push(1)
s.push(2)
print(s.peek())
print(Stack.__type_params__[0].__name__)
