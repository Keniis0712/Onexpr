class Base:
    def __init__(self, **kw):
        self.kw = kw

    def describe(self):
        return f'Base({sorted(self.kw.items())})'


class Mid(Base):
    def __init__(self, x, **kw):
        super().__init__(**kw)
        self.x = x

    def describe(self):
        return f'Mid(x={self.x}, parent={super().describe()})'


class Leaf(Mid):
    def __init__(self, x, y, **kw):
        super().__init__(x, **kw)
        self.y = y

    def describe(self):
        return f'Leaf(y={self.y}, parent={super().describe()})'


leaf = Leaf(1, 2, foo='bar', baz=42)
print(leaf.describe())


class WithKwOnly:
    def __init__(self, *, a, b=10):
        self.a = a
        self.b = b


w = WithKwOnly(a=1)
print(w.a, w.b)
w2 = WithKwOnly(a=1, b=99)
print(w2.a, w2.b)


def passes_kw(f, *args, **kwargs):
    return f(*args, **kwargs)


def takes_named(name, age=18, *, role='user'):
    return f'{name}, {age}, {role}'


print(passes_kw(takes_named, 'alice'))
print(passes_kw(takes_named, 'bob', age=20))
print(passes_kw(takes_named, 'eve', role='admin'))
print(passes_kw(takes_named, 'frank', 25, role='admin'))


class StoresAll:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def show(self):
        return (self.args, sorted(self.kwargs.items()))


s = StoresAll(1, 2, 3, x='a', y='b')
print(s.show())


def double_call_super_in_one_method():
    class P:
        def m(self):
            return 'P'

    class C(P):
        def m(self):
            return super().m() + super().m() + 'C'
    return C().m()


print(double_call_super_in_one_method())


def super_inside_lambda_via_method():
    class P:
        def m(self):
            return 'P'

    class C(P):
        def factory(self):
            base = super().m()
            return lambda: base + '-from-lambda'
    return C().factory()()


print(super_inside_lambda_via_method())


class Box:
    def __init__(self, v):
        self._v = v

    def __repr__(self):
        return f'Box({self._v})'


class MutableBox(Box):
    def __init__(self, v):
        super().__init__(v)

    def set(self, v):
        self._v = v


b = MutableBox(0)
print(b)
b.set(42)
print(b)


def kwargs_with_defaults_chain(a, b=2, c=3, *, d=4, e=5):
    return (a, b, c, d, e)


print(kwargs_with_defaults_chain(1))
print(kwargs_with_defaults_chain(1, 20))
print(kwargs_with_defaults_chain(1, c=30))
print(kwargs_with_defaults_chain(1, e=50))
print(kwargs_with_defaults_chain(a=100, e=500))


def deep_recursion_kw(n, acc=0):
    if n == 0:
        return acc
    return deep_recursion_kw(n - 1, acc=acc + n)


print(deep_recursion_kw(10))
print(deep_recursion_kw(100))


def tricky_default_capture():
    captured = []
    def inner(x, log=captured):
        log.append(x)
        return list(log)
    return inner


t = tricky_default_capture()
print(t(1))
print(t(2))
print(t(3))


def boolean_returning(value):
    return bool(value) and (value if isinstance(value, int) else len(value)) > 0


print(boolean_returning(0))
print(boolean_returning(-1))
print(boolean_returning(5))
print(boolean_returning([]))
print(boolean_returning([1]))


def chained_method_with_return(strs):
    parts = []
    for s in strs:
        if not s:
            return None
        parts.append(s.strip().upper())
    return '-'.join(parts)


print(chained_method_with_return(['a', ' b ', 'c ']))
print(chained_method_with_return(['a', '', 'b']))
