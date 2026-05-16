class V:
    def __init__(self, *xs):
        self.xs = list(xs)

    def __add__(self, other):
        return V(*[a + b for a, b in zip(self.xs, other.xs)])

    def __sub__(self, other):
        return V(*[a - b for a, b in zip(self.xs, other.xs)])

    def __mul__(self, k):
        return V(*[a * k for a in self.xs])

    def __rmul__(self, k):
        return V(*[k * a for a in self.xs])

    def __neg__(self):
        return V(*[-a for a in self.xs])

    def __eq__(self, other):
        return isinstance(other, V) and self.xs == other.xs

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return sum(a * a for a in self.xs) < sum(a * a for a in other.xs)

    def __le__(self, other):
        return self == other or self < other

    def __hash__(self):
        return hash(tuple(self.xs))

    def __repr__(self):
        return f'V{tuple(self.xs)}'

    def __len__(self):
        return len(self.xs)

    def __bool__(self):
        return any(a != 0 for a in self.xs)

    def __getitem__(self, i):
        return self.xs[i]

    def __setitem__(self, i, v):
        self.xs[i] = v

    def __contains__(self, v):
        return v in self.xs

    def __iter__(self):
        return iter(self.xs)


a = V(1, 2, 3)
b = V(4, 5, 6)
print(a + b)
print(b - a)
print(a * 10)
print(10 * a)
print(-a)
print(a == V(1, 2, 3))
print(a == b)
print(a != b)
print(a < b)
print(a <= V(1, 2, 3))
print(repr(a))
print(len(a))
print(bool(a))
print(bool(V(0, 0, 0)))
print(a[1])
a[1] = 99
print(a)
print(2 in a)
print(99 in a)
print(7 in a)
print({a, V(1, 99, 3), b})
print(list(a))


for x in V(10, 20, 30):
    print('iter', x)


s = 0
for x in a:
    s += x
print('sum', s)


print([x * 2 for x in V(1, 2, 3)])


class CountUp:
    def __init__(self, n):
        self.n = n
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


cu = CountUp(4)
print(list(cu))


for v in CountUp(3):
    print('cu', v)


for v in CountUp(5):
    if v == 3:
        break
    print('break test', v)


def find_in_iter(it, target):
    for v in it:
        if v == target:
            return v
    return None


print(find_in_iter(CountUp(10), 7))
print(find_in_iter(CountUp(3), 99))


class Range3:
    def __init__(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= 3:
            raise StopIteration
        v = self.i
        self.i += 1
        return v


for v in Range3():
    print('range3', v)


class Dual:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __iter__(self):
        return iter((self.a, self.b))


x, y = Dual(11, 22)
print(x, y)


class Box:
    def __init__(self, v):
        self.v = v

    def __enter__(self):
        print('enter', self.v)
        return self.v

    def __exit__(self, *exc):
        print('exit', self.v)
        return False


bx = Box('thing')
g = bx.__enter__()
print(g)
bx.__exit__(None, None, None)


class Callable:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x, y=1):
        return x * self.factor + y


c = Callable(10)
print(c(5))
print(c(5, 3))
print(c(x=2, y=2))


class Indexable:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return ('slice', key.start, key.stop, key.step)
        return ('index', key)


idx = Indexable()
print(idx[5])
print(idx[1:10:2])
print(idx['key'])
