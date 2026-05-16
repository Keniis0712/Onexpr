x: int = 5
print(x)


y: str
y = 'hello'
print(y)


z: list[int] = [1, 2, 3]
print(z)


a: int
b: str
c: float
a = 1
b = 'b'
c = 3.14
print(a, b, c)


def func_with_anno(x: int, y: str = 'default') -> str:
    result: str = f'{x}-{y}'
    return result


print(func_with_anno(1))
print(func_with_anno(2, 'custom'))


class WithAnnos:
    name: str
    count: int = 0

    def __init__(self, name: str):
        self.name = name
        self.count: int = 1

    def bump(self) -> int:
        self.count += 1
        return self.count


w = WithAnnos('item')
print(w.name)
print(w.count)
print(w.bump())
print(w.bump())


def loop_with_anno():
    total: int = 0
    for i in range(5):
        total += i
    return total


print(loop_with_anno())


d: dict[str, int] = {}
d['a'] = 1
d['b']: int = 2
print(sorted(d.items()))


nums: list = []
for i in range(3):
    nums.append(i * 10)
print(nums)


def mutates(value: int) -> tuple:
    inner: int = value * 2
    pair: tuple = (value, inner)
    return pair


print(mutates(7))


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __repr__(self) -> str:
        return f'Point({self.x}, {self.y})'


p = Point(1.5, 2.5)
print(p)
print(p.x, p.y)
