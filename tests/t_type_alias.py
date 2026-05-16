type X = int

# onexpr's simplified type alias: `type X = int` becomes `X = int`
# (not a TypeAliasType). We test that the alias is usable.
x: X = 10
print(x)


type Vector = list[float]

v: Vector = [1.0, 2.0]
print(v)


type IntList = list[int]
v2: IntList = [1, 2, 3]
print(v2)


type Pair[T] = tuple[T, T]

# Type params become typing.TypeVar at runtime in onexpr's transform.
p: Pair = (1, 1)
print(p)


type Result[T, E] = T | E

# Union types work at runtime for isinstance in 3.10+.
print(isinstance(1, int | str))


type Number = int | float


def is_number(x):
    return isinstance(x, (int, float))


print(is_number(1))
print(is_number(2.5))
print(is_number('hi'))


# Used as type hint in a function (not at runtime, but onexpr should
# at least parse and run the runtime aliasing without crashing).
type Greeter = str


def greet(name: Greeter) -> str:
    return f'hello, {name}'


print(greet('world'))


# Inside a class
class Container:
    type Item = int

    def __init__(self):
        self.items = []

    def add(self, x):
        self.items.append(x)


c = Container()
c.add(1)
c.add(2)
print(c.items)
