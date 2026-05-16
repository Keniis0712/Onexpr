import typing


# 1) Plain alias: round-trip prints type, value, type_params, isinstance.
type X = list[int]
print(type(X).__name__ in ('TypeAliasType', '_LazyAlias'))
print(X.__value__)
print(X.__type_params__)
print(isinstance(X, typing.TypeAliasType))


# 2) Generic alias.
type Pair[T] = tuple[T, T]
print(Pair.__type_params__[0].__class__.__name__)
print(Pair.__value__)
print(isinstance(Pair, typing.TypeAliasType))


# 3) Type vars do NOT leak into the surrounding scope.
type Inner[Q] = list[Q]
print('Q' in dir())


# 4) Subscripting an alias.
type Box[T] = list[T]
print(Box[int])


# 5) Multiple type vars.
type Result[T, E] = T | E
print(len(Result.__type_params__))
print(isinstance(1, int | str))


# 6) Used as a type hint at runtime — lookup must not crash.
type Greeter = str


def greet(name: Greeter) -> str:
    return f'hello, {name}'


print(greet('world'))


# 7) Inside a class.
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


# 8) Lazy evaluation: RHS is not evaluated until __value__ is read.
type Lazy = Undefined  # noqa: F821 — name doesn't exist yet
try:
    Lazy.__value__
except NameError:
    print('lazy ok')


# 9) ParamSpec.
type Cb[**P] = typing.Callable[P, int]
print(type(Cb.__type_params__[0]).__name__)


# 10) TypeVarTuple.
type Tup[*Ts] = tuple[*Ts]
print(type(Tup.__type_params__[0]).__name__)


# 11) TypeVar with bound.
type Bounded[T: int] = list[T]
print(Bounded.__type_params__[0].__bound__)


# 12) TypeVar with constraints.
type Constrained[T: (int, str)] = list[T]
print(Constrained.__type_params__[0].__constraints__)
