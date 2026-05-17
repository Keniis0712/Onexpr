class Meta(type):
    def __new__(cls, name, bases, namespace):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, namespace)

def my_decorator(method):
    def wrapper(self, *args, **kwargs):
        print(f"Before {method.__name__}")
        result = method(self, *args, **kwargs)
        print("After")
        return result

    return wrapper


class Parent(metaclass=Meta):

    __slots__ = ['name']
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Parent: {self.name}")


class Child(Parent):
    __slots__ = ['name', 'age']

    @my_decorator
    def greet(self):
        print(f"Child: {self.name}")

    @property
    def info(self):
        return f"{self.name}, {self.age}"

    @info.setter
    def info(self, value):
        self.name, self.age = value.split(',')

    @classmethod
    def create_default(cls):
        return cls("Default")

    @staticmethod
    def helper():
        print("Static help")


child1 = Child.create_default()
Child.helper()
child1.age = 10
child1.greet()
child1.info = "Alice,15"
print(child1.info)

# Regression: walrus inside a class body produces a class attribute.
class WalrusBody:
    a = (b := 10)
    c = b * 2


print(WalrusBody.a, WalrusBody.b, WalrusBody.c)


# Regression: `class C(Base, kw=v)` forwards kw to metaclass +
# __init_subclass__. PEP 487.
class _PEP487Base:
    def __init_subclass__(cls, **kw):
        cls.kw = kw


class _PEP487Child(_PEP487Base, foo='bar', baz=42):
    pass


print(_PEP487Child.kw)


# Regression: PEP 560 — `class K(typing.NamedTuple): ...` and
# `class K(Generic[T]): ...` use bases that aren't classes
# themselves; they expose `__mro_entries__`. _make_class now
# resolves those into the actual base classes and stashes the
# original subscripted forms on `__orig_bases__` so the metaclass
# / __init_subclass__ can read them. Also injects __module__ into
# the namespace dict because typing.NamedTupleMeta.__new__ requires
# it.
from typing import NamedTuple as _NT


class _Point(_NT):
    x: int
    y: int


_p = _Point(1, 2)
print(_p, _p.x, _p.y, _p[0])


from typing import Generic as _Generic, TypeVar as _TypeVar

_T = _TypeVar('_T')


class _Container(_Generic[_T]):
    def __init__(self, v):
        self.v = v


_c = _Container[int](42)
print(_c.v)
