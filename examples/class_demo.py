"""Class system demo: descriptor, dataclass, NamedTuple, __init_subclass__,
generic class via PEP 695, metaclass."""

from dataclasses import dataclass, field
from typing import NamedTuple


class Validated:
    """Descriptor that validates writes against a predicate."""

    def __init__(self, predicate, msg):
        self.predicate = predicate
        self.msg = msg

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not self.predicate(value):
            raise ValueError(f"{self.name}: {self.msg}, got {value!r}")
        obj.__dict__[self.name] = value


class Point2D(NamedTuple):
    x: float
    y: float

    def magnitude(self):
        return (self.x * self.x + self.y * self.y) ** 0.5


@dataclass(order=True)
class Item:
    priority: int
    name: str = field(compare=False)


# PEP 487: __init_subclass__ collects subclasses into a registry.
class Animal:
    _registry: dict = {}

    def __init_subclass__(cls, kind, **kw):
        super().__init_subclass__(**kw)
        cls.kind = kind
        Animal._registry[kind] = cls


class Dog(Animal, kind="dog"):
    sound = "woof"


class Cat(Animal, kind="cat"):
    sound = "meow"


# PEP 695 generic
class Box[T]:
    def __init__(self, value: T):
        self.value: T = value

    def __repr__(self):
        return f"Box({self.value!r})"


# Object with descriptor-validated attributes.
class Account:
    balance = Validated(lambda v: v >= 0, "balance must be non-negative")
    name = Validated(lambda v: isinstance(v, str) and v, "name must be non-empty")

    def __init__(self, name, balance):
        self.name = name
        self.balance = balance


# Demonstrate everything.
p = Point2D(3.0, 4.0)
print("point:", p, "magnitude:", p.magnitude())

items = sorted([Item(2, "milk"), Item(1, "bread"), Item(3, "egg")])
print("items:", [(it.priority, it.name) for it in items])

print("registry:", sorted(Animal._registry.keys()))
print("dog says:", Dog.sound, "kind:", Dog.kind)

b: Box[int] = Box(42)
print("box:", b, "value:", b.value)

a = Account("alice", 100)
print("account:", a.name, a.balance)
try:
    a.balance = -10
except ValueError as e:
    print("rejected:", e)
