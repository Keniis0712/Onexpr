# Module-level annotations
x: int = 10
y: str = 'hello'
z: float

print('Module __annotations__:', __annotations__)
print('x =', x, ', y =', y)


# Function annotations
def greet(name: str, age: int = 0) -> str:
    return f'{name} is {age}'


print('greet.__annotations__:', greet.__annotations__)
print(greet('Alice', 30))


# Class annotations
class Person:
    name: str
    age: int = 0

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


print('Person.__annotations__:', Person.__annotations__)
p = Person('Bob', 25)
print(f'{p.name} is {p.age}')


# Function with *args, **kwargs annotations
def process(*items: int, flag: bool = False, **opts: str) -> list:
    return list(items)


print('process.__annotations__:', process.__annotations__)
print(process(1, 2, 3, flag=True, x='a'))


# Nested function annotations
def outer(x: int) -> int:
    def inner(y: int) -> int:
        return x + y
    return inner(10)


print('outer.__annotations__:', outer.__annotations__)
print(outer(5))


# Class with methods
class Calculator:
    value: int = 0

    def add(self, x: int) -> int:
        self.value += x
        return self.value

    def get(self) -> int:
        return self.value


print('Calculator.__annotations__:', Calculator.__annotations__)
print('Calculator.add.__annotations__:', Calculator.add.__annotations__)
print('Calculator.get.__annotations__:', Calculator.get.__annotations__)
c = Calculator()
c.add(5)
c.add(3)
print('calc value:', c.get())
