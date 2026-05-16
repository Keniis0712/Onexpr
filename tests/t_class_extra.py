class Outer:
    class Inner:
        x = 10

        def hello(self):
            return 'inner hello'

    def make(self):
        return Outer.Inner()


o = Outer()
print(Outer.Inner.x)
print(o.make().hello())


def make_class(name):
    class C:
        kind = name

        def describe(self):
            return f'I am {self.kind}'

    return C


Cat = make_class('cat')
Dog = make_class('dog')
print(Cat().describe())
print(Dog().describe())
print(Cat is Dog)


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self, step=1):
        self.n += step
        return self.n


c = Counter()
print(c())
print(c())
print(c(5))
print(c.n)


class Box:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class NamedBox(Box):
    def __init__(self, name, v):
        super().__init__(v)
        self.name = name

    def describe(self):
        return f'{self.name}={self.get()}'


nb = NamedBox('a', 100)
print(nb.describe())
nb.set(200)
print(nb.describe())


class WithProperty:
    def __init__(self):
        self._x = 0

    @property
    def x(self):
        return self._x * 2

    @x.setter
    def x(self, value):
        self._x = value + 1


wp = WithProperty()
print(wp.x)
wp.x = 5
print(wp.x)
print(wp._x)


class HasClassMethod:
    counter = 0

    @classmethod
    def bump(cls):
        cls.counter += 1
        return cls.counter

    @staticmethod
    def static_thing(x):
        return x * 3


print(HasClassMethod.bump())
print(HasClassMethod.bump())
print(HasClassMethod.bump())
print(HasClassMethod.static_thing(7))


class A:
    def kind(self):
        return 'A'


class B:
    def kind(self):
        return 'B'


class AB(A, B):
    pass


class BA(B, A):
    pass


print(AB().kind())
print(BA().kind())


class Defines:
    SIZE = 16
    HALF = SIZE // 2
    NAME = 'Defines'
    TAG = NAME + '_v1'


print(Defines.SIZE)
print(Defines.HALF)
print(Defines.NAME)
print(Defines.TAG)
