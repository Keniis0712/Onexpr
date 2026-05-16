def trace(label):
    def deco(func):
        def wrapper(*args, **kwargs):
            print(f'[{label}] enter')
            r = func(*args, **kwargs)
            print(f'[{label}] exit')
            return r
        wrapper.__name__ = func.__name__
        return wrapper
    return deco


@trace('outer')
@trace('inner')
def hello(name):
    print(f'hi, {name}')
    return len(name)


print(hello('world'))
print(hello.__name__)


def memoize(f):
    cache = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return wrapper


@memoize
def slow_square(x):
    print(f'computing {x}')
    return x * x


print(slow_square(3))
print(slow_square(3))
print(slow_square(4))
print(slow_square(3))


calls = []


def log_call(f):
    def wrapper(*args, **kwargs):
        calls.append((f.__name__, args, tuple(sorted(kwargs.items()))))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@log_call
def add(a, b):
    return a + b


@log_call
def mul(a, b):
    return a * b


add(1, 2)
mul(3, 4)
add(b=10, a=20)
print(calls)


def with_pre_post(pre, post):
    def deco(f):
        def wrapper(*args, **kwargs):
            pre()
            try_value = f(*args, **kwargs)
            post()
            return try_value
        return wrapper
    return deco


@with_pre_post(lambda: print('pre'), lambda: print('post'))
def doit():
    print('doing')


doit()


def double_return(f):
    def wrapper(*args, **kwargs):
        r = f(*args, **kwargs)
        return (r, r)
    return wrapper


@double_return
def get_n():
    return 42


print(get_n())


def class_decorator(cls):
    cls.tagged = True
    return cls


@class_decorator
class Tagged:
    pass


print(Tagged.tagged)


def add_method(method_name):
    def deco(cls):
        def fn(self):
            return f'method {method_name} on {type(self).__name__}'
        setattr(cls, method_name, fn)
        return cls
    return deco


@add_method('greet')
@add_method('wave')
class Person:
    pass


p = Person()
print(p.greet())
print(p.wave())
