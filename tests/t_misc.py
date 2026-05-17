def returns_in_listcomp_filter():
    def helper():
        return [x for x in range(5) if x % 2 == 0]
    return helper()


print(returns_in_listcomp_filter())


def use_walrus_in_if(xs):
    out = []
    for x in xs:
        if (squared := x * x) > 10:
            out.append(squared)
    return out


print(use_walrus_in_if([1, 2, 3, 4, 5]))


def assign_via_walrus_in_or():
    if (n := 5) and (m := n * 2):
        return n + m
    return -1


print(assign_via_walrus_in_or())


def conditional_call(x):
    return (lambda v: v + 1)(x) if x > 0 else (lambda v: -v)(x)


print(conditional_call(5))
print(conditional_call(-3))


def multi_return_paths(x):
    if x < 0:
        return 'neg'
    if x == 0:
        return 'zero'
    if x < 10:
        return 'small'
    if x < 100:
        return 'medium'
    return 'large'


for v in [-5, 0, 5, 50, 500]:
    print(multi_return_paths(v))


class Box:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def double(self):
        return Box(self.v * 2)

    def add(self, other):
        return Box(self.get() + other.get())


b = Box(10).double().add(Box(3)).double()
print(b.get())


def nested_with_for_returning():
    def search(grid, target):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == target:
                    return (i, j)
        return None
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return search(g, 5)


print(nested_with_for_returning())


def returns_func_with_capture(threshold):
    def check(x):
        if x > threshold:
            return 'big'
        if x < -threshold:
            return 'big neg'
        return 'small'
    return check


c5 = returns_func_with_capture(5)
print(c5(10))
print(c5(0))
print(c5(-10))


def captures_loop_var_with_return():
    funcs = []
    for i in range(3):
        def maker(j):
            def inner():
                return j * 10
            return inner
        funcs.append(maker(i))
    return [f() for f in funcs]


print(captures_loop_var_with_return())


def returns_dict_inside_for():
    out = {}
    for k, v in [('a', 1), ('b', 2), ('c', 3)]:
        out[k] = v
    return out


print(sorted(returns_dict_inside_for().items()))


class Stateful:
    def __init__(self):
        self.log = []

    def step(self, x):
        self.log.append(x)
        return self

    def end(self):
        return list(self.log)


s = Stateful().step(1).step(2).step(3).end()
print(s)


def func_with_string_default(name='anonymous'):
    return f'hello, {name}'


print(func_with_string_default())
print(func_with_string_default('alice'))


def func_with_none_default(x=None):
    if x is None:
        return 'no x'
    return f'got {x}'


print(func_with_none_default())
print(func_with_none_default(0))
print(func_with_none_default(False))


def returns_generator_or_list(gen=False):
    if gen:
        return (x * 2 for x in range(3))
    return [x * 2 for x in range(3)]


print(list(returns_generator_or_list()))
print(list(returns_generator_or_list(True)))


def conditional_loop_break(xs, max_iter):
    count = 0
    for x in xs:
        count += 1
        if count >= max_iter:
            return ('iter limit', count)
        if x < 0:
            return ('saw neg', x, count)
    return ('done', count)


print(conditional_loop_break([1, 2, 3, 4, 5], 3))
print(conditional_loop_break([1, 2, -3, 4], 10))
print(conditional_loop_break([1, 2, 3], 10))


# Regression: __name__ / __qualname__ on generator/coroutine
# forwarders. The lambda forwarder used to expose '<lambda>'.
def gen():
    yield 1


print(gen.__name__, gen.__qualname__)


async def coro():
    return 1


print(coro.__name__)


import inspect

print(inspect.isgeneratorfunction(gen))
print(inspect.iscoroutinefunction(coro))


# Regression: inspect.isasyncgenfunction works for our async generator
# forwarders via the inspect._has_code_flag monkey-patch.
async def ag_for_inspect():
    yield 1


print(inspect.isasyncgenfunction(ag_for_inspect))
print(inspect.isasyncgenfunction(coro))   # async non-generator: False
print(inspect.isasyncgenfunction(gen))    # plain generator: False
