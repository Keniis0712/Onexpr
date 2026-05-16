temp_0 = 'user_var'
print(temp_0)


def with_temp_var_arg(temp_1, temp_2):
    return temp_1 + temp_2


print(with_temp_var_arg(10, 20))


class HasTempField:
    def __init__(self):
        self.temp_5 = 'field'

    def use(self):
        temp_3 = self.temp_5
        return temp_3


print(HasTempField().use())


def shadowing(x):
    print('outer x', x)
    x = x + 100
    print('inner x', x)
    return x


print(shadowing(5))


x = 'global x'


def reads_global():
    return x


print(reads_global())


def shadow_with_local():
    x = 'local'
    return x


print(shadow_with_local())
print(x)


def returns_locals_view():
    a = 1
    b = 2
    return (a, b)


print(returns_locals_view())


def call_default():
    print('default evaluated')
    return [1]


def with_mutable_default(items=call_default()):
    items.append(99)
    return items


print(with_mutable_default())
print(with_mutable_default())
print(with_mutable_default([5]))
print(with_mutable_default())


seq = []


def with_dynamic_default(x, y=len(seq)):
    return (x, y)


seq.append(0)
print(with_dynamic_default(1))
seq.append(0)
print(with_dynamic_default(1))


def make_closures(n):
    fns = []
    for i in range(n):
        fns.append(lambda x, i=i: x * 10 + i)
    return fns


for f in make_closures(3):
    print(f(7))


def outer_var():
    x = 1

    def inner():
        return x
    return inner


f = outer_var()
print(f())


def branch_returns(x):
    if x == 1:
        return 'one'
    elif x == 2:
        return 'two'
    elif x == 3:
        return 'three'
    elif x == 4:
        return 'four'
    else:
        return 'other'


for v in [1, 2, 3, 4, 5]:
    print(branch_returns(v))


def returns_via_nested_if(x):
    if x > 0:
        if x < 10:
            if x > 5:
                return 'mid-high'
            else:
                return 'mid-low'
        else:
            return 'high'
    return 'non-positive'


for v in [-1, 3, 7, 100]:
    print(returns_via_nested_if(v))


class Outer:
    class Inner:
        @staticmethod
        def foo():
            return 'inner foo'


class Sub(Outer.Inner):
    def foo(self):
        base = super().foo()
        return base + ' / sub'


print(Sub().foo())


class A:
    def m(self):
        return 'A.m'


class B(A):
    def m(self):
        return super().m() + '/B'


class C(B):
    def m(self):
        return super().m() + '/C'


print(C().m())


def chained_assignment():
    a = b = c = []
    a.append(1)
    return (a is b, b is c, a, b, c)


print(chained_assignment())


def starred_unpack():
    a, *b, c = [1, 2, 3, 4, 5]
    return (a, b, c)


print(starred_unpack())


def starred_at_start():
    *a, b = [1, 2, 3, 4]
    return (a, b)


print(starred_at_start())


def starred_at_end():
    a, *b = [1, 2, 3, 4]
    return (a, b)


print(starred_at_end())


def swap_in_function(a, b):
    a, b = b, a
    return (a, b)


print(swap_in_function(1, 'two'))


def use_string_methods(s):
    return [s.upper(), s.lower(), s.title(), s.capitalize(), s.swapcase()]


print(use_string_methods('Hello World'))


def use_list_methods():
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    a.sort()
    a.reverse()
    a.append(99)
    a.insert(0, -1)
    a.remove(1)
    last = a.pop()
    return (a, last)


print(use_list_methods())


def chained_method_calls(s):
    return s.strip().upper().replace('A', '@').split('-')


print(chained_method_calls('  abc-def-ghi  '))


def use_dict_methods():
    d = {'a': 1, 'b': 2}
    d.update({'c': 3})
    d.setdefault('a', 99)
    d.setdefault('d', 4)
    keys = sorted(d.keys())
    vals = sorted(d.values())
    items = sorted(d.items())
    return (keys, vals, items)


print(use_dict_methods())
