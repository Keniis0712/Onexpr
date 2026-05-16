def make_counter():
    count = 0

    def inc():
        nonlocal_emulation = count
        return nonlocal_emulation + 1

    return inc


def make_adder(x):
    def add(y):
        return x + y

    return add


def make_multi(x):
    def a():
        return x

    def b(z):
        return x + z

    return a, b


add5 = make_adder(5)
add10 = make_adder(10)
print(add5(3))
print(add10(3))
print(add5(100))

a1, b1 = make_multi(7)
print(a1())
print(b1(2))


def deeply_nested(n):
    def lvl1():
        def lvl2():
            def lvl3():
                return n * 10
            return lvl3
        return lvl2
    return lvl1


print(deeply_nested(4)()()())


def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)


print(fact(0))
print(fact(1))
print(fact(5))
print(fact(8))


def even(n):
    if n == 0:
        return True
    return odd(n - 1)


def odd(n):
    if n == 0:
        return False
    return even(n - 1)


print(even(10))
print(odd(10))
print(even(7))
print(odd(7))


def returns_func():
    def inner():
        return 'inner result'
    return inner


print(returns_func()())


def returns_lambda(x):
    return lambda y: x * y


print(returns_lambda(3)(4))
print(returns_lambda(7)(0))
