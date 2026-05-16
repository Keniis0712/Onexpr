def basic_counter():
    n = 0

    def inc():
        nonlocal n
        n = n + 1
        return n
    return inc, lambda: n


inc, peek = basic_counter()
print(inc())
print(inc())
print(inc())
print(peek())


def two_nonlocals():
    a = 1
    b = 2

    def update():
        nonlocal a, b
        a = a + 10
        b = b + 100
    update()
    return (a, b)


print(two_nonlocals())


def deeper_nesting():
    x = 'init'

    def middle():
        def inner():
            nonlocal x
            x = 'set by inner'
        inner()
    middle()
    return x


print(deeper_nesting())


def middle_owns_not_outer():
    x = 'outer'

    def middle():
        x = 'middle'

        def inner():
            nonlocal x
            x = 'set by inner'
        inner()
        return x
    middle_result = middle()
    return (middle_result, x)


print(middle_owns_not_outer())


def aug_assign_nonlocal():
    total = 0

    def add(n):
        nonlocal total
        total += n
    add(5)
    add(10)
    add(20)
    return total


print(aug_assign_nonlocal())


def conditional_nonlocal():
    state = 'idle'

    def go(cmd):
        nonlocal state
        if cmd == 'start':
            state = 'running'
        elif cmd == 'stop':
            state = 'stopped'

    go('start')
    s1 = state
    go('stop')
    s2 = state
    return (s1, s2)


print(conditional_nonlocal())


def loop_modifies_nonlocal():
    out = []

    def add(x):
        nonlocal out
        out = out + [x]

    for i in range(4):
        add(i)
    return out


print(loop_modifies_nonlocal())


def multiple_inner_share_nonlocal():
    counter = 100

    def get():
        return counter

    def reset():
        nonlocal counter
        counter = 0

    def bump():
        nonlocal counter
        counter += 1

    bump()
    bump()
    bump()
    a = get()
    reset()
    b = get()
    bump()
    c = get()
    return (a, b, c, counter)


print(multiple_inner_share_nonlocal())


def inner_param_shadows_nonlocal():
    x = 'outer x'

    def inner(x):
        return x + ' (param)'
    return (inner('arg'), x)


print(inner_param_shadows_nonlocal())


def two_levels_inner_nonlocal_to_middle():
    def middle():
        v = 'middle v'

        def inner():
            nonlocal v
            v = 'changed by inner'
        inner()
        return v
    return middle()


print(two_levels_inner_nonlocal_to_middle())


def two_levels_inner_nonlocal_skips_middle():
    v = 'outer v'

    def middle():
        def inner():
            nonlocal v
            v = 'changed via middle'
        inner()
    middle()
    return v


print(two_levels_inner_nonlocal_skips_middle())


def returns_value_via_nonlocal():
    result = None

    def compute(n):
        nonlocal result
        if n == 0:
            result = 'zero'
        else:
            result = ('nonzero', n)

    compute(0)
    a = result
    compute(5)
    b = result
    return (a, b)


print(returns_value_via_nonlocal())


def closure_chain():
    chain = []

    def step(label):
        nonlocal chain

        def commit():
            nonlocal chain
            chain = chain + [label]
        commit()
    step('a')
    step('b')
    step('c')
    return chain


print(closure_chain())


def nonlocal_var_used_in_inner_lambda():
    factor = 10

    def make():
        return lambda x: x * factor
    f = make()
    print(f(3))
    print(f(5))

    def update():
        nonlocal factor
        factor = 100
    update()
    print(f(3))


nonlocal_var_used_in_inner_lambda()


def class_inside_function_with_nonlocal():
    state = 'init'

    class C:
        def get(self):
            return state

        def set(self, v):
            nonlocal state
            state = v
    obj = C()
    a = obj.get()
    obj.set('changed')
    b = obj.get()
    return (a, b, state)


print(class_inside_function_with_nonlocal())
