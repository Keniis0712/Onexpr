def three_levels_all_participate():
    a = 'level0'

    def lvl1():
        b = 'level1'

        def lvl2():
            nonlocal a, b
            a = 'a-changed'
            b = 'b-changed'
        lvl2()
        return b
    inner_b = lvl1()
    return (a, inner_b)


print(three_levels_all_participate())


def two_outers_two_inners():
    x = 'outer1'

    def fa():
        nonlocal x
        x = 'fa wrote'

    def fb():
        nonlocal x
        x = 'fb wrote'
    fa()
    a_seen = x
    fb()
    b_seen = x
    fa()
    a2_seen = x
    return (a_seen, b_seen, a2_seen)


print(two_outers_two_inners())


def recursive_with_nonlocal(n):
    depth = 0
    max_depth = 0

    def recurse(remaining):
        nonlocal depth, max_depth
        depth += 1
        if depth > max_depth:
            max_depth = depth
        if remaining > 0:
            recurse(remaining - 1)
        depth -= 1

    recurse(n)
    return (depth, max_depth)


print(recursive_with_nonlocal(5))
print(recursive_with_nonlocal(10))


def mutate_via_method_no_nonlocal_needed():
    # Mutating a list via method calls doesn't need nonlocal — only
    # rebinding does. Make sure transformer doesn't break this case.
    items = []

    def add(x):
        items.append(x)
    add(1)
    add(2)
    add(3)
    return items


print(mutate_via_method_no_nonlocal_needed())


def closure_over_dict_inner_modifies_keys():
    state = {'count': 0, 'log': []}

    def step(label):
        nonlocal state
        state['count'] += 1
        state['log'].append(label)

    step('a')
    step('b')
    step('c')
    return state


print(closure_over_dict_inner_modifies_keys())


def returned_inner_keeps_box_alive():
    counter = 0

    def make_inc():
        def inc():
            nonlocal counter
            counter += 1
            return counter
        return inc
    return make_inc()


inc = returned_inner_keeps_box_alive()
print(inc())
print(inc())
print(inc())


def two_independent_boxes():
    def factory(start):
        n = start

        def inc():
            nonlocal n
            n += 1
            return n
        return inc
    a = factory(0)
    b = factory(100)
    return (a(), a(), b(), a(), b())


print(two_independent_boxes())


def comprehension_inside_inner_referencing_nonlocal():
    multiplier = 10

    def collect(xs):
        nonlocal multiplier
        result = [x * multiplier for x in xs]
        multiplier += 1
        return result
    a = collect([1, 2, 3])
    b = collect([1, 2, 3])
    c = collect([1, 2, 3])
    return (a, b, c, multiplier)


print(comprehension_inside_inner_referencing_nonlocal())


def lambda_captures_nonlocal_via_inner_func():
    seed = 5

    def make_adder():
        return lambda x: x + seed
    f1 = make_adder()
    print(f1(10))

    def bump():
        nonlocal seed
        seed = 100
    bump()
    print(f1(10))


lambda_captures_nonlocal_via_inner_func()


def nonlocal_used_inside_for_loop():
    total = 0

    def process(values):
        nonlocal total
        for v in values:
            total = total + v
        return total
    a = process([1, 2, 3])
    b = process([10, 20])
    return (a, b, total)


print(nonlocal_used_inside_for_loop())


def deeply_nested_method_class_function():
    def make_holder():
        value = 'initial'

        class Holder:
            def get(self):
                return value

            def set(self, v):
                nonlocal value
                value = v

            def update_via_method(self, fn):
                nonlocal value
                value = fn(value)
        return Holder()
    h = make_holder()
    print(h.get())
    h.set('changed')
    print(h.get())
    h.update_via_method(lambda v: v + '!')
    print(h.get())


deeply_nested_method_class_function()


def class_method_modifies_outer_function_local():
    state = {'count': 0}

    class Bumper:
        @staticmethod
        def bump():
            nonlocal state
            state = {'count': state['count'] + 1}
    Bumper.bump()
    Bumper.bump()
    Bumper.bump()
    return state


print(class_method_modifies_outer_function_local())


def mutual_inner_share_box():
    history = []

    def push(x):
        nonlocal history
        history = history + [x]

    def pop():
        nonlocal history
        if history:
            x = history[-1]
            history = history[:-1]
            return x
        return None

    push('a')
    push('b')
    push('c')
    a = pop()
    push('d')
    b = pop()
    c = pop()
    return (a, b, c, history)


print(mutual_inner_share_box())


def two_levels_each_owns_different_name():
    outer_var = 'outer'

    def middle():
        middle_var = 'middle'

        def inner():
            nonlocal outer_var, middle_var
            outer_var = 'outer changed'
            middle_var = 'middle changed'
        inner()
        return middle_var
    middle_result = middle()
    return (outer_var, middle_result)


print(two_levels_each_owns_different_name())


def nonlocal_var_initialized_via_inner_call():
    state = None

    def initialize():
        nonlocal state
        state = []

    def add(x):
        state.append(x)
    initialize()
    add(1)
    add(2)
    add(3)
    return state


print(nonlocal_var_initialized_via_inner_call())


def nonlocal_with_default_param_capturing_it():
    base = 100

    def make_with_default():
        def inner(x=base):
            return x
        return inner
    f = make_with_default()
    print(f())

    def update():
        nonlocal base
        base = 999
    update()
    print(f())


nonlocal_with_default_param_capturing_it()


def chained_nonlocal_through_three_levels():
    log = []

    def first():
        def second():
            def third():
                nonlocal log
                log = log + ['third']
            third()
        second()

    first()
    print(log)
    first()
    print(log)
    first()
    print(log)


chained_nonlocal_through_three_levels()


def nonlocal_in_branch_only():
    state = 'unset'

    def maybe_set(cond):
        if cond:
            nonlocal state
            state = 'set'
    maybe_set(False)
    a = state
    maybe_set(True)
    b = state
    return (a, b)


print(nonlocal_in_branch_only())


def nonlocal_writes_in_loop():
    last_seen = None

    def consume(items):
        nonlocal last_seen
        for item in items:
            last_seen = item

    consume([])
    a = last_seen
    consume(['a', 'b', 'c'])
    b = last_seen
    consume([42])
    c = last_seen
    return (a, b, c)


print(nonlocal_writes_in_loop())


def names_that_clash_with_helper_fields():
    # `value` and `returned` are the names _FuncHelper uses internally.
    # Boxing nonlocals shouldn't put them straight on the helper as
    # `value` / `returned` — the return machinery would overwrite them.
    value = 'user-value'
    returned = 'user-returned'

    def update():
        nonlocal value, returned
        value = 'new-value'
        returned = 'new-returned'
    update()
    return (value, returned)


print(names_that_clash_with_helper_fields())


def value_via_class_method():
    value = 'init'

    class Holder:
        def get(self):
            return value

        def set(self, v):
            nonlocal value
            value = v
    h = Holder()
    print(h.get())
    h.set('changed')
    print(h.get())
    h.set('again')
    print(h.get())


value_via_class_method()
