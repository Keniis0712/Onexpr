counter = 0


def bump():
    global counter
    counter += 1
    return counter


print(bump())
print(bump())
print(bump())
print(counter)


name = 'init'
items = []


def update_both(new_name, new_item):
    global name, items
    name = new_name
    items.append(new_item)


update_both('alice', 'a')
update_both('bob', 'b')
print(name)
print(items)


def shadow_without_global():
    name = 'inner'
    return name


print(shadow_without_global())
print(name)


flag = False


def flip():
    global flag
    flag = not flag


flip()
print(flag)
flip()
print(flag)


total = 0


def add_to_total(*xs):
    global total
    for x in xs:
        total = total + x
    return total


print(add_to_total(1, 2, 3))
print(add_to_total(10, 20))
print(total)


score = 100


def reset_and_set(v):
    global score
    score = v


reset_and_set(42)
print(score)
reset_and_set(7)
print(score)


nested_calls = 0


def outer_caller():
    def inner_caller():
        global nested_calls
        nested_calls += 1
        return nested_calls
    return inner_caller()


print(outer_caller())
print(outer_caller())
print(outer_caller())
print(nested_calls)


def inner_can_redeclare():
    val = 'outer'

    def inner():
        global val
        val = 'set by inner'
    inner()
    return val


print(inner_can_redeclare())
print(val)


big = 0


def assigns_via_branches(n):
    global big
    if n > 10:
        big = 'big'
    elif n > 0:
        big = 'small'
    else:
        big = 'zero or neg'


assigns_via_branches(50)
print(big)
assigns_via_branches(5)
print(big)
assigns_via_branches(-1)
print(big)


visits = []


def visit(label):
    global visits
    visits = visits + [label]


visit('a')
visit('b')
visit('c')
print(visits)


def reads_global_without_declaration():
    return counter


print(reads_global_without_declaration())


registry = {}


def register(key, value):
    global registry
    registry[key] = value


register('x', 1)
register('y', 2)
register('x', 99)
print(sorted(registry.items()))


depth = 0


def three_levels():
    def middle():
        def inner():
            global depth
            depth += 1
        return inner
    return middle()


f = three_levels()
f()
f()
f()
print(depth)


flag2 = False


def toggle_via_inner():
    def helper():
        global flag2
        flag2 = not flag2
    helper()


toggle_via_inner()
print(flag2)
toggle_via_inner()
print(flag2)


outer_var_for_global_inner = []


def no_global_in_outer_but_inner_has():
    def inner():
        global outer_var_for_global_inner
        outer_var_for_global_inner = 'set by inner'
    inner()


no_global_in_outer_but_inner_has()
print(outer_var_for_global_inner)
