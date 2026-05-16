def basic_leak():
    for i in range(5):
        pass
    return i


print(basic_leak())


def leak_after_break():
    for i in range(10):
        if i >= 4:
            break
    return i


print(leak_after_break())


def leak_after_break_with_body():
    last = -1
    for i in range(10):
        last = i
        if i >= 3:
            break
    return (i, last)


print(leak_after_break_with_body())


def empty_iter_does_not_bind():
    i = 'pre'
    for i in []:
        pass
    return i


print(empty_iter_does_not_bind())


def truthy_falsy_values_dont_short_circuit():
    seen = []
    for x in [10, 0, 20, '', 30]:
        seen.append(x)
    return (x, seen)


print(truthy_falsy_values_dont_short_circuit())


def nested_loops_each_leaks():
    for i in range(3):
        for j in range(2):
            pass
    return (i, j)


print(nested_loops_each_leaks())


def nested_with_outer_unread():
    counts = []
    for i in range(3):
        for j in range(4):
            counts.append((i, j))
    return j


print(nested_with_outer_unread())


def truthy_iter_then_outer_uses_var():
    log = []
    for x in 'abc':
        log.append(x)
    log.append(('after', x))
    return log


print(truthy_iter_then_outer_uses_var())


def for_then_for_then_read():
    for i in range(2):
        pass
    for i in range(5, 8):
        pass
    return i


print(for_then_for_then_read())


def for_inside_function_leaks_locally():
    def inner():
        for i in range(4):
            pass
        return i
    return inner()


print(for_inside_function_leaks_locally())


def for_with_truthy_return_inside():
    log = []
    for i in range(5):
        log.append(i)
        if i == 2:
            break
    log.append(('saw', i))
    return log


print(for_with_truthy_return_inside())


def loop_with_object_target_then_use():
    class Item:
        def __init__(self, v):
            self.v = v
    items = [Item(1), Item(2), Item(3)]
    for it in items:
        pass
    return it.v


print(loop_with_object_target_then_use())


def deeply_nested_returns_then_leak():
    def inner():
        for i in range(5):
            if i == 3:
                break
        return i
    return inner()


print(deeply_nested_returns_then_leak())


def truthy_leak_does_not_kill_caller():
    def populate():
        out = []
        for v in [10, 20, 30]:
            out.append(v)
        return out
    print('calling populate')
    r = populate()
    print('got', r)
    return r


truthy_leak_does_not_kill_caller()
