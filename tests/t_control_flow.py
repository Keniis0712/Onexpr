def break_inside_inner_loop():
    out = []
    for i in range(3):
        for j in range(3):
            if j == 2:
                break
            out.append((i, j))
    return out


print(break_inside_inner_loop())


def continue_inside_inner_loop():
    out = []
    for i in range(3):
        for j in range(3):
            if j == 1:
                continue
            out.append((i, j))
    return out


print(continue_inside_inner_loop())


def break_outer_via_flag():
    out = []
    done = False
    for i in range(5):
        if done:
            break
        for j in range(5):
            if j == 3:
                done = True
                break
            out.append((i, j))
    return out


print(break_outer_via_flag())


def for_else_runs_when_no_break():
    log = []
    for i in range(3):
        log.append(('iter', i))
    else:
        log.append('else')
    return log


print(for_else_runs_when_no_break())


def for_else_skips_after_break():
    log = []
    for i in range(5):
        if i == 2:
            log.append('break')
            break
        log.append(('iter', i))
    else:
        log.append('else')
    return log


print(for_else_skips_after_break())


def for_else_continues_when_only_continue():
    log = []
    for i in range(5):
        if i % 2 == 0:
            continue
        log.append(('odd', i))
    else:
        log.append('else')
    return log


print(for_else_continues_when_only_continue())


def while_else_runs_when_cond_fails():
    log = []
    n = 0
    while n < 3:
        log.append(n)
        n += 1
    else:
        log.append('else')
    return log


print(while_else_runs_when_cond_fails())


def while_else_skips_after_break():
    log = []
    n = 0
    while n < 5:
        if n == 2:
            log.append('break')
            break
        log.append(n)
        n += 1
    else:
        log.append('else')
    return log


print(while_else_skips_after_break())


def break_then_else_in_outer():
    log = []
    for i in range(3):
        for j in range(3):
            if j == 1:
                log.append(('inner break', i, j))
                break
            log.append(('inner', i, j))
        else:
            log.append(('inner else', i))
        log.append(('after inner', i))
    else:
        log.append('outer else')
    return log


print(break_then_else_in_outer())


def return_skips_for_else():
    log = []

    def body():
        for i in range(5):
            if i == 2:
                return 'returned'
            log.append(i)
        else:
            log.append('else')
            return 'else done'
        return 'fell through'
    r = body()
    return (r, log)


print(return_skips_for_else())


def return_skips_while_else():
    log = []

    def body():
        n = 0
        while n < 5:
            if n == 2:
                return 'returned'
            log.append(n)
            n += 1
        else:
            log.append('else')
            return 'else done'
        return 'fell through'
    r = body()
    return (r, log)


print(return_skips_while_else())


def break_in_for_else_clause():
    log = []
    for i in range(2):
        log.append(('outer', i))
        for j in range(3):
            if j == 1:
                break
            log.append(('inner', i, j))
        else:
            log.append(('inner else', i))
            break
        log.append(('after inner', i))
    return log


print(break_in_for_else_clause())


def deep_break_does_not_propagate():
    out = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if k == 2:
                    break
                out.append((i, j, k))
    return out


print(deep_break_does_not_propagate())


def deep_continue_does_not_propagate():
    out = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if k == 1:
                    continue
                out.append((i, j, k))
    return out


print(deep_continue_does_not_propagate())


def break_after_inner_return():
    log = []

    def find(g, t):
        for row in g:
            for cell in row:
                if cell == t:
                    return cell
                log.append(cell)
            log.append('row done')
        return None
    r = find([['a', 'b'], ['c', 'd', 'e'], ['f']], 'd')
    return (r, log)


print(break_after_inner_return())


def break_in_while_inside_for():
    log = []
    for i in range(3):
        n = 0
        while n < 5:
            if n == i:
                break
            n += 1
        log.append((i, n))
    return log


print(break_in_while_inside_for())


def continue_in_while_inside_for():
    log = []
    for i in range(3):
        n = 0
        while n < 4:
            n += 1
            if n == 2:
                continue
            log.append((i, n))
    return log


print(continue_in_while_inside_for())


def return_through_loops_and_branches(matrix, target):
    for row in matrix:
        for x in row:
            if x == target:
                return ('found', x)
            if x < 0:
                return ('neg', x)
    return ('not found', None)


print(return_through_loops_and_branches([[1, 2], [3, -1, 4]], 99))
print(return_through_loops_and_branches([[1, 2], [3, 4, 5]], 4))
print(return_through_loops_and_branches([[1, 2], [3, 4]], 99))


def break_then_return_outer():
    def body():
        result = None
        for i in range(5):
            if i == 3:
                result = i
                break
        if result is None:
            return 'not found'
        return ('found', result)
    return body()


print(break_then_return_outer())


def assignment_via_break(items):
    found_at = -1
    for i, x in enumerate(items):
        if x == 'target':
            found_at = i
            break
    return found_at


print(assignment_via_break(['a', 'b', 'target', 'c']))
print(assignment_via_break(['a', 'b', 'c']))


def while_true_then_break():
    log = []
    n = 0
    while True:
        if n >= 3:
            break
        log.append(n)
        n += 1
    return log


print(while_true_then_break())


def break_while_with_else_when_condition_breaks():
    log = []
    n = 0
    while n < 10:
        log.append(n)
        if n >= 3:
            break
        n += 1
    else:
        log.append('else not reached')
    return log


print(break_while_with_else_when_condition_breaks())


def continue_in_while_resets_logic():
    log = []
    n = 0
    while n < 5:
        n += 1
        if n == 3:
            continue
        log.append(n)
    return log


print(continue_in_while_resets_logic())


def for_with_break_in_nested_if():
    log = []
    for x in range(5):
        if x % 2 == 0:
            if x == 4:
                log.append(('break at', x))
                break
            log.append(('even', x))
        else:
            log.append(('odd', x))
    return log


print(for_with_break_in_nested_if())


def two_breaks_in_one_loop_only_first_runs():
    log = []
    for i in range(5):
        if i == 1:
            log.append('first break')
            break
        if i == 2:
            log.append('second break')
            break
        log.append(i)
    return log


print(two_breaks_in_one_loop_only_first_runs())


def loop_var_visible_after_loop():
    last = -1
    for i in range(3):
        last = i
    return last


print(loop_var_visible_after_loop())


def early_return_inside_three_levels():
    log = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 1 and b == 2 and c == 1:
                    return (a, b, c, log)
                log.append((a, b, c))
    return ('done', log)


r = early_return_inside_three_levels()
print(r[:3])
print(len(r[3]))


def break_in_for_iterates_to_value():
    seen = []
    last = None
    for v in 'abcdef':
        seen.append(v)
        last = v
        if v == 'c':
            break
    return (seen, last)


print(break_in_for_iterates_to_value())


def gauntlet():
    out = []
    for i in range(4):
        if i == 0:
            continue
        for j in range(4):
            if j == 0:
                continue
            if i == 2 and j == 2:
                break
            if i == 3:
                if j == 1:
                    return ('early', out)
            out.append((i, j))
        else:
            out.append(('inner else', i))
    else:
        out.append('outer else')
    return ('done', out)


print(gauntlet())
