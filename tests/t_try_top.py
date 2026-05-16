x = 'before'
try:
    x = 'inside-try'
except:
    pass
print(x)


try:
    raise ValueError('boom')
except ValueError:
    y = 'set-by-handler'
print(y)


total = 0
try:
    for i in range(5):
        total += i
except:
    pass
print(total)


try:
    z = 100
    raise RuntimeError('after-set')
except RuntimeError:
    z = z + 1
print(z)


a = 'pre'
try:
    raise ValueError('e')
except ValueError as exc:
    a = ('caught', str(exc))
print(a)


try:
    pre_else = 'pre'
except:
    pass
else:
    pre_else = 'post-else'
print(pre_else)


try:
    fin_var = 'try-set'
finally:
    fin_var = 'finally-overrides'
print(fin_var)


try:
    nested_outer = 'outer-try'
    try:
        nested_inner = 'inner-try'
    except:
        pass
except:
    pass
print(nested_outer)
print(nested_inner)
