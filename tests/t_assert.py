assert True
print('after assert True')

assert 1 == 1
print('after assert eq')

x = 5
assert x > 0
print('after assert x > 0')

assert x, 'should not trigger'
print('after assert with msg')

assert [1, 2, 3]
print('after assert truthy list')


def f(v):
    assert v is not None
    return v * 2


print(f(3))
print(f('a'))
