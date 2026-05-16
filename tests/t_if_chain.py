x = 5

if x < 0:
    print('neg')
elif x == 0:
    print('zero')
elif x < 10:
    print('small')
elif x < 100:
    print('medium')
else:
    print('big')


def classify(n):
    if n < 0:
        return 'neg'
    elif n == 0:
        return 'zero'
    elif n % 2 == 0:
        return 'even pos'
    else:
        return 'odd pos'


for v in [-3, 0, 1, 4, 7, 100]:
    print(classify(v))


def chained(a, b, c):
    if a < b < c:
        return 'asc'
    if a > b > c:
        return 'desc'
    return 'other'


print(chained(1, 2, 3))
print(chained(3, 2, 1))
print(chained(1, 1, 2))


def cmp_eq(a, b):
    if a == b == 0:
        return 'both zero'
    return 'not'


print(cmp_eq(0, 0))
print(cmp_eq(0, 1))


def boolean_short(a, b):
    if a and b:
        return 'both'
    if a or b:
        return 'one'
    return 'none'


print(boolean_short(1, 1))
print(boolean_short(0, 1))
print(boolean_short(0, 0))


def use_in(x):
    if x in (1, 2, 3):
        return 'low'
    if x in {10, 20, 30}:
        return 'mid'
    return 'unknown'


print(use_in(2))
print(use_in(20))
print(use_in(5))
