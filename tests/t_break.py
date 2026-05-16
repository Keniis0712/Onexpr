for i in range(10):
    if i == 3:
        break
    print('for', i)
else:
    print('for else')
print('after for')


for i in range(10):
    print('full', i)
    if i == 9:
        break
else:
    print('for else 2')
print('after for 2')


x = 0
while x < 5:
    if x == 2:
        break
    print('while', x)
    x = x + 1
else:
    print('while else')
print('after while')


for i in range(3):
    for j in range(5):
        if j == 2:
            break
        print('nested', i, j)
print('after nested')


# Bare break (not inside an if). Code after `break` is unreachable in
# Python and `break` itself exits the loop without running it.
out = []
for i in range(5):
    out.append(('before', i))
    break
    out.append(('unreachable', i))
print('bare break for:', out)


# Same but with a while.
out2 = []
n = 0
while n < 10:
    out2.append(n)
    break
    out2.append(('unreachable', n))
    n += 1
print('bare break while:', out2)


# Nested: bare break only exits the inner loop.
out3 = []
for i in range(3):
    for j in range(5):
        out3.append(('inner', i, j))
        break
        out3.append(('inner-unreachable',))
    out3.append(('outer-after', i))
print('nested bare break:', out3)
