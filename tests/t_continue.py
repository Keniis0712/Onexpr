for i in range(5):
    if i % 2 == 0:
        continue
    print('odd', i)
print('after for')


for i in range(5):
    if i == 3:
        continue
    print('skip3', i)
    if i == 4:
        print('and saw 4')
print('after for 2')


x = 0
while x < 6:
    x = x + 1
    if x % 2 == 0:
        continue
    print('w odd', x)
print('after while')


for i in range(3):
    for j in range(3):
        if j == 1:
            continue
        print('n', i, j)
print('after nested')


# Bare continue (not inside an if). Code after is unreachable.
out = []
for i in range(3):
    out.append(('before', i))
    continue
    out.append(('unreachable', i))
print('bare continue for:', out)


out2 = []
n = 0
while n < 4:
    n += 1
    out2.append(n)
    continue
    out2.append(('unreachable', n))
print('bare continue while:', out2)


# Bare continue only skips the inner-loop iteration; outer keeps going.
out3 = []
for i in range(3):
    for j in range(2):
        out3.append(('inner', i, j))
        continue
        out3.append(('inner-unreachable',))
    out3.append(('outer-after', i))
print('nested bare continue:', out3)
