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
