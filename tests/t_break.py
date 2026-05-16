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
