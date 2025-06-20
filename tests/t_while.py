count = 0
limit = 3

while count < limit:
    n = 0
    while n < 2:
        if (count + n) % 2 == 0:
            n += 1
            continue
        print(count, n)
        break
    else:
        print("inner done")
    count += 1
else:
    print("outer done")

flag = True
attempts = 0

while flag:
    if attempts * 2 > 4:
        flag = False
        break
    attempts += 1
    print(attempts)

items = [1, 2, 3, 4]
i = 0

while i < len(items):
    if items[i] % 2 == 0:
        print(items[i])
    i += 1
