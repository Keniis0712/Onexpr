data = {"a": 1, "b": 2}

for i in range(3):
    for key, value in data.items():
        if value % 2 == 1:
            print(key, value)
else:
    for x in (j for j in (i**2 for i in range(2))):
        print(x)

for a, b in zip("abc", range(3)):
    if a < 'b':
        pass
    else:
        print(a, b)

for i, x in enumerate([0, 1, 2, 3]):
    if x % 2 == 0:
        continue
    for j in "xy":
        if j == 'y':
            break
        print(i, x, j)
else:
    for _ in ():
        print("never")
