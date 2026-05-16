x = 5
print(x if x > 0 else -x)
print('big' if x >= 5 else 'small' if x >= 0 else 'neg')

vals = [10, 20, 30]
print(vals[0])
print(vals[-1])
print(vals[1:])
print(vals[::-1])

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(mat[1][2])
mat[1][2] = 99
print(mat)


class Holder:
    def __init__(self):
        self.box = {'k': [0, 0, 0]}


h = Holder()
h.box['k'][1] = 42
print(h.box)

obj = type('X', (), {})()
obj.a = type('Y', (), {})()
obj.a.b = 'deep'
print(obj.a.b)


pets = {'cat': 4, 'dog': 4, 'bird': 2}
total_legs = sum(n for _, n in pets.items())
print(total_legs)
print(sorted(pets.keys()))
print(sorted(pets.values()))


name = 'World'
n = 7
print(f'Hello, {name}!')
print(f'{n} * 2 = {n * 2}')
print(f'{n:04d}')
print(f'{n!r}')


print(1 + 2 * 3)
print((1 + 2) * 3)
print(2 ** 10)
print(7 // 2, 7 % 2)
print(-(-5))
print(not (1 == 2))
print(~0)


s = {1, 2, 3}
t = {2, 3, 4}
print(sorted(s | t))
print(sorted(s & t))
print(sorted(s - t))
print(sorted(s ^ t))


a = 1
a += 5
a -= 2
a *= 3
a //= 2
a **= 2
print(a)


print(1 < 2 < 3)
print(1 < 2 > 3)
print(0 <= 5 <= 10)
print('a' < 'b' < 'c')


print({k: v * 2 for k, v in pets.items()}['cat'])
print({x for x in range(10) if x % 3 == 0})
