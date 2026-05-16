add = lambda a, b: a + b
print(add(2, 3))

mul = lambda x, y=10: x * y
print(mul(3))
print(mul(3, 4))

print((lambda x: x ** 2)(5))


funcs = [lambda x, k=k: x + k for k in range(4)]
print([f(10) for f in funcs])


pairs = [(1, 'a'), (3, 'c'), (2, 'b')]
pairs.sort(key=lambda p: p[0])
print(pairs)


def apply(f, x):
    return f(x)


print(apply(lambda v: v * 100, 7))
print(apply(lambda v: -v, 9))


squares = [x * x for x in range(5)]
print(squares)


pairs2 = [(x, y) for x in range(3) for y in 'ab']
print(pairs2)


odd_squares = [x * x for x in range(10) if x % 2 == 1]
print(odd_squares)


d = {x: x * x for x in range(4)}
print(sorted(d.items()))


s = {x % 3 for x in range(10)}
print(sorted(s))


total = sum(x * x for x in range(5))
print(total)


grid = [[i * 10 + j for j in range(3)] for i in range(3)]
print(grid)
