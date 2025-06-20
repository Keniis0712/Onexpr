a = 1
print(a)
a = b = 2
print(a, b)
a, b = [0, 1]
print(a, b)
a, b = b, a
print(a, b)
a = [1, 2, 3]
print(a)
a[b] = 5
print(a)


class A:
    def __init__(self, val):
        self.val = val


a = A(0)
b = A(5)
print(a.val)
print(b.val)
a.val = 1
print(a.val)
a.val = 2
print(a.val)
a.val, b.val = b.val, a.val
print(a.val)
print(b.val)