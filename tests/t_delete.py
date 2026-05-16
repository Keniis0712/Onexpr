x = 5
print(x)
del x
print('x' in globals())


a = 1
b = 2
c = 3
del a, b
print('a' in globals(), 'b' in globals(), 'c' in globals())
print(c)


class Box:
    def __init__(self):
        self.foo = 'foo_value'
        self.bar = 'bar_value'


bx = Box()
print(bx.foo, bx.bar)
del bx.foo
print(hasattr(bx, 'foo'))
print(hasattr(bx, 'bar'))


d = {'a': 1, 'b': 2, 'c': 3}
del d['b']
print(sorted(d.items()))


lst = [10, 20, 30, 40, 50]
del lst[2]
print(lst)


lst2 = [1, 2, 3, 4, 5]
del lst2[1:3]
print(lst2)


lst3 = [10, 20, 30, 40, 50]
lst3[1:4] = [99, 100]
print(lst3)


lst4 = [0, 0, 0, 0, 0]
lst4[::2] = [1, 1, 1]
print(lst4)


class Multi:
    def __init__(self):
        self.x = 'x'
        self.y = 'y'


m = Multi()
d2 = {'k': 'v', 'l': 'w'}
ll = [100, 200, 300]
del m.x, d2['k'], ll[0]
print(hasattr(m, 'x'), hasattr(m, 'y'))
print(sorted(d2.items()))
print(ll)


nested = [[1, 2, 3], [4, 5, 6]]
del nested[0][1]
print(nested)


class Tracker:
    def __init__(self):
        self.items = {'a': 1, 'b': 2}

    def remove(self, key):
        del self.items[key]


t = Tracker()
t.remove('a')
print(sorted(t.items.items()))


def del_local_module_global():
    return globals().get('g', 'no g')


g = 100
print(del_local_module_global())
print(g)


temp = 'temporary'
del temp
print('temp' in globals())
