class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add(self, child):
        self.children.append(child)
        return child


root = TreeNode('root')
a = root.add(TreeNode('a'))
b = root.add(TreeNode('b'))
a.add(TreeNode('a1'))
a.add(TreeNode('a2'))
a1_grand = a.children[0].add(TreeNode('a1.1'))
b.add(TreeNode('b1'))


def collect_dfs(node):
    out = [node.value]
    for c in node.children:
        for v in collect_dfs(c):
            out.append(v)
    return out


print(collect_dfs(root))


def depth(node):
    if not node.children:
        return 1
    best = 0
    for c in node.children:
        d = depth(c)
        if d > best:
            best = d
    return best + 1


print(depth(root))
print(depth(b))


def find(node, target):
    if node.value == target:
        return node
    for c in node.children:
        result = find(c, target)
        if result is not None:
            return result
    return None


print(find(root, 'a1.1').value)
print(find(root, 'missing'))


class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, v):
        node = {'value': v, 'next': self.head}
        self.head = node
        self.size += 1

    def to_list(self):
        out = []
        cur = self.head
        while cur is not None:
            out.append(cur['value'])
            cur = cur['next']
        return out

    def reverse(self):
        prev = None
        cur = self.head
        while cur is not None:
            nxt = cur['next']
            cur['next'] = prev
            prev = cur
            cur = nxt
        self.head = prev


ll = LinkedList()
for v in [1, 2, 3, 4, 5]:
    ll.push(v)
print(ll.to_list())
print(ll.size)
ll.reverse()
print(ll.to_list())


def fib_seq(n):
    if n <= 0:
        return []
    out = [0]
    if n == 1:
        return out
    out.append(1)
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out


print(fib_seq(0))
print(fib_seq(1))
print(fib_seq(10))


def quicksort(xs):
    if len(xs) <= 1:
        return list(xs)
    pivot = xs[0]
    less = [x for x in xs[1:] if x < pivot]
    eq = [x for x in xs if x == pivot]
    more = [x for x in xs[1:] if x > pivot]
    return quicksort(less) + eq + quicksort(more)


print(quicksort([]))
print(quicksort([5]))
print(quicksort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]))
print(quicksort([5, 4, 3, 2, 1]))


def reduce(f, xs, init):
    acc = init
    for x in xs:
        acc = f(acc, x)
    return acc


print(reduce(lambda a, b: a + b, [1, 2, 3, 4, 5], 0))
print(reduce(lambda a, b: a * b, [1, 2, 3, 4, 5], 1))
print(reduce(lambda a, b: a if a > b else b, [3, 1, 4, 1, 5, 9, 2, 6], 0))
