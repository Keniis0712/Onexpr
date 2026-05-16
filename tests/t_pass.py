class Empty:
    pass


print(Empty.__name__)


def noop():
    pass


print(noop())


for i in range(3):
    pass
print('after for-pass')

x = 5
if x > 0:
    pass
else:
    print('neg')
print('after if-pass')
