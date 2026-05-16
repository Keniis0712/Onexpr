print(repr('plain'))
print(repr("double"))
print(repr("with 'quote'"))
print(repr('with "quote"'))


s1 = 'a\tb\nc'
print(repr(s1))
print(len(s1))
print(s1.split('\n'))


s2 = r'C:\path\to\nowhere'
print(s2)
print(len(s2))


s3 = '''line1
line2
line3'''
print(s3)
print(s3.count('\n'))


s4 = """hello
\\world"""
print(s4)


b1 = b'\x00\x01\x02'
print(len(b1))
print(list(b1))


b2 = b'hello'
print(b2.upper())
print(b2[1:4])


print('a' + 'b' + 'c')
print('-' * 5)
print('abc' * 3)
print('abcd'.replace('b', 'BB'))
print('  spaces  '.strip())
print('a,b,c'.split(','))


name = 'Alice'
age = 30
print(f'{name} is {age}')
print(f'{name!r}')
print(f'{age:5d}')
print(f'{age:>5}')
print(f'{3.14159:.2f}')
print(f'{255:#x}')
x = 7
print(f'{x = }')
print(f'{name = }')


print(0x1f)
print(0b1010)
print(0o17)
print(1_000_000)
print(2 ** 100)
print(10 ** 18)


print(1.5)
print(1e3)
print(1.5e-2)
print(round(0.1 + 0.2, 10))


z = 3 + 4j
print(z)
print(z.real, z.imag)
print(abs(z))


print(int('42'))
print(int('ff', 16))
print(int(1.9))
print(float('3.14'))
print(str(123))
print(bool([]))
print(bool([0]))


print(divmod(17, 5))
print(min([3, 1, 4]))
print(max([3, 1, 4]))
print(sorted([3, 1, 4, 1, 5]))
print(sorted([3, 1, 4, 1, 5], reverse=True))
print(any([False, False, True]))
print(all([True, True, False]))
print(list(zip([1, 2, 3], ['a', 'b', 'c'])))
print(list(enumerate(['a', 'b', 'c'], start=10)))
print(list(map(lambda x: x * 2, [1, 2, 3])))
print(list(filter(lambda x: x > 1, [0, 1, 2, 3])))
print(list(reversed([1, 2, 3])))
