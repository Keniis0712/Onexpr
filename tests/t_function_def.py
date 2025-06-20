def deco(f):
    def wrapper(*args, **kwargs):
        print(args, kwargs)
        return f(*args, **kwargs)
    return wrapper

@deco
def func(a):
    print(a+1)
    return 2*a

print(func(func(4)))

@deco
def fact(n):
    if n == 0:
        return 1
    return n*fact(n-1)

print(fact(5))
print(fact(2))
