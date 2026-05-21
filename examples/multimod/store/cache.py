"""Decorator-based LRU cache (separate module to test bundling).

Stores results keyed by (args, frozenset(kwargs)). Trims to maxsize
on every miss. Plays well with closure semantics — the decorator
returns a function that captures `cache` / `order` from the
enclosing scope.
"""
from collections import OrderedDict


def memoize(maxsize=128):
    def decorator(fn):
        cache = OrderedDict()

        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            value = fn(*args, **kwargs)
            cache[key] = value
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        wrapper.cache_size = lambda: len(cache)
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator
