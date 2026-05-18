"""_await_iter: drives `await x` by extracting the underlying iterator."""


def _await_iter(x):
    if hasattr(x, '__await__'):
        return x.__await__()
    return iter(x)
