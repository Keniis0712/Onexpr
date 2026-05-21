"""Fixture for the strip pass — exercises functions / classes that
have docstrings, type annotations, and class-level AnnAssigns. The
behaviour shouldn't depend on any of those, so the round-trip test
(default --strip=none, run via tests/test.py) should pass without
quirks.

A separate harness in test.py drives this fixture under each --strip
mode and verifies the program still produces the expected stdout.
"""


def plain(x: int, y: int = 1) -> int:
    """plain docstring."""
    return x + y


# obfuscate: keep
def kept(x: int) -> int:
    """kept docstring — protected by the marker above."""
    return x * 2


class Box:
    """class docstring."""
    name: str = 'box'
    count: int = 0

    def m(self, k: str) -> str:
        """method docstring."""
        return self.name + k


print(plain(2, 3))
print(kept(5))
print(Box().m('-x'))
print('done')
