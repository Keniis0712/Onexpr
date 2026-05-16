def case_literal():
    out = []
    for x in (1, 2, 'foo', None, True):
        match x:
            case 1:
                out.append('one')
            case 2:
                out.append('two')
            case 'foo':
                out.append('str-foo')
            case None:
                out.append('none')
            case True:
                out.append('true')
            case _:
                out.append(('other', x))
    return out


print(case_literal())


def case_capture():
    match 42:
        case x:
            return x


print(case_capture())


def case_wildcard_only():
    match 'anything':
        case _:
            return 'matched'


print(case_wildcard_only())


def case_or():
    out = []
    for x in (1, 2, 3, 4, 5):
        match x:
            case 1 | 3 | 5:
                out.append(('odd', x))
            case 2 | 4:
                out.append(('even', x))
    return out


print(case_or())


def case_sequence_fixed():
    out = []
    for x in ([1, 2], (1, 2, 3), [1], 'no'):
        match x:
            case [1, 2]:
                out.append('two')
            case [1, 2, 3]:
                out.append('three')
            case [1]:
                out.append('one')
            case _:
                out.append(('skip', x))
    return out


print(case_sequence_fixed())


def case_sequence_capture():
    out = []
    for x in ([1, 2, 3], [10, 20], []):
        match x:
            case [a, b, c]:
                out.append(('three', a, b, c))
            case [a, b]:
                out.append(('two', a, b))
            case []:
                out.append('empty')
    return out


print(case_sequence_capture())


def case_sequence_star():
    out = []
    for x in ([1, 2, 3, 4], [1], [1, 2], [10, 20, 30, 40, 50]):
        match x:
            case [first, *rest]:
                out.append((first, rest))
    return out


print(case_sequence_star())


def case_sequence_star_middle():
    match [1, 2, 3, 4, 5]:
        case [first, *middle, last]:
            return (first, middle, last)


print(case_sequence_star_middle())


def case_sequence_excludes_str():
    # PEP 634: str / bytes / bytearray do NOT match sequence patterns.
    out = []
    for x in ('abc', [1, 2, 3], b'xyz'):
        match x:
            case [a, b, c]:
                out.append(('seq', a, b, c))
            case _:
                out.append(('not-seq', type(x).__name__))
    return out


print(case_sequence_excludes_str())


def case_mapping():
    out = []
    for x in ({'k': 'v'}, {'k': 1, 'extra': 2}, {}, [1, 2]):
        match x:
            case {'k': v}:
                out.append(('k', v))
            case {}:
                out.append('empty-dict')
            case _:
                out.append(('other', type(x).__name__))
    return out


print(case_mapping())


def case_mapping_rest():
    match {'a': 1, 'b': 2, 'c': 3}:
        case {'a': a, **rest}:
            return (a, sorted(rest.items()))


print(case_mapping_rest())


class Point:
    __match_args__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


def case_class_positional():
    out = []
    for p in (Point(0, 0), Point(0, 5), Point(3, 4)):
        match p:
            case Point(0, 0):
                out.append('origin')
            case Point(0, y):
                out.append(('y-axis', y))
            case Point(x, y):
                out.append(('point', x, y))
    return out


print(case_class_positional())


def case_class_keyword():
    p = Point(7, 8)
    match p:
        case Point(x=a, y=b):
            return (a, b)


print(case_class_keyword())


def case_class_mixed():
    p = Point(1, 2)
    match p:
        case Point(1, y=y):
            return ('mixed', y)


print(case_class_mixed())


def case_as():
    match [1, 2]:
        case [_, _] as p:
            return p


print(case_as())


def case_as_with_capture():
    match (10, 20):
        case (a, b) as p:
            return (a, b, p)


print(case_as_with_capture())


def case_guard():
    out = []
    for x in (-5, 0, 5, 100):
        match x:
            case n if n < 0:
                out.append(('neg', n))
            case 0:
                out.append('zero')
            case n if n < 10:
                out.append(('small', n))
            case n:
                out.append(('big', n))
    return out


print(case_guard())


def case_nested():
    out = []
    for d in [
        {'type': 'point', 'x': 1, 'y': 2},
        {'type': 'circle', 'radius': 5},
        {'type': 'point', 'x': 0, 'y': 0},
        {'type': 'unknown'},
    ]:
        match d:
            case {'type': 'point', 'x': 0, 'y': 0}:
                out.append('origin')
            case {'type': 'point', 'x': x, 'y': y}:
                out.append(('point', x, y))
            case {'type': 'circle', 'radius': r}:
                out.append(('circle', r))
            case _:
                out.append('other')
    return out


print(case_nested())


def case_nested_class_in_seq():
    points = [Point(1, 2), Point(3, 4)]
    match points:
        case [Point(x=x1), Point(x=x2)]:
            return (x1, x2)


print(case_nested_class_in_seq())


def case_no_match_falls_through():
    out = []
    match 99:
        case 1:
            out.append('1')
        case 2:
            out.append('2')
    out.append('after')
    return out


print(case_no_match_falls_through())


def case_only_wildcard_in_loop():
    out = []
    for x in range(3):
        match x:
            case _:
                out.append(x)
    return out


print(case_only_wildcard_in_loop())


def case_named_constant():
    SENTINEL = object()
    out = []
    for x in (SENTINEL, 1, 'other'):
        # MatchValue with Attribute / Name expr is supported; here we
        # use an attribute on a class to test that.
        match x:
            case 1:
                out.append('one')
            case _:
                out.append('other')
    return out


print(case_named_constant())


from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def case_enum_value():
    out = []
    for c in (Color.RED, Color.GREEN, Color.BLUE):
        match c:
            case Color.RED:
                out.append('red')
            case Color.GREEN:
                out.append('green')
            case _:
                out.append('other')
    return out


print(case_enum_value())


def case_match_in_function_with_capture():
    def classify(p):
        match p:
            case Point(0, 0):
                return 'origin'
            case Point(x, y) if x == y:
                return ('diagonal', x)
            case Point(x, y):
                return ('point', x, y)
            case _:
                return 'unknown'
    return [
        classify(Point(0, 0)),
        classify(Point(5, 5)),
        classify(Point(1, 2)),
        classify('not a point'),
    ]


print(case_match_in_function_with_capture())


def case_enum_iterable():
    # Enum requires metaclass.__prepare__ to return _EnumDict;
    # _make_class honors that.
    return [c.name for c in Color]


print(case_enum_iterable())


def case_enum_value_lookup():
    return Color(2).name


print(case_enum_value_lookup())
