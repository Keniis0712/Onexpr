"""match/case (PEP 634) compilation.

We turn a `match expr` into a chain of IfExp tests over the subject,
each test being one case's pattern check + guard. Patterns are
compiled in place — there's no runtime helper, every check expands
to plain expressions.

Compilation strategy
--------------------

`compile_match(stmt, frame)` returns the list of statements that
replace the Match node in the caller's body.

For each pattern we produce a `(test_expr, bind_exprs)` pair via
`_compile_pattern`:
- `test_expr` is a Python expression that evaluates truthy iff the
  subject matches the pattern (also performs all captures via walrus
  along the way; we do the captures during the test rather than
  saving and replaying because each capture's truth value is folded
  into the test using `((name := value), True)[-1]` so it never
  short-circuits the rest).
- `bind_exprs` is currently always empty — we bind in-line. It's a
  hook for if we later want a two-phase commit.

Patterns supported (PEP 634): MatchValue, MatchSingleton, MatchAs
(capture, wildcard, AS-form), MatchOr, MatchSequence (incl.
MatchStar), MatchMapping, MatchClass, plus per-case guard.

Caveats vs. CPython's reference implementation:
- We don't validate that all alternatives of a MatchOr bind the same
  set of names. Mismatched alternatives produce undefined behavior
  (i.e. whatever variables the matched alternative bound).
- Failed cases may leave behind partial captures from their pattern,
  whereas CPython tries harder to roll them back. This matches the
  letter of PEP 634 ("captured names ... will not be bound if the
  case is not selected") imperfectly, but it's the same compromise
  that other source-level rewriters make.
- We don't validate __match_args__ / kwd_attrs duplicates. The user
  is expected to write valid patterns.
"""
import ast
from typing import Optional


def compile_match(stmt: ast.Match, frame) -> list:
    """Turn a Match statement into a list of plain ast nodes.

    The shape we emit:
        subj_var = <subject>
        if test_1: body_1
        elif test_2: body_2
        ...
    """
    subj_var = frame.get_temp_var()
    out = [
        ast.Assign(
            targets=[ast.Name(id=subj_var, ctx=ast.Store())],
            value=stmt.subject,
        )
    ]

    # Build the if/elif chain in reverse order so we can nest.
    # Fall-through (no case matched) becomes the innermost orelse=[].
    chain: list = []
    subj_load = lambda: ast.Name(id=subj_var, ctx=ast.Load())

    for case in stmt.cases:
        test_expr = _compile_pattern(case.pattern, subj_load(), frame)
        if case.guard is not None:
            test_expr = ast.BoolOp(
                op=ast.And(),
                values=[test_expr, case.guard],
            )
        chain.append((test_expr, case.body))

    # Build nested If statements.
    if not chain:
        return out
    # Innermost first
    cur_orelse: list = []
    for test_expr, body in reversed(chain):
        cur = ast.If(test=test_expr, body=body, orelse=cur_orelse)
        cur_orelse = [cur]
    out.extend(cur_orelse)
    return out


def _compile_pattern(pat: ast.AST, subj: ast.expr, frame) -> ast.expr:
    """Return a truthy-iff-match expression. Captures are performed in
    place via walrus operators inside this expression."""
    if isinstance(pat, ast.MatchValue):
        return ast.Compare(
            left=subj,
            ops=[ast.Eq()],
            comparators=[pat.value],
        )

    if isinstance(pat, ast.MatchSingleton):
        return ast.Compare(
            left=subj,
            ops=[ast.Is()],
            comparators=[ast.Constant(value=pat.value)],
        )

    if isinstance(pat, ast.MatchAs):
        if pat.pattern is None and pat.name is None:
            # wildcard
            return ast.Constant(value=True)
        if pat.pattern is None:
            # capture
            return _capture(pat.name, subj)
        # P as name
        return ast.BoolOp(
            op=ast.And(),
            values=[
                _compile_pattern(pat.pattern, subj, frame),
                _capture(pat.name, subj),
            ],
        )

    if isinstance(pat, ast.MatchOr):
        return ast.BoolOp(
            op=ast.Or(),
            values=[
                _compile_pattern(p, subj, frame) for p in pat.patterns
            ],
        )

    if isinstance(pat, ast.MatchSequence):
        return _compile_sequence(pat, subj, frame)

    if isinstance(pat, ast.MatchMapping):
        return _compile_mapping(pat, subj, frame)

    if isinstance(pat, ast.MatchClass):
        return _compile_class(pat, subj, frame)

    raise NotImplementedError(f'match pattern: {type(pat).__name__}')


def _capture(name: str, value: ast.expr) -> ast.expr:
    """`(name := value, True)[-1]` — always truthy, performs the bind."""
    return ast.Subscript(
        value=ast.Tuple(
            elts=[
                ast.NamedExpr(
                    target=ast.Name(id=name, ctx=ast.Store()),
                    value=value,
                ),
                ast.Constant(value=True),
            ],
            ctx=ast.Load(),
        ),
        slice=ast.Constant(value=-1),
        ctx=ast.Load(),
    )


def _is_sequence_check(subj: ast.expr) -> ast.expr:
    """Match the same definition CPython uses: an instance of
    collections.abc.Sequence that is not a str/bytes/bytearray."""
    return ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Call(
                func=ast.Name(id='isinstance', ctx=ast.Load()),
                args=[
                    subj,
                    ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='__import__', ctx=ast.Load()),
                            args=[ast.Constant(value='collections.abc')],
                            keywords=[
                                ast.keyword(
                                    arg='fromlist',
                                    value=ast.List(
                                        elts=[ast.Constant(value='abc')],
                                        ctx=ast.Load(),
                                    ),
                                ),
                            ],
                        ),
                        attr='Sequence',
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            ),
            ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id='isinstance', ctx=ast.Load()),
                    args=[
                        subj,
                        ast.Tuple(
                            elts=[
                                ast.Name(id='str', ctx=ast.Load()),
                                ast.Name(id='bytes', ctx=ast.Load()),
                                ast.Name(id='bytearray', ctx=ast.Load()),
                            ],
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                ),
            ),
        ],
    )


def _compile_sequence(pat: ast.MatchSequence, subj: ast.expr, frame) -> ast.expr:
    patterns = pat.patterns
    star_index = None
    for i, p in enumerate(patterns):
        if isinstance(p, ast.MatchStar):
            star_index = i
            break

    fixed_count = len(patterns) - (1 if star_index is not None else 0)

    # length check
    if star_index is None:
        len_check = ast.Compare(
            left=ast.Call(
                func=ast.Name(id='len', ctx=ast.Load()),
                args=[subj],
                keywords=[],
            ),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=len(patterns))],
        )
    else:
        len_check = ast.Compare(
            left=ast.Call(
                func=ast.Name(id='len', ctx=ast.Load()),
                args=[subj],
                keywords=[],
            ),
            ops=[ast.GtE()],
            comparators=[ast.Constant(value=fixed_count)],
        )

    # Sub-pattern checks
    sub_tests: list = []
    if star_index is None:
        for i, p in enumerate(patterns):
            sub_tests.append(_compile_pattern(p, _index(subj, i), frame))
    else:
        before = patterns[:star_index]
        star = patterns[star_index]
        after = patterns[star_index + 1:]
        for i, p in enumerate(before):
            sub_tests.append(_compile_pattern(p, _index(subj, i), frame))
        if star.name is not None:
            sub_tests.append(
                _capture(
                    star.name,
                    ast.Call(
                        func=ast.Name(id='list', ctx=ast.Load()),
                        args=[_slice(subj, star_index, -len(after) if after else None)],
                        keywords=[],
                    ),
                )
            )
        for i, p in enumerate(after):
            offset = i - len(after)  # negative index
            sub_tests.append(_compile_pattern(p, _index(subj, offset), frame))

    parts = [_is_sequence_check(subj), len_check] + sub_tests
    return ast.BoolOp(op=ast.And(), values=parts)


def _index(subj: ast.expr, i: int) -> ast.expr:
    return ast.Subscript(
        value=subj,
        slice=ast.Constant(value=i),
        ctx=ast.Load(),
    )


def _slice(subj: ast.expr, start: int, stop) -> ast.expr:
    return ast.Subscript(
        value=subj,
        slice=ast.Slice(
            lower=ast.Constant(value=start),
            upper=ast.Constant(value=stop),
            step=None,
        ),
        ctx=ast.Load(),
    )


def _compile_mapping(pat: ast.MatchMapping, subj: ast.expr, frame) -> ast.expr:
    parts: list = [
        ast.Call(
            func=ast.Name(id='isinstance', ctx=ast.Load()),
            args=[
                subj,
                ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id='__import__', ctx=ast.Load()),
                        args=[ast.Constant(value='collections.abc')],
                        keywords=[
                            ast.keyword(
                                arg='fromlist',
                                value=ast.List(
                                    elts=[ast.Constant(value='abc')],
                                    ctx=ast.Load(),
                                ),
                            ),
                        ],
                    ),
                    attr='Mapping',
                    ctx=ast.Load(),
                ),
            ],
            keywords=[],
        ),
    ]

    for key, sub_pat in zip(pat.keys, pat.patterns):
        # subj has the key
        parts.append(
            ast.Compare(
                left=key,
                ops=[ast.In()],
                comparators=[subj],
            )
        )
        # the value at key matches sub_pat
        parts.append(
            _compile_pattern(
                sub_pat,
                ast.Subscript(value=subj, slice=key, ctx=ast.Load()),
                frame,
            )
        )

    if pat.rest is not None:
        # Bind rest = {k: v for k, v in subj.items() if k not in keys}
        keys_tuple = ast.Tuple(elts=list(pat.keys), ctx=ast.Load())
        dict_comp = ast.DictComp(
            key=ast.Name(id='_k', ctx=ast.Load()),
            value=ast.Name(id='_v', ctx=ast.Load()),
            generators=[
                ast.comprehension(
                    target=ast.Tuple(
                        elts=[
                            ast.Name(id='_k', ctx=ast.Store()),
                            ast.Name(id='_v', ctx=ast.Store()),
                        ],
                        ctx=ast.Store(),
                    ),
                    iter=ast.Call(
                        func=ast.Attribute(
                            value=subj,
                            attr='items',
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    ifs=[
                        ast.Compare(
                            left=ast.Name(id='_k', ctx=ast.Load()),
                            ops=[ast.NotIn()],
                            comparators=[keys_tuple],
                        ),
                    ],
                    is_async=0,
                ),
            ],
        )
        parts.append(_capture(pat.rest, dict_comp))

    return ast.BoolOp(op=ast.And(), values=parts)


def _compile_class(pat: ast.MatchClass, subj: ast.expr, frame) -> ast.expr:
    parts: list = [
        ast.Call(
            func=ast.Name(id='isinstance', ctx=ast.Load()),
            args=[subj, pat.cls],
            keywords=[],
        ),
    ]

    # Positional sub-patterns are matched against attributes named by
    # `cls.__match_args__[i]`. We resolve __match_args__ at runtime
    # since it's a class-level attribute.
    if pat.patterns:
        # match_args = cls.__match_args__
        # then for each positional pattern, getattr(subj, match_args[i])
        # We expand inline rather than using a temp variable to keep
        # everything as a single expression.
        match_args_expr = ast.Attribute(
            value=pat.cls,
            attr='__match_args__',
            ctx=ast.Load(),
        )
        for i, sub_pat in enumerate(pat.patterns):
            attr_name = ast.Subscript(
                value=match_args_expr,
                slice=ast.Constant(value=i),
                ctx=ast.Load(),
            )
            parts.append(
                _compile_pattern(
                    sub_pat,
                    ast.Call(
                        func=ast.Name(id='getattr', ctx=ast.Load()),
                        args=[subj, attr_name],
                        keywords=[],
                    ),
                    frame,
                )
            )

    # Keyword sub-patterns: each (name, pattern) means subj.name matches pattern.
    for attr_name, sub_pat in zip(pat.kwd_attrs, pat.kwd_patterns):
        parts.append(
            _compile_pattern(
                sub_pat,
                ast.Attribute(
                    value=subj,
                    attr=attr_name,
                    ctx=ast.Load(),
                ),
                frame,
            )
        )

    return ast.BoolOp(op=ast.And(), values=parts)
