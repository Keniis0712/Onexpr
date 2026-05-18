import _ast
import ast

from ..frame import Frame
from ..helpers import for_helper_name, while_helper_name, try_helper_name
from .dispatch import parse_stmts


def make_loop_var_escape(target, frame) -> ast.expr:
    """Surface the for-loop target to the enclosing scope.

    Python binds `i` after `for i in ...:` to the last iterated value
    (and leaves `i` unbound when the iterable was empty). The ListComp
    we rewrite to has its own scope, so we have to re-emit the binding
    explicitly.

    Supports plain Name targets and Tuple/List targets (`for a, b in
    ...`), including a single Starred element anywhere in the tuple
    (`for *a, b in ...`, `for a, *b in ...`, `for a, *b, c in ...`).

    The expression must always evaluate falsy so it doesn't short-
    circuit the surrounding Or chain — even when last_yielded is a
    truthy value.
    """
    helper = ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
    last_yielded = ast.Attribute(value=helper, attr='last_yielded', ctx=ast.Load())

    if isinstance(target, ast.Name):
        bind = ast.NamedExpr(
            target=ast.Name(id=target.id, ctx=ast.Store()),
            value=last_yielded,
        )
        # Wrap the bind in `(bind, False)[1]` so the surrounding Or
        # chain doesn't call __bool__ on whatever `last_yielded`
        # holds — numpy arrays raise on bool() of multi-element
        # instances. When `was_iterated` is False we short-circuit
        # before touching the bind at all.
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Attribute(value=helper, attr='was_iterated', ctx=ast.Load()),
                ast.Subscript(
                    value=ast.Tuple(
                        elts=[bind, ast.Constant(value=False)],
                        ctx=ast.Load(),
                    ),
                    slice=ast.Constant(value=1),
                    ctx=ast.Load(),
                ),
            ],
        )

    if isinstance(target, (ast.Tuple, ast.List)) and all(
        isinstance(e, ast.Name) or
        (isinstance(e, ast.Starred) and isinstance(e.value, ast.Name))
        for e in target.elts
    ):
        # Materialize last_yielded once into a list so subscripting and
        # slicing both work regardless of the iterable's underlying
        # type. The list is bound via walrus, then each name is bound
        # by index/slice from it.
        materialized = frame.get_temp_var()
        elts = target.elts
        starred_idx = None
        for i, e in enumerate(elts):
            if isinstance(e, ast.Starred):
                starred_idx = i
                break

        binds = [
            ast.NamedExpr(
                target=ast.Name(id=materialized, ctx=ast.Store()),
                value=ast.Call(
                    func=ast.Name(id='list', ctx=ast.Load()),
                    args=[last_yielded],
                    keywords=[],
                ),
            )
        ]

        def name_of(e):
            return e.value.id if isinstance(e, ast.Starred) else e.id

        if starred_idx is None:
            for i, e in enumerate(elts):
                binds.append(
                    ast.NamedExpr(
                        target=ast.Name(id=name_of(e), ctx=ast.Store()),
                        value=ast.Subscript(
                            value=ast.Name(id=materialized, ctx=ast.Load()),
                            slice=ast.Constant(value=i),
                            ctx=ast.Load(),
                        ),
                    )
                )
        else:
            # Pre-star names from positive indices.
            for i in range(starred_idx):
                binds.append(
                    ast.NamedExpr(
                        target=ast.Name(id=name_of(elts[i]), ctx=ast.Store()),
                        value=ast.Subscript(
                            value=ast.Name(id=materialized, ctx=ast.Load()),
                            slice=ast.Constant(value=i),
                            ctx=ast.Load(),
                        ),
                    )
                )
            # Post-star names from negative indices.
            after = len(elts) - starred_idx - 1
            for j in range(after):
                neg = -(after - j)
                binds.append(
                    ast.NamedExpr(
                        target=ast.Name(id=name_of(elts[starred_idx + 1 + j]), ctx=ast.Store()),
                        value=ast.Subscript(
                            value=ast.Name(id=materialized, ctx=ast.Load()),
                            slice=ast.Constant(value=neg),
                            ctx=ast.Load(),
                        ),
                    )
                )
            # Starred name takes the middle slice.
            upper = ast.Constant(value=-after) if after > 0 else ast.Constant(value=None)
            binds.append(
                ast.NamedExpr(
                    target=ast.Name(id=name_of(elts[starred_idx]), ctx=ast.Store()),
                    value=ast.Subscript(
                        value=ast.Name(id=materialized, ctx=ast.Load()),
                        slice=ast.Slice(
                            lower=ast.Constant(value=starred_idx),
                            upper=upper,
                            step=None,
                        ),
                        ctx=ast.Load(),
                    ),
                )
            )

        bind_tuple = ast.Subscript(
            value=ast.Tuple(
                elts=binds + [ast.Constant(value=False)],
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=-1),
            ctx=ast.Load(),
        )
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Attribute(value=helper, attr='was_iterated', ctx=ast.Load()),
                bind_tuple,
            ],
        )

    # Anything more exotic (nested unpacks) — give up and return a no-op
    # falsy.
    return ast.Constant(value=False)


def make_return_propagator(frame: Frame) -> ast.expr:
    """If the current frame returned inside the loop body, surface the
    return signal to the enclosing Or chain so it short-circuits."""
    helper = ast.Name(id=frame.func_helper_var, ctx=ast.Load())
    return ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Attribute(value=helper, attr='returned', ctx=ast.Load()),
            ast.Constant(value=True),
        ],
    )


def add_orelse(orelse, frame):
    return ast.If(
        test=ast.BoolOp(
            op=ast.And(),
            values=[
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Attribute(
                        value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                        attr='stopped',
                        ctx=ast.Load(),
                    )
                ),
                ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Attribute(
                        value=ast.Name(id=frame.func_helper_var, ctx=ast.Load()),
                        attr='returned',
                        ctx=ast.Load(),
                    )
                ),
            ],
        ),
        body=orelse,
        orelse=[],
    )


def parse_for(stmt: ast.For, frame: Frame) -> list[_ast.AST]:
    frame.enter_loop()
    stmts: list[_ast.AST] = [
        ast.Assign(
            targets=[ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=frame.get_helper_name(for_helper_name), ctx=ast.Load()),
                args=[
                    stmt.iter,
                    ast.Name(id=frame.func_helper_var, ctx=ast.Load()),
                ],
                keywords=[],
            )
        ),
        ast.Expr(
            value=ast.ListComp(
                elt=ast.Constant(value=None),
                generators=[
                    ast.comprehension(
                        target=stmt.target,
                        iter=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                        ifs=[parse_stmts(stmt.body, frame)],
                        is_async=0
                    )
                ]
            )
        )
    ]
    if stmt.orelse:
        stmts.append(
            add_orelse(stmt.orelse, frame)
        )
    stmts.append(make_loop_var_escape(stmt.target, frame))
    stmts.append(make_return_propagator(frame))
    frame.exit_loop()
    return stmts


def parse_async_for(stmt: ast.AsyncFor, frame: Frame) -> list[_ast.AST]:
    pass


def parse_while(stmt: ast.While, frame: Frame) -> list[_ast.AST]:
    frame.enter_loop()
    body: list = [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='cond',
                ctx=ast.Load(),
            ),
            args=[stmt.test],
            keywords=[],
        )
    ]
    body += stmt.body

    stmts: list[_ast.AST] = [
        ast.Assign(
            targets=[ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())],
            value=ast.Call(
                func=ast.Name(id=frame.get_helper_name(while_helper_name), ctx=ast.Load()),
                args=[ast.Name(id=frame.func_helper_var, ctx=ast.Load())],
                keywords=[],
            )
        ),
        ast.Expr(
            value=ast.ListComp(
                elt=ast.Constant(value=None),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id='_', ctx=ast.Load()),
                        iter=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                        ifs=[parse_stmts(body, frame)],
                        is_async=0
                    )
                ]
            )
        )
    ]
    if stmt.orelse:
        stmts.append(
            add_orelse(stmt.orelse, frame)
        )
    stmts.append(make_return_propagator(frame))
    frame.exit_loop()
    return stmts


def parse_if(stmt: ast.If, frame: Frame) -> list[_ast.AST]:
    return [
        ast.IfExp(
            test=stmt.test,
            body=parse_stmts(stmt.body, frame),
            orelse=parse_stmts(stmt.orelse, frame) if stmt.orelse else ast.Constant(None),
        )
    ]


def parse_with(stmt: ast.With, frame: Frame) -> list[_ast.AST]:
    # `with a, b as x: BODY` is equivalent to `with a: with b as x: BODY`.
    # Unwrap multiple items into nested Withs and let the recursion
    # below (parse_stmts -> parse_stmt -> parse_with) handle the rest.
    if len(stmt.items) > 1:
        first, *rest = stmt.items
        inner = ast.With(items=rest, body=stmt.body)
        return [ast.With(items=[first], body=[inner])]

    item = stmt.items[0]
    helper_name_node = ast.Name(id=frame.func_helper_var, ctx=ast.Load())
    loop_name_node = (
        ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
        if frame.loops else ast.Constant(value=None)
    )

    # Allocate a temp to hold the manager so __enter__ runs once and
    # __exit__ later refers to the same instance. PEP 343 says
    # __exit__ must be looked up on the type, not the instance — our
    # _TryHelper.with_block does that, so emit `mgr.__enter__()` here
    # but pass mgr to with_block.
    mgr_var = frame.get_temp_var()

    stmts = [
        ast.Assign(
            targets=[ast.Name(id=mgr_var, ctx=ast.Store())],
            value=item.context_expr,
        ),
    ]

    # Bind the `as` target if present. We resolve __enter__ via
    # type(mgr).__enter__(mgr) to mirror CPython's behavior of
    # ignoring instance attribute shadowing.
    enter_call = ast.Call(
        func=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id='type', ctx=ast.Load()),
                args=[ast.Name(id=mgr_var, ctx=ast.Load())],
                keywords=[],
            ),
            attr='__enter__',
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=mgr_var, ctx=ast.Load())],
        keywords=[],
    )
    if item.optional_vars is not None:
        stmts.append(
            ast.Assign(
                targets=[item.optional_vars],
                value=enter_call,
            )
        )
    else:
        # Still call __enter__ for its side effect / required by PEP 343,
        # just discard the value.
        stmts.append(ast.Expr(value=enter_call))

    # Body lambda
    body_or = parse_stmts(stmt.body, frame)
    body_lambda = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=body_or,
    )

    e_pending = frame.get_temp_var()
    # e_pending = _TryHelper.with_block(mgr, body_lambda)
    stmts.append(
        ast.Assign(
            targets=[ast.Name(id=e_pending, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=frame.get_helper_name(try_helper_name), ctx=ast.Load()),
                    attr='with_block',
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id=mgr_var, ctx=ast.Load()),
                    body_lambda,
                ],
                keywords=[],
            ),
        )
    )

    # if e_pending is not None and not helper.returned [and not loop.stopped]:
    #     throw e_pending
    throw_expr = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.GeneratorExp(
                    elt=ast.Constant(value=None),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id='_', ctx=ast.Store()),
                            iter=ast.List(elts=[], ctx=ast.Load()),
                            ifs=[],
                            is_async=0,
                        )
                    ],
                ),
                attr='throw',
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=e_pending, ctx=ast.Load())],
            keywords=[],
        )
    )
    not_returned = ast.UnaryOp(
        op=ast.Not(),
        operand=ast.Attribute(value=helper_name_node, attr='returned', ctx=ast.Load()),
    )
    throw_test_terms = [
        ast.Compare(
            left=ast.Name(id=e_pending, ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        not_returned,
    ]
    if frame.loops:
        throw_test_terms.append(
            ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Attribute(value=loop_name_node, attr='stopped', ctx=ast.Load()),
            )
        )
    stmts.append(
        ast.If(
            test=ast.BoolOp(op=ast.And(), values=throw_test_terms),
            body=[throw_expr],
            orelse=[],
        )
    )

    # Return / break / continue propagators (same as parse_try).
    stmts.append(make_return_propagator(frame))
    if frame.loops:
        loop_ref = ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
        stmts.append(
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Attribute(value=loop_ref, attr='stopped', ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
        stmts.append(
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Attribute(value=loop_ref, attr='pending_continue', ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
    return stmts


def parse_async_with(stmt: ast.AsyncWith, frame: Frame) -> list[_ast.AST]:
    pass


def parse_match(stmt: ast.Match, frame: Frame) -> list[_ast.AST]:
    from ..match_patterns import compile_match
    return compile_match(stmt, frame)
