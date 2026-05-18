import _ast
import ast

from ..frame import Frame
from ..helpers import try_helper_name
from .dispatch import parse_stmts
from .control_flow import make_return_propagator


def parse_raise(stmt: ast.Raise, frame: Frame) -> list[_ast.AST]:
    # Bare `raise` reraises the current handler's exception.
    #   - If we're textually inside an `except E as e:` clause being
    #     transformed right now, the parser pushed `e` onto
    #     frame.exc_stack — use that directly.
    #   - Otherwise (the bare `raise` is in a function CALLED from a
    #     handler — common pattern), look up the active exception at
    #     runtime via _TryHelper._exc_stack, which our dispatch fills
    #     in. Falls back to sys.exc_info()[1] for code that runs in a
    #     real Python except (e.g. via a callback the C API invoked
    #     from one).
    if stmt.exc is None:
        if frame.exc_stack:
            exc_to_throw = ast.Name(id=frame.exc_stack[-1], ctx=ast.Load())
        else:
            # Build: (_TryHelper._exc_stack[-1]
            #        if _TryHelper._exc_stack
            #        else __import__('sys').exc_info()[1])
            stack_attr = ast.Attribute(
                value=ast.Name(id=frame.get_helper_name(try_helper_name), ctx=ast.Load()),
                attr=frame.get_helper_member('_exc_stack'), ctx=ast.Load(),
            )
            sys_excinfo = ast.Subscript(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='__import__', ctx=ast.Load()),
                            args=[ast.Constant(value='sys')],
                            keywords=[],
                        ),
                        attr='exc_info', ctx=ast.Load(),
                    ),
                    args=[], keywords=[],
                ),
                slice=ast.Constant(value=1), ctx=ast.Load(),
            )
            exc_to_throw = ast.IfExp(
                test=stack_attr,
                body=ast.Subscript(
                    value=stack_attr,
                    slice=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                    ctx=ast.Load(),
                ),
                orelse=sys_excinfo,
            )
        stmts = []
    elif stmt.cause:
        temp_exc_var = frame.get_temp_var()
        stmts = [
            ast.Assign(
                targets=[ast.Name(id=temp_exc_var)],
                value=stmt.exc
            ),
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=temp_exc_var, ctx=ast.Load()),
                        attr='__cause__',
                        ctx=ast.Store(),
                    )
                ],
                value=stmt.cause
            )
        ]
        exc_to_throw = ast.Name(id=temp_exc_var, ctx=ast.Load())
    else:
        stmts = []
        exc_to_throw = stmt.exc

    stmts += [
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.GeneratorExp(
                        elt=ast.Constant(value=None),
                        generators=[
                            ast.comprehension(
                                target=ast.Name(id='_', ctx=ast.Store()),
                                iter=ast.List(elts=[], ctx=ast.Load()),
                                ifs=[],
                                is_async=0)
                        ]
                    ),
                    attr='throw',
                    ctx=ast.Load()
                ),
                args=[exc_to_throw],
                keywords=[],
            )
        )
    ]
    return stmts


def parse_try(stmt: ast.Try, frame: Frame) -> list[_ast.AST]:
    return _parse_try_common(stmt, frame, dispatch_attr=frame.get_helper_member('dispatch'))


def _parse_try_common(stmt, frame: Frame, dispatch_attr: str) -> list[_ast.AST]:
    helper_name = ast.Name(id=frame.func_helper_var, ctx=ast.Load())
    loop_name = (
        ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
        if frame.loops else ast.Constant(value=None)
    )

    # body lambda: lambda: <or chain of body>
    body_or = parse_stmts(stmt.body, frame)
    body_lambda = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=body_or,
    )

    # handlers list: [(exc_types_or_None, lambda exc_name: handler_or_chain), ...]
    handler_tuples = []
    for h in stmt.handlers:
        exc_types = h.type if h.type is not None else ast.Constant(value=None)
        # The user-facing as-name (or a synthetic one if `except E:` has
        # no `as`). Either way, this becomes the handler lambda's
        # parameter, so bare `raise` inside the handler can reference
        # it via frame.exc_stack.
        param_name = h.name if h.name is not None else frame.get_temp_var()
        frame.exc_stack.append(param_name)
        handler_or = parse_stmts(h.body, frame)
        frame.exc_stack.pop()
        handler_lambda = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=param_name, annotation=None)],
                vararg=None,
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=handler_or,
        )
        handler_tuples.append(ast.Tuple(
            elts=[exc_types, handler_lambda],
            ctx=ast.Load(),
        ))
    handlers_list = ast.List(elts=handler_tuples, ctx=ast.Load())

    # else lambda
    if stmt.orelse:
        else_or = parse_stmts(stmt.orelse, frame)
        else_lambda = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=else_or,
        )
    else:
        else_lambda = ast.Constant(value=None)

    # finally lambda
    has_finally = bool(stmt.finalbody)
    if has_finally:
        finally_or = parse_stmts(stmt.finalbody, frame)
        finally_lambda = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=finally_or,
        )

    # Names for the staged exception values.
    e_pending = frame.get_temp_var()
    e_finally_var = frame.get_temp_var() if has_finally else None
    e_to_raise = frame.get_temp_var()

    stmts = [
        # e_pending = _TryHelper.dispatch(body, handlers, else, fhelper, lhelper)
        ast.Assign(
            targets=[ast.Name(id=e_pending, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=frame.get_helper_name(try_helper_name), ctx=ast.Load()),
                    attr=dispatch_attr,
                    ctx=ast.Load(),
                ),
                args=[body_lambda, handlers_list, else_lambda, helper_name, loop_name],
                keywords=[],
            ),
        ),
    ]

    if has_finally:
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=e_finally_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=frame.get_helper_name(try_helper_name), ctx=ast.Load()),
                        attr=frame.get_helper_member('guarded'),
                        ctx=ast.Load(),
                    ),
                    args=[finally_lambda],
                    keywords=[],
                ),
            )
        )
        # Chain context: if both `finally` and the pre-finally body
        # produced exceptions, Python sets e_finally.__context__ to
        # e_pending so the user sees "during handling of <e_pending>,
        # another exception occurred". The generator-throw trick
        # bypasses CPython's automatic chaining, so re-create it.
        stmts.append(
            ast.Expr(
                value=ast.IfExp(
                    test=ast.BoolOp(
                        op=ast.And(),
                        values=[
                            ast.Compare(
                                left=ast.Name(id=e_finally_var, ctx=ast.Load()),
                                ops=[ast.IsNot()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            ast.Compare(
                                left=ast.Name(id=e_pending, ctx=ast.Load()),
                                ops=[ast.IsNot()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            ast.Compare(
                                left=ast.Attribute(
                                    value=ast.Name(id=e_finally_var, ctx=ast.Load()),
                                    attr='__context__',
                                    ctx=ast.Load(),
                                ),
                                ops=[ast.Is()],
                                comparators=[ast.Constant(value=None)],
                            ),
                        ],
                    ),
                    body=ast.Call(
                        func=ast.Name(id='setattr', ctx=ast.Load()),
                        args=[
                            ast.Name(id=e_finally_var, ctx=ast.Load()),
                            ast.Constant(value='__context__'),
                            ast.Name(id=e_pending, ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                    orelse=ast.Constant(value=None),
                )
            )
        )

    # e_to_raise = e_finally if has_finally and e_finally is not None else e_pending
    if has_finally:
        to_raise_value = ast.IfExp(
            test=ast.Compare(
                left=ast.Name(id=e_finally_var, ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=ast.Name(id=e_finally_var, ctx=ast.Load()),
            orelse=ast.Name(id=e_pending, ctx=ast.Load()),
        )
    else:
        to_raise_value = ast.Name(id=e_pending, ctx=ast.Load())

    stmts.append(
        ast.Assign(
            targets=[ast.Name(id=e_to_raise, ctx=ast.Store())],
            value=to_raise_value,
        )
    )

    # if e_to_raise is not None: <throw e_to_raise>
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
            args=[ast.Name(id=e_to_raise, ctx=ast.Load())],
            keywords=[],
        )
    )
    # if e_to_raise is not None and the function/loop hasn't already
    # diverted (return / break swallows pending exceptions, matching
    # Python's `finally: return` semantics): throw e_to_raise.
    not_returned = ast.UnaryOp(
        op=ast.Not(),
        operand=ast.Attribute(
            value=helper_name, attr=frame.get_helper_member('returned'), ctx=ast.Load(),
        ),
    )
    throw_test_terms = [
        ast.Compare(
            left=ast.Name(id=e_to_raise, ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        not_returned,
    ]
    if frame.loops:
        loop_var_ref = ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
        throw_test_terms.append(
            ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Attribute(
                    value=loop_var_ref, attr=frame.get_helper_member('stopped'), ctx=ast.Load(),
                ),
            )
        )
    stmts.append(
        ast.If(
            test=ast.BoolOp(op=ast.And(), values=throw_test_terms),
            body=[throw_expr],
            orelse=[],
        )
    )
    # If the try body / a handler / else / finally did `return`, surface
    # that to the enclosing function's Or chain so the rest of the body
    # is short-circuited.
    stmts.append(make_return_propagator(frame))
    # If the try clause did `break` or `continue`, the truthy return
    # value of loop.stop() / loop.do_continue() was eaten at the per-
    # clause lambda boundary. Re-surface those by reading the loop
    # helper's flags. Both are emitted as plain expressions (returning
    # truthy from the test short-circuits the enclosing for-body Or
    # chain so the statements after this try don't run on this
    # iteration). Only meaningful when the try sits inside a loop.
    if frame.loops:
        loop_var_ref = ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
        stmts.append(
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Attribute(value=loop_var_ref, attr=frame.get_helper_member('stopped'), ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
        stmts.append(
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Attribute(value=loop_var_ref, attr=frame.get_helper_member('pending_continue'), ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
    return stmts


def parse_try_star(stmt: ast.TryStar, frame: Frame) -> list[_ast.AST]:
    return _parse_try_common(stmt, frame, dispatch_attr=frame.get_helper_member('dispatch_star'))


def parse_assert(stmt: ast.Assert, frame: Frame) -> list[_ast.AST]:
    return [
        ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=stmt.test),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='AssertionError', ctx=ast.Load()),
                        args=[stmt.msg] if stmt.msg is not None else [],
                        keywords=[],
                    )
                )
            ],
            orelse=[],
        )
    ]
