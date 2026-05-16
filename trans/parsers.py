import _ast
import ast
from typing import Callable

from .frame import Frame
from .helpers import for_helper_name, while_helper_name, func_helper_name, try_helper_name


def add_deco(name: str, decorators: list[ast.expr], func: _ast.expr) -> ast.expr:
    node = func
    for decorator in reversed(decorators):
        node = ast.Call(
            func=decorator,
            args=[node],
            keywords=[],
        )
    return node


def slice_to_callable(key: ast.expr) -> ast.expr:
    """If key is an ast.Slice, convert it to a slice(...) call so it can
    be passed as a function argument (raw `1:3` syntax is only valid
    inside subscript brackets)."""
    if isinstance(key, ast.Slice):
        return ast.Call(
            func=ast.Name(id='slice', ctx=ast.Load()),
            args=[
                key.lower if key.lower is not None else ast.Constant(value=None),
                key.upper if key.upper is not None else ast.Constant(value=None),
                key.step if key.step is not None else ast.Constant(value=None),
            ],
            keywords=[],
        )
    return key


def strip_arg_annotations(args: ast.arguments) -> None:
    """Lambda doesn't accept annotated parameters; strip them in place."""
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            a.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None


def gen_func(stmt: ast.FunctionDef | ast.AsyncFunctionDef, sub_frame):
    temp_func_var = sub_frame.get_temp_var()
    strip_arg_annotations(stmt.args)
    if sub_frame.legacy_return:
        # Legacy mode: append `return None` to the body so the body's Or
        # chain ends with a (value, True) tuple, then take [0] in the
        # lambda. Used by the internal helper classes themselves.
        body = stmt.body
        if not body or not isinstance(body[-1], ast.Return):
            body = body + [ast.Return(value=ast.Constant(value=None))]
        body_or = parse_stmts(body, frame=sub_frame)
        lambda_body = ast.Subscript(
            value=body_or,
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
    else:
        helper_var = sub_frame.func_helper_var
        body_or = parse_stmts(stmt.body, frame=sub_frame)
        lambda_body = ast.Subscript(
            value=ast.Tuple(
                elts=[
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.BoolOp(
                                op=ast.And(),
                                values=[
                                    ast.NamedExpr(
                                        target=ast.Name(id=helper_var, ctx=ast.Store()),
                                        value=ast.Call(
                                            func=ast.Name(id=func_helper_name, ctx=ast.Load()),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                    ast.Constant(value=False),
                                ],
                            ),
                            body_or,
                        ],
                    ),
                    ast.Attribute(
                        value=ast.Name(id=helper_var, ctx=ast.Load()),
                        attr='value',
                        ctx=ast.Load(),
                    ),
                ],
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=1),
            ctx=ast.Load(),
        )
    return [
        ast.Assign(
            targets=[ast.Name(id=temp_func_var, ctx=ast.Store())],
            value=ast.Lambda(
                args=stmt.args,
                body=lambda_body,
            )
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=temp_func_var, ctx=ast.Load()),
                    attr='__name__'
                )
            ],
            value=ast.Constant(value=stmt.name)
        ),
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(
                stmt.name,
                stmt.decorator_list,
                ast.Name(id=temp_func_var, ctx=ast.Load()),
            )
        )
    ]


def parse_function_def(stmt: ast.FunctionDef, frame: Frame) -> list[_ast.AST]:
    sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[])
    # Inherit legacy mode from the enclosing frame, or pick it up from a
    # transform-time annotation on the FunctionDef itself.
    sub_frame.legacy_return = (
        frame.legacy_return or getattr(stmt, '_use_legacy_return', False)
    )
    if not sub_frame.legacy_return:
        # If the nonlocal pre-pass marked this function as the owner of one
        # or more boxed variables, reuse that exact helper var name so the
        # rewritten Attribute(Name('temp_N'), 'x') references stay correct.
        box_var = getattr(stmt, '_box_helper_var', None)
        sub_frame.func_helper_var = box_var if box_var is not None else sub_frame.get_temp_var()
    return gen_func(stmt, sub_frame)


def parse_async_function_def(stmt: ast.AsyncFunctionDef, frame: Frame) -> list[_ast.AST]:
    return None


def parse_class_def(stmt: ast.ClassDef, frame: Frame) -> list[_ast.AST]:
    sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[])
    sub_frame.legacy_return = (
        frame.legacy_return or getattr(stmt, '_use_legacy_return', False)
    )
    if not sub_frame.legacy_return:
        sub_frame.func_helper_var = sub_frame.get_temp_var()
    helper_var = sub_frame.func_helper_var
    cls_body = stmt.body
    cls_body.append(
        ast.Return(
            value=ast.Call(
                func=ast.Name(id='locals', ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )
    )
    metaclass = ast.Name(id='type', ctx=ast.Load())
    for kwd in stmt.keywords:
        if kwd.arg == 'metaclass':
            metaclass = kwd.value
            break

    if sub_frame.legacy_return:
        # Legacy: body's Or chain ends with (locals(), True); take [0].
        body_or = parse_stmts(cls_body, frame=sub_frame)
        cls_lambda_body = ast.Subscript(
            value=body_or,
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
    else:
        body_or = parse_stmts(cls_body, frame=sub_frame)
        cls_lambda_body = ast.Subscript(
            value=ast.Tuple(
                elts=[
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.BoolOp(
                                op=ast.And(),
                                values=[
                                    ast.NamedExpr(
                                        target=ast.Name(id=helper_var, ctx=ast.Store()),
                                        value=ast.Call(
                                            func=ast.Name(id=func_helper_name, ctx=ast.Load()),
                                            args=[],
                                            keywords=[],
                                        ),
                                    ),
                                    ast.Constant(value=False),
                                ],
                            ),
                            body_or,
                        ],
                    ),
                    ast.Attribute(
                        value=ast.Name(id=helper_var, ctx=ast.Load()),
                        attr='value',
                        ctx=ast.Load(),
                    ),
                ],
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=1),
            ctx=ast.Load(),
        )

    return [
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(
                stmt.name,
                stmt.decorator_list,
                ast.Call(
                    func=metaclass,
                    args=[
                        ast.Constant(stmt.name),
                        ast.Tuple(elts=stmt.bases, ctx=ast.Load()),
                        ast.Call(
                            func=ast.Lambda(
                                args=ast.arguments(
                                    posonlyargs=[],
                                    args=[],
                                    vararg=None,
                                    kwonlyargs=[],
                                    kw_defaults=[],
                                    defaults=[],
                                ),
                                body=cls_lambda_body,
                            ),
                            args=[],
                            keywords=[],
                        ),
                    ],
                    keywords=[],
                )
            )
        )
    ]


def parse_return(stmt: ast.Return, frame: Frame) -> list[_ast.AST]:
    value = stmt.value if stmt.value is not None else ast.Constant(value=None)
    if frame.legacy_return:
        # Old style: produce a (value, True) tuple so the body's Or
        # chain short-circuits truthy. gen_func's `[0]` extraction
        # picks `value` out at the lambda boundary.
        return [
            ast.Tuple(
                elts=[value, ast.Constant(value=True)],
                ctx=ast.Load(),
            )
        ]
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.func_helper_var, ctx=ast.Load()),
                attr='do_return',
                ctx=ast.Load(),
            ),
            args=[value],
            keywords=[],
        )
    ]


def parse_delete(stmt: ast.Delete, frame: Frame) -> list[_ast.AST]:
    out = []
    for target in stmt.targets:
        if isinstance(target, ast.Attribute):
            # del obj.x -> delattr(obj, 'x')
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='delattr', ctx=ast.Load()),
                    args=[
                        target.value,
                        ast.Constant(value=target.attr),
                    ],
                    keywords=[],
                )
            ))
        elif isinstance(target, ast.Subscript):
            # del obj[k] -> obj.__delitem__(k); slice key needs to be
            # built explicitly because `1:3` syntax isn't valid as a
            # call argument.
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=target.value,
                        attr='__delitem__',
                        ctx=ast.Load(),
                    ),
                    args=[slice_to_callable(target.slice)],
                    keywords=[],
                )
            ))
        elif isinstance(target, ast.Name):
            # del x: best-effort. Drop from globals() so module-level usage
            # works. For names that live in a lambda's locals (function or
            # class body) this is a silent no-op — the original `del`
            # semantics around local NameError can't be reproduced inside
            # an expression.
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='globals', ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        attr='pop',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=target.id),
                        ast.Constant(value=None),
                    ],
                    keywords=[],
                )
            ))
        else:
            raise NotImplementedError(f"del target: {type(target).__name__}")
    return out


def parse_assign(stmt: ast.Assign, frame: Frame) -> list[_ast.AST]:
    if len(stmt.targets) != 1:
        # This is look like "a = b = 0"

        # First copy the value to a temp var
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[
                    ast.Name(id=temp_var, ctx=ast.Store())
                ],
                value=stmt.value,
            )
        ]

        # Then copy the temp var to each target
        assigns.extend(
            ast.Assign(
                targets=[target],
                value=ast.Name(id=temp_var, ctx=ast.Load())
            )
            for target in stmt.targets
        )

        return assigns

    target = stmt.targets[0]

    if isinstance(target, (ast.List, ast.Tuple)):
        # This is look like "a, b = c"
        # First materialize the RHS as a list so subscripting works for any
        # iterable (including iterators and instances with __iter__ but no
        # __getitem__).
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='list', ctx=ast.Load()),
                    args=[stmt.value],
                    keywords=[],
                ),
            )
        ]
        # Then get each element from the temp var

        # If there is a starred name, we have to get how many elements before and after it
        names = target.elts
        before_star = None
        starred_name = None

        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                before_star = pos

                assert isinstance(name.value, ast.Name)
                starred_name = name.value.id
                break

        if before_star is not None:
            after_star = len(names) - before_star - 1
            upper = ast.Constant(-after_star) if after_star > 0 else ast.Constant(value=None)
            assigns.append(
                ast.Assign(
                    targets=[ast.Name(id=starred_name, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Slice(
                            lower=ast.Constant(before_star),
                            upper=upper,
                            col_offset=0,
                            end_col_offset=None,
                            end_lineno=None,
                            lineno=0,
                        )
                    )
                )
            )

        # Finally we start to build the assigns
        parse_before_star = True
        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                # We already parsed it
                parse_before_star = False
                continue
            assigns.append(
                ast.Assign(
                    targets=[name],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Constant(pos if parse_before_star else pos - len(names)),
                    )
                )
            )
        return assigns

    if isinstance(target, (ast.Subscript, ast.Attribute)):
        if isinstance(target, ast.Subscript):
            return [
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=target.value,
                            attr='__setitem__',
                            ctx=ast.Load(),
                        ),
                        args=[
                            slice_to_callable(target.slice),
                            stmt.value,
                        ],
                        keywords=[],
                    )
                )
            ]
        # Attribute assignment: use builtin setattr() so it works on both
        # instances and class objects (cls.__setattr__('a', v) fails because
        # __setattr__ on a class is a metaclass method that needs 3 args).
        return [
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='setattr', ctx=ast.Load()),
                    args=[
                        target.value,
                        ast.Constant(target.attr),
                        stmt.value,
                    ],
                    keywords=[],
                )
            )
        ]

    # Now is a simple assign like "a = 1"
    assert isinstance(target, ast.Name)
    if target.id in frame.global_vars:
        # `global x` was declared in this scope; route writes through
        # globals() so they land in the module dict instead of the
        # generated lambda's locals.
        return [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='globals', ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        attr='__setitem__',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=target.id),
                        stmt.value,
                    ],
                    keywords=[],
                )
            )
        ]
    return [
        ast.Expr(
            value=ast.NamedExpr(
                target=target,
                value=stmt.value,
            )
        )
    ]


def parse_type_alias(stmt: ast.TypeAlias, frame: Frame) -> list[_ast.AST]:
    pass


def parse_aug_assign(stmt: ast.AugAssign, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Assign(
            targets=[stmt.target],
            value=ast.BinOp(
                left=stmt.target,
                op=stmt.op,
                right=stmt.value
            ),
        )
    ]


def parse_ann_assign(stmt: ast.AnnAssign, frame: Frame) -> list[_ast.AST]:
    if stmt.value is None:
        # `x: int` is a bare annotation. At runtime it's a no-op for the
        # value (annotations land in __annotations__ but we ignore that).
        return [ast.Constant(value=False)]
    # `x: int = v` -> `x = v`. Drop the annotation; let parse_assign handle
    # the rest (Name / Attribute / Subscript targets).
    return [ast.Assign(targets=[stmt.target], value=stmt.value)]


def parse_for(stmt: ast.For, frame: Frame) -> list[_ast.AST]:
    frame.enter_loop()
    stmts: list[_ast.AST] = [
        ast.Assign(
            targets=[ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=for_helper_name, ctx=ast.Load()),
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


def make_loop_var_escape(target, frame) -> ast.expr:
    """Surface the for-loop target to the enclosing scope.

    Python binds `i` after `for i in ...:` to the last iterated value
    (and leaves `i` unbound when the iterable was empty). The ListComp
    we rewrite to has its own scope, so we have to re-emit the binding
    explicitly. We only handle simple Name targets; tuple-unpack targets
    (`for a, b in ...`) silently keep the old non-leaking behavior.

    The expression must always evaluate falsy so it doesn't short-circuit
    the surrounding Or chain — even when last_yielded is a truthy value.
    """
    if not isinstance(target, ast.Name):
        # Fall back to a no-op (falsy) so it doesn't break the Or chain.
        return ast.Constant(value=False)
    helper = ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())
    return ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Attribute(value=helper, attr='was_iterated', ctx=ast.Load()),
            ast.NamedExpr(
                target=ast.Name(id=target.id, ctx=ast.Store()),
                value=ast.Attribute(value=helper, attr='last_yielded', ctx=ast.Load()),
            ),
            ast.Constant(value=False),
        ],
    )


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
                func=ast.Name(id=while_helper_name, ctx=ast.Load()),
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
    pass


def parse_async_with(stmt: ast.AsyncWith, frame: Frame) -> list[_ast.AST]:
    pass


def parse_match(stmt: ast.Match, frame: Frame) -> list[_ast.AST]:
    pass


def parse_raise(stmt: ast.Raise, frame: Frame) -> list[_ast.AST]:
    # Bare `raise` reraises the current handler's exception. We rewrite
    # `except E as e` to a lambda taking `e` as parameter and push the
    # parameter name onto frame.exc_stack while parsing the handler
    # body, so a bare raise here just throws that name.
    if stmt.exc is None:
        if not frame.exc_stack:
            raise NotImplementedError(
                "bare `raise` outside of an except clause is not supported"
            )
        exc_to_throw = ast.Name(id=frame.exc_stack[-1], ctx=ast.Load())
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
                    value=ast.Name(id=try_helper_name, ctx=ast.Load()),
                    attr='dispatch',
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
                        value=ast.Name(id=try_helper_name, ctx=ast.Load()),
                        attr='guarded',
                        ctx=ast.Load(),
                    ),
                    args=[finally_lambda],
                    keywords=[],
                ),
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
            value=helper_name, attr='returned', ctx=ast.Load(),
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
                    value=loop_var_ref, attr='stopped', ctx=ast.Load(),
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
    return stmts


def parse_try_star(stmt: ast.TryStar, frame: Frame) -> list[_ast.AST]:
    pass


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


def parse_import(stmt: ast.Import, frame: Frame) -> list[_ast.AST]:
    stmts = []
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            as_name = name.split('.')[0]
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=as_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='__import__', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=name),
                    ],
                    keywords=[],
                ),
            )
        )
    return stmts


def parse_import_from(stmt: ast.ImportFrom, frame: Frame) -> list[_ast.AST]:
    stmts = []
    module_var = frame.get_temp_var()
    stmts.append(
        ast.Assign(
            targets=[ast.Name(id=module_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='__import__', ctx=ast.Load()),
                args=[
                    ast.Constant(value=stmt.module),
                ],
                keywords=[
                    ast.keyword(
                        arg='fromlist',
                        value=ast.Constant(
                            [
                                stmt.module,
                            ]
                        ),
                    )
                ],
            ),
        )
    )
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            as_name = name
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=as_name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(id=module_var, ctx=ast.Load()),
                    attr=name,
                    ctx=ast.Load(),
                ),
            )
        )
    return stmts


def parse_global(stmt: ast.Global, frame: Frame) -> list[_ast.AST]:
    for name in stmt.names:
        if name not in frame.global_vars:
            frame.global_vars.append(name)
    return [ast.Constant(value=False)]


def parse_nonlocal(stmt: ast.Nonlocal, frame: Frame) -> list[_ast.AST]:
    # The nonlocal pre-pass already rewrote every read/write of these
    # names to go through the owner function's box, so the declaration
    # itself becomes a no-op at this point.
    return [ast.Constant(value=False)]


def parse_expr(stmt: ast.Expr, frame: Frame) -> list[_ast.AST]:
    if frame.in_async_def and isinstance(stmt.value, ast.Await):
        value = ast.YieldFrom(value=stmt.value.value)
    else:
        value = stmt.value
    return [
        ast.BoolOp(
            op=ast.And(),
            values=[
                value,
                ast.Constant(value=False),
            ]
        )
    ]


def parse_pass(_: ast.Pass, __: Frame) -> list[_ast.AST]:
    return [
        ast.Constant(value=False),
    ]


def parse_break(_: ast.Break, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='stop',
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    ]


def parse_continue(stmt: ast.Continue, frame: Frame) -> list[_ast.AST]:
    # Use Constant instead of Expr to avoid adding "and False"
    return [
        ast.Constant(value=True),
    ]


name2func: dict[str, Callable] = {
    "FunctionDef": parse_function_def,
    "AsyncFunctionDef": parse_async_function_def,
    "ClassDef": parse_class_def,
    "Return": parse_return,
    "Delete": parse_delete,
    "Assign": parse_assign,
    "TypeAlias": parse_type_alias,
    "AugAssign": parse_aug_assign,
    "AnnAssign": parse_ann_assign,
    "For": parse_for,
    "AsyncFor": parse_async_for,
    "While": parse_while,
    "If": parse_if,
    "With": parse_with,
    "AsyncWith": parse_async_with,
    "Match": parse_match,
    "Raise": parse_raise,
    "Try": parse_try,
    "TryStar": parse_try_star,
    "Assert": parse_assert,
    "Import": parse_import,
    "ImportFrom": parse_import_from,
    "Global": parse_global,
    "Nonlocal": parse_nonlocal,
    "Expr": parse_expr,
    "Pass": parse_pass,
    "Break": parse_break,
    "Continue": parse_continue,
}


def parse_stmt(stmt: ast.stmt, frame: Frame) -> list[ast.expr]:
    exprs = [stmt]

    while not all(isinstance(expr, ast.expr) for expr in exprs):
        new_exprs = []

        for expr in exprs:
            if isinstance(expr, ast.expr):
                new_exprs.append(expr)
                continue
            class_name = expr.__class__.__name__
            assert class_name in name2func

            parse_func = name2func[class_name]
            exprs = parse_func(expr, frame)
            if exprs is None:
                raise NotImplementedError(f"Not implemented for {class_name}")

            new_exprs.extend(exprs)

        exprs = new_exprs

    return exprs


def parse_stmts(stmts: list[ast.stmt], frame: Frame) -> _ast.BoolOp:
    exprs = []
    for stmt in stmts:
        expr = parse_stmt(stmt, frame)
        exprs.extend(expr)

    return ast.BoolOp(
        op=ast.Or(),
        values=exprs
    )
