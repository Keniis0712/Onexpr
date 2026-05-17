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


def strip_arg_annotations(args: ast.arguments) -> dict:
    """Lambda doesn't accept annotated parameters; strip them in
    place. Returns a {param_name: annotation_expr} dict so the caller
    can re-attach them to the resulting function's __annotations__.
    The dict preserves Python's parameter order: posonly, args, vararg,
    kwonly, kwarg."""
    annotations = {}
    for group in (args.posonlyargs, args.args):
        for a in group:
            if a.annotation is not None:
                annotations[a.arg] = a.annotation
                a.annotation = None
    if args.vararg is not None:
        if args.vararg.annotation is not None:
            annotations[args.vararg.arg] = args.vararg.annotation
            args.vararg.annotation = None
    for a in args.kwonlyargs:
        if a.annotation is not None:
            annotations[a.arg] = a.annotation
            a.annotation = None
    if args.kwarg is not None:
        if args.kwarg.annotation is not None:
            annotations[args.kwarg.arg] = args.kwarg.annotation
            args.kwarg.annotation = None
    return annotations


def gen_func(stmt: ast.FunctionDef | ast.AsyncFunctionDef, sub_frame):
    temp_func_var = sub_frame.get_temp_var()
    arg_annotations = strip_arg_annotations(stmt.args)
    return_annotation = stmt.returns
    stmt.returns = None
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

    out = [
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
    ]
    # Reconstruct func.__annotations__ from the parameter annotations
    # we stripped + the return annotation so introspection (e.g.
    # typing.get_type_hints, dataclasses, pydantic) sees them.
    if arg_annotations or return_annotation is not None:
        ann_keys = []
        ann_values = []
        for k, v in arg_annotations.items():
            ann_keys.append(ast.Constant(value=k))
            ann_values.append(v)
        if return_annotation is not None:
            ann_keys.append(ast.Constant(value='return'))
            ann_values.append(return_annotation)
        out.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=temp_func_var, ctx=ast.Load()),
                        attr='__annotations__',
                    )
                ],
                value=ast.Dict(keys=ann_keys, values=ann_values),
            )
        )
    out.append(
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(
                stmt.name,
                stmt.decorator_list,
                ast.Name(id=temp_func_var, ctx=ast.Load()),
            )
        )
    )
    return out


def parse_function_def(stmt: ast.FunctionDef, frame: Frame) -> list[_ast.AST]:
    # PEP 695 generic function (`def f[T](...)`)? Wrap construction in
    # an outer function so type vars stay scoped. Inside the wrapper we
    # rebuild the same FunctionDef sans type_params, attach
    # __type_params__, then apply the decorators (the wrapper returns
    # the decorated callable). The outer assignment is what binds `f`.
    if getattr(stmt, 'type_params', None):
        return _wrap_pep695_def(stmt, frame, is_class=False)

    # Generator function (body has yield)? Compile to a state-machine
    # class instead of a lambda. We delegate to gen_compile, which
    # returns a list of statements (the class def + a forwarder
    # binding the user's name) — those are spliced back in and
    # processed normally by parse_stmts (so the class itself goes
    # through parse_class_def, etc.).
    if _generator_body_has_yield(stmt):
        from .gen_compile import compile_generator
        return compile_generator(stmt, frame)

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


def _wrap_pep695_def(stmt, frame: Frame, is_class: bool) -> list[_ast.AST]:
    """Rewrite a generic def/class declaration (PEP 695) into a wrapper
    function call. The wrapper isolates the type variables, assigns
    __type_params__ on the inner construct, applies decorators, and
    returns the decorated callable/class. The outer assignment binds
    the user's chosen name."""
    type_params = stmt.type_params
    decorators = list(stmt.decorator_list)
    name = stmt.name
    inner_temp = frame.get_temp_var()
    # Inner declaration: same node minus type_params and decorators,
    # bound to a temp name. The decorators get re-applied below after
    # __type_params__ is attached.
    inner_stmt = ast.copy_location(
        type(stmt)(
            **{f: getattr(stmt, f) for f in stmt._fields if f not in ('decorator_list', 'type_params', 'name')},
            name=inner_temp,
            decorator_list=[],
            type_params=[],
        ),
        stmt,
    )
    # Attach __type_params__ tuple
    type_params_tuple = ast.Tuple(
        elts=[ast.Name(id=tp.name, ctx=ast.Load()) for tp in type_params],
        ctx=ast.Load(),
    )
    inner_setup = [
        inner_stmt,
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__type_params__',
                    ctx=ast.Store(),
                )
            ],
            value=type_params_tuple,
        ),
        # Restore the user-visible name on the inner construct (we
        # built it with a temp name to avoid shadowing the outer
        # binding).
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__name__',
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=name),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__qualname__',
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=name),
        ),
    ]
    if is_class:
        # PEP 695 generic classes get an auto-generated __class_getitem__
        # so `Cls[int]` works. CPython does this in the compiler; we
        # have to emit it explicitly. types.GenericAlias matches what
        # the real PEP 695 path produces (modulo the exact class name
        # of the alias).
        inner_setup.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=inner_temp, ctx=ast.Load()),
                        attr='__class_getitem__',
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='classmethod', ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[
                                    ast.arg(arg='cls', annotation=None),
                                    ast.arg(arg='__p', annotation=None),
                                ],
                                vararg=None,
                                kwonlyargs=[], kw_defaults=[], defaults=[],
                            ),
                            body=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id='__import__', ctx=ast.Load()),
                                        args=[ast.Constant(value='types')],
                                        keywords=[],
                                    ),
                                    attr='GenericAlias',
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Name(id='cls', ctx=ast.Load()),
                                    ast.Name(id='__p', ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                        ),
                    ],
                    keywords=[],
                ),
            )
        )
    # Build the decorated return expression: deco_n(...deco_1(inner)...)
    return_expr = add_deco(
        name, decorators, ast.Name(id=inner_temp, ctx=ast.Load())
    )
    return _make_pep695_wrapper(
        name=name,
        type_params=type_params,
        inner_stmts=inner_setup,
        return_expr=return_expr,
        frame=frame,
    )


def _generator_body_has_yield(fdef) -> bool:
    """True iff fdef's body uses yield / yield from directly, not in
    a nested function or lambda."""
    class _V(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Lambda(self, node): pass
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_Yield(self, node): self.found = True
        def visit_YieldFrom(self, node): self.found = True

    v = _V()
    for s in fdef.body:
        v.visit(s)
    return v.found


def parse_async_function_def(stmt: ast.AsyncFunctionDef, frame: Frame) -> list[_ast.AST]:
    # Lower `async def f(args): BODY` to a regular generator function.
    # Each `await x` inside BODY (not crossing nested def/class/lambda)
    # becomes `yield from _await_iter(x)`. async for / async with also
    # get lowered to their await-based equivalents.
    #
    # If the user's body had a direct yield, this is an async
    # generator (PEP 525): we wrap the state-machine instance in an
    # _AsyncGenWrapper so it satisfies the async-iter protocol.
    # Otherwise the function is a coroutine: we wrap the forwarder
    # in types.coroutine. (See gen_compile.emit_state_machine for
    # the coroutine forwarder shape.)
    is_async_gen = _body_has_yield_or_yield_from(stmt.body)
    stmt.body = _lower_async_constructs(stmt.body)
    _rewrite_await_in_body(stmt.body)
    if not _body_has_yield_or_yield_from(stmt.body):
        stmt.body.insert(
            0,
            ast.If(
                test=ast.Constant(value=False),
                body=[ast.Expr(value=ast.Yield(value=None))],
                orelse=[],
            ),
        )
    fdef = ast.FunctionDef(
        name=stmt.name,
        args=stmt.args,
        body=stmt.body,
        decorator_list=stmt.decorator_list,
        returns=stmt.returns,
        type_params=getattr(stmt, 'type_params', []) or [],
    )
    for attr in ('_use_legacy_return', '_box_helper_var'):
        if hasattr(stmt, attr):
            setattr(fdef, attr, getattr(stmt, attr))
    from .gen_compile import compile_generator
    if is_async_gen:
        return compile_generator(fdef, frame, async_kind='gen')
    return compile_generator(fdef, frame, async_kind='coro')


def _lower_async_constructs(body):
    """Lower `async for` / `async with` to `await`-based equivalents.
    Recurses into compound statements but stops at nested function /
    class / lambda boundaries (they have their own scope; if any of
    them is an `async def`, parse_async_function_def will lower it
    when the time comes)."""
    out = []
    for stmt in body:
        if isinstance(stmt, ast.AsyncFor):
            out.extend(_lower_async_for(stmt))
            continue
        if isinstance(stmt, ast.AsyncWith):
            out.extend(_lower_async_with(stmt))
            continue
        if isinstance(stmt, ast.If):
            stmt.body = _lower_async_constructs(stmt.body)
            stmt.orelse = _lower_async_constructs(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.For):
            stmt.body = _lower_async_constructs(stmt.body)
            stmt.orelse = _lower_async_constructs(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.While):
            stmt.body = _lower_async_constructs(stmt.body)
            stmt.orelse = _lower_async_constructs(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.Try):
            stmt.body = _lower_async_constructs(stmt.body)
            for h in stmt.handlers:
                h.body = _lower_async_constructs(h.body)
            stmt.orelse = _lower_async_constructs(stmt.orelse)
            stmt.finalbody = _lower_async_constructs(stmt.finalbody)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.With):
            stmt.body = _lower_async_constructs(stmt.body)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = _lower_async_constructs(case.body)
            out.append(stmt)
            continue
        out.append(stmt)
    return out


def _lower_async_for(stmt: ast.AsyncFor) -> list:
    """`async for x in iter: BODY else: ELSE` becomes:

        _it = type(iter).__aiter__(iter)
        while True:
            try:
                x = await type(_it).__anext__(_it)
            except StopAsyncIteration:
                break
            BODY
        else:
            ELSE
    """
    it_var = '_aiter_' + str(id(stmt))[-6:]
    # _it = type(iter).__aiter__(iter)
    setup = ast.Assign(
        targets=[ast.Name(id=it_var, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='type', ctx=ast.Load()),
                    args=[stmt.iter],
                    keywords=[],
                ),
                attr='__aiter__', ctx=ast.Load(),
            ),
            args=[stmt.iter],
            keywords=[],
        ),
    )
    # x = await type(_it).__anext__(_it)
    next_call = ast.Await(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='type', ctx=ast.Load()),
                    args=[ast.Name(id=it_var, ctx=ast.Load())],
                    keywords=[],
                ),
                attr='__anext__', ctx=ast.Load(),
            ),
            args=[ast.Name(id=it_var, ctx=ast.Load())],
            keywords=[],
        ),
    )
    try_body = [
        ast.Assign(
            targets=[stmt.target],
            value=next_call,
        ),
    ]
    try_node = ast.Try(
        body=try_body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id='StopAsyncIteration', ctx=ast.Load()),
                name=None,
                body=[ast.Break()],
            ),
        ],
        orelse=[],
        finalbody=[],
    )
    while_body = [try_node] + _lower_async_constructs(stmt.body)
    while_node = ast.While(
        test=ast.Constant(value=True),
        body=while_body,
        orelse=_lower_async_constructs(stmt.orelse),
    )
    return [setup, while_node]


def _lower_async_with(stmt: ast.AsyncWith) -> list:
    """`async with X as v: BODY` lowers to PEP 492 equivalent using
    await for __aenter__ and __aexit__. Multiple items recurse."""
    if len(stmt.items) > 1:
        first, *rest = stmt.items
        inner = ast.AsyncWith(items=rest, body=stmt.body)
        return _lower_async_with(
            ast.AsyncWith(items=[first], body=[inner]),
        )
    item = stmt.items[0]
    mgr = '_amgr_' + str(id(stmt))[-6:]
    exc_flag = '_aexc_' + str(id(stmt))[-6:]
    e_name = '_ae_' + str(id(stmt))[-6:]

    setup = [
        ast.Assign(
            targets=[ast.Name(id=mgr, ctx=ast.Store())],
            value=item.context_expr,
        ),
    ]
    enter_call = ast.Await(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='type', ctx=ast.Load()),
                    args=[ast.Name(id=mgr, ctx=ast.Load())],
                    keywords=[],
                ),
                attr='__aenter__', ctx=ast.Load(),
            ),
            args=[ast.Name(id=mgr, ctx=ast.Load())],
            keywords=[],
        ),
    )
    if item.optional_vars is not None:
        setup.append(ast.Assign(
            targets=[item.optional_vars],
            value=enter_call,
        ))
    else:
        setup.append(ast.Expr(value=enter_call))
    setup.append(ast.Assign(
        targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
        value=ast.Constant(value=True),
    ))
    aexit = lambda *args: ast.Await(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='type', ctx=ast.Load()),
                    args=[ast.Name(id=mgr, ctx=ast.Load())],
                    keywords=[],
                ),
                attr='__aexit__', ctx=ast.Load(),
            ),
            args=[ast.Name(id=mgr, ctx=ast.Load())] + list(args),
            keywords=[],
        ),
    )
    handler_body = [
        ast.Assign(
            targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
            value=ast.Constant(value=False),
        ),
        ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=aexit(
                    ast.Call(
                        func=ast.Name(id='type', ctx=ast.Load()),
                        args=[ast.Name(id=e_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    ast.Name(id=e_name, ctx=ast.Load()),
                    ast.Attribute(
                        value=ast.Name(id=e_name, ctx=ast.Load()),
                        attr='__traceback__', ctx=ast.Load(),
                    ),
                ),
            ),
            body=[ast.Raise(
                exc=ast.Name(id=e_name, ctx=ast.Load()),
                cause=None,
            )],
            orelse=[],
        ),
    ]
    final_body = [
        ast.If(
            test=ast.Name(id=exc_flag, ctx=ast.Load()),
            body=[
                ast.Expr(value=aexit(
                    ast.Constant(value=None),
                    ast.Constant(value=None),
                    ast.Constant(value=None),
                )),
            ],
            orelse=[],
        ),
    ]
    try_node = ast.Try(
        body=_lower_async_constructs(stmt.body),
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id='BaseException', ctx=ast.Load()),
                name=e_name,
                body=handler_body,
            )
        ],
        orelse=[],
        finalbody=final_body,
    )
    return setup + [try_node]


def _rewrite_await_in_body(body):
    """Walk the function body, replacing `await x` with
    `yield from _await_iter(x)` (the latter wrapped in YieldFrom so
    the existing generator pipeline sees a yield)."""
    class _R(ast.NodeTransformer):
        def visit_Await(self, node):
            self.generic_visit(node)
            return ast.YieldFrom(
                value=ast.Call(
                    func=ast.Name(id='_await_iter', ctx=ast.Load()),
                    args=[node.value],
                    keywords=[],
                )
            )

        def visit_FunctionDef(self, node): return node
        def visit_AsyncFunctionDef(self, node): return node
        def visit_ClassDef(self, node): return node
        def visit_Lambda(self, node): return node

    r = _R()
    for i, s in enumerate(body):
        body[i] = r.visit(s)


def _body_has_yield_or_yield_from(body):
    class _V(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Yield(self, node): self.found = True
        def visit_YieldFrom(self, node): self.found = True
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

    v = _V()
    for s in body:
        v.visit(s)
    return v.found


def _collect_class_body_names(body: list) -> list:
    """Names that the user defines at class-body level (in source
    order, deduped). Skips any nested function / class / lambda
    bodies because those have their own scope. Used by
    parse_class_def to build an explicit dict for the metaclass
    instead of returning locals() (which would also include onexpr's
    helper temps)."""
    seen = set()
    out = []

    def add(name):
        if name not in seen:
            seen.add(name)
            out.append(name)

    def collect_target(t):
        if isinstance(t, ast.Name):
            add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                collect_target(e)
        elif isinstance(t, ast.Starred):
            collect_target(t.value)

    class _Walker(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            add(node.name)
        def visit_AsyncFunctionDef(self, node):
            add(node.name)
        def visit_ClassDef(self, node):
            add(node.name)
        def visit_Lambda(self, node):
            pass

        def visit_Assign(self, node):
            for t in node.targets:
                collect_target(t)
            self.visit(node.value)

        def visit_AugAssign(self, node):
            collect_target(node.target)
            self.visit(node.value)

        def visit_AnnAssign(self, node):
            # Bare annotation `x: int` doesn't create a class
            # attribute; only when there's an initializer.
            if node.value is not None:
                collect_target(node.target)
                self.visit(node.value)

        def visit_For(self, node):
            collect_target(node.target)
            self.visit(node.iter)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_AsyncFor(self, node):
            self.visit_For(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars is not None:
                    collect_target(item.optional_vars)
                self.visit(item.context_expr)
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node):
            self.visit_With(node)

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_Import(self, node):
            for alias in node.names:
                add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                add(alias.asname if alias.asname else alias.name)

    walker = _Walker()
    for stmt in body:
        walker.visit(stmt)
    return out


def parse_class_def(stmt: ast.ClassDef, frame: Frame) -> list[_ast.AST]:
    if getattr(stmt, 'type_params', None):
        return _wrap_pep695_def(stmt, frame, is_class=True)

    sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[])
    sub_frame.is_class_body = True
    sub_frame.legacy_return = (
        frame.legacy_return or getattr(stmt, '_use_legacy_return', False)
    )
    if not sub_frame.legacy_return:
        # If the nonlocal pass marked this class body as the owner of
        # one or more boxed names (try-clause assignments inside the
        # class body), reuse that exact helper var name so the rewritten
        # Attribute(Name('temp_N'), '_b_x') references stay correct.
        box_var = getattr(stmt, '_box_helper_var', None)
        sub_frame.func_helper_var = box_var if box_var is not None else sub_frame.get_temp_var()
    helper_var = sub_frame.func_helper_var
    cls_body = stmt.body

    # Class body needs an __annotations__ dict so parse_ann_assign can
    # write to it. Prepend `__annotations__ = {}` at the top of the
    # body so it's available before any annotated assignments.
    cls_body.insert(
        0,
        ast.Assign(
            targets=[ast.Name(id='__annotations__', ctx=ast.Store())],
            value=ast.Dict(keys=[], values=[]),
        )
    )

    # If the nonlocal pass boxed any names assigned inside try clauses
    # of this class body, those writes went to helper._b_<name>. We
    # reverse-copy each one into a regular local at the end of the
    # body, before we extract the user namespace, so the namespace
    # ends up with the right keys.
    class_boxed_names = getattr(stmt, '_class_boxed_names', None) or set()
    for name in sorted(class_boxed_names):
        cls_body.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(id=helper_var, ctx=ast.Load()),
                    attr='_b_' + name,
                    ctx=ast.Load(),
                ),
            )
        )

    # Collect the names the user actually defined at class-body level
    # so we can return them as an explicit dict instead of `locals()`.
    # `locals()` would include all of onexpr's helper temps (the
    # _FuncHelper instance, intermediate temp_N vars, etc.) and feeding
    # those into a metaclass like Enum's would either trip its name
    # validation or pollute the resulting class with garbage.
    user_names = _collect_class_body_names(cls_body)
    # Always include __annotations__ in the returned dict so the class
    # gets it even if the user didn't write any annotated assignments.
    if '__annotations__' not in user_names:
        user_names.append('__annotations__')

    # Replace the trailing Return(locals()) with Return({name: name, ...}).
    cls_body.append(
        ast.Return(
            value=ast.Dict(
                keys=[ast.Constant(value=n) for n in user_names],
                values=[ast.Name(id=n, ctx=ast.Load()) for n in user_names],
            )
        )
    )

    metaclass = ast.Name(id='type', ctx=ast.Load())
    for kwd in stmt.keywords:
        if kwd.arg == 'metaclass':
            metaclass = kwd.value
            break

    if sub_frame.legacy_return:
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

    # Build the metaclass instantiation. The runtime helper
    # _make_class handles everything: picking the right metaclass
    # (user override > most-derived from bases > type), running its
    # __prepare__ to get the proper namespace, populating it from
    # the body dict, and finally calling the metaclass.
    #
    # Internal helper classes (legacy_return mode) skip _make_class:
    # _make_class itself lives in the same helper module, so calling
    # it from there would be a forward reference. Plain
    # type(name, bases, dict) is correct for them since they don't
    # have custom metaclasses or weird namespace requirements.
    if sub_frame.legacy_return:
        body_call = ast.Call(
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
        )
        construct = ast.Call(
            func=metaclass,
            args=[
                ast.Constant(stmt.name),
                ast.Tuple(elts=stmt.bases, ctx=ast.Load()),
                body_call,
            ],
            keywords=[],
        )
        return [
            ast.Assign(
                targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
                value=add_deco(stmt.name, stmt.decorator_list, construct),
            )
        ]

    body_call = ast.Call(
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
    )
    construct = ast.Call(
        func=ast.Name(id='_make_class', ctx=ast.Load()),
        args=[
            metaclass,
            ast.Constant(stmt.name),
            ast.Tuple(elts=list(stmt.bases), ctx=ast.Load()),
            body_call,
        ],
        keywords=[],
    )

    return [
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(stmt.name, stmt.decorator_list, construct),
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
            # del x: best-effort. _del_local handles both the module-
            # level case (deletes from globals) and the function/lambda
            # local case (sets the slot to None on Python 3.13+ via the
            # PEP 667 f_locals proxy; on earlier versions the local
            # case is a silent no-op). Either way, the original
            # NameError-on-subsequent-access semantics can't be
            # reproduced — once a slot exists in fast locals there's
            # no Python-level way to unbind it.
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='_del_local', ctx=ast.Load()),
                    args=[ast.Constant(value=target.id)],
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


def _make_typevar_call(tp: ast.AST) -> ast.expr:
    """Build a typing.TypeVar / ParamSpec / TypeVarTuple constructor
    call for a PEP 695 type-param node."""
    typing_mod = ast.Call(
        func=ast.Name(id='__import__', ctx=ast.Load()),
        args=[ast.Constant(value='typing')],
        keywords=[],
    )
    if isinstance(tp, ast.TypeVar):
        kind = 'TypeVar'
        kwargs = []
        # `T: bound` — single bound expression. `T: (a, b)` — constraints.
        if tp.bound is not None:
            if isinstance(tp.bound, ast.Tuple):
                # constraints: pass as positional args after the name
                args = [ast.Constant(value=tp.name)] + list(tp.bound.elts)
            else:
                args = [ast.Constant(value=tp.name)]
                kwargs.append(ast.keyword(arg='bound', value=tp.bound))
        else:
            args = [ast.Constant(value=tp.name)]
        # PEP 695 type vars are implicitly variance-inferring (which
        # also drops the leading ~/+/- in repr); typing.TypeVar's
        # default is invariant.
        kwargs.append(ast.keyword(arg='infer_variance', value=ast.Constant(value=True)))
        # PEP 696 default
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    elif isinstance(tp, ast.ParamSpec):
        kind = 'ParamSpec'
        args = [ast.Constant(value=tp.name)]
        kwargs = []
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    elif isinstance(tp, ast.TypeVarTuple):
        kind = 'TypeVarTuple'
        args = [ast.Constant(value=tp.name)]
        kwargs = []
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    else:
        raise NotImplementedError(f'PEP 695 type param: {type(tp).__name__}')
    return ast.Call(
        func=ast.Attribute(value=typing_mod, attr=kind, ctx=ast.Load()),
        args=args,
        keywords=kwargs,
    )


def _make_pep695_wrapper(
    name: str,
    type_params: list,
    inner_stmts: list,
    return_expr: ast.expr,
    frame: Frame,
) -> list[_ast.AST]:
    """Emit the standard PEP 695 wrapper:

        def <wrapper>():
            T = typing.TypeVar('T')   # for each type param
            ...
            <inner_stmts>             # caller-provided body
            return <return_expr>
        <name> = <wrapper>()

    The wrapper isolates the type vars in its own scope so they don't
    leak into the surrounding namespace, and gives the alias / def /
    class body's RHS access to them via closure."""
    wrapper_name = frame.get_temp_var()
    body = []
    for tp in type_params:
        body.append(
            ast.Assign(
                targets=[ast.Name(id=tp.name, ctx=ast.Store())],
                value=_make_typevar_call(tp),
            )
        )
    body.extend(inner_stmts)
    body.append(ast.Return(value=return_expr))
    return [
        ast.FunctionDef(
            name=wrapper_name,
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_params=[],
        ),
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=wrapper_name, ctx=ast.Load()),
                args=[], keywords=[],
            ),
        ),
    ]


def parse_type_alias(stmt: ast.TypeAlias, frame: Frame) -> list[_ast.AST]:
    # PEP 695 `type X[T, ...] = expr`. We wrap the construction in a
    # zero-arg function so the type vars stay scoped, then build a
    # _LazyAlias whose `__value__` is computed lazily from a thunk
    # (so RHS forward-references / side-effects match Python's real
    # PEP 695 semantics).
    #
    #     def _wrap():
    #         T = typing.TypeVar('T')
    #         return _LazyAlias('X', lambda: <expr>, type_params=(T,))
    #     X = _wrap()
    type_param_names = [tp.name for tp in stmt.type_params]
    thunk = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=stmt.value,
    )
    type_params_tuple = ast.Tuple(
        elts=[ast.Name(id=n, ctx=ast.Load()) for n in type_param_names],
        ctx=ast.Load(),
    )
    alias_construct = ast.Call(
        func=ast.Name(id='_LazyAlias', ctx=ast.Load()),
        args=[
            ast.Constant(value=stmt.name.id),
            thunk,
        ],
        keywords=[ast.keyword(arg='type_params', value=type_params_tuple)],
    )
    return _make_pep695_wrapper(
        name=stmt.name.id,
        type_params=stmt.type_params,
        inner_stmts=[],
        return_expr=alias_construct,
        frame=frame,
    )


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
    # `x: int [= v]` at module level or in a class body writes to
    # __annotations__. Inside a function body, the annotation is
    # ignored at runtime (it's only for static checkers).
    should_annotate = (
        frame.prev is None  # module level
        or frame.is_class_body
    ) and isinstance(stmt.target, ast.Name)

    out = []
    if should_annotate:
        # __annotations__['x'] = int
        out.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='__annotations__', ctx=ast.Load()),
                        attr='__setitem__',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=stmt.target.id),
                        stmt.annotation,
                    ],
                    keywords=[],
                )
            )
        )
    if stmt.value is None:
        # Bare annotation `x: int` with no value. If we wrote to
        # __annotations__, that's all. Otherwise it's a no-op.
        if not out:
            out.append(ast.Constant(value=False))
        return out
    # `x: int = v` -> also do `x = v`.
    out.append(ast.Assign(targets=[stmt.target], value=stmt.value))
    return out


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
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Attribute(value=helper, attr='was_iterated', ctx=ast.Load()),
                bind,
                ast.Constant(value=False),
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
                    value=ast.Name(id=try_helper_name, ctx=ast.Load()),
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
    from .match_patterns import compile_match
    return compile_match(stmt, frame)


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
    return _parse_try_common(stmt, frame, dispatch_attr='dispatch')


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
                    value=ast.Name(id=try_helper_name, ctx=ast.Load()),
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
                        value=ast.Name(id=try_helper_name, ctx=ast.Load()),
                        attr='guarded',
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
                    ast.Attribute(value=loop_var_ref, attr='stopped', ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
        stmts.append(
            ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Attribute(value=loop_var_ref, attr='pending_continue', ctx=ast.Load()),
                    ast.Constant(value=True),
                ],
            )
        )
    return stmts


def parse_try_star(stmt: ast.TryStar, frame: Frame) -> list[_ast.AST]:
    return _parse_try_common(stmt, frame, dispatch_attr='dispatch_star')


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
            # `import a.b.c` binds the top-level package `a` to the
            # current scope. __import__('a.b.c') already returns `a`.
            top = name.split('.')[0]
            stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=top, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id='__import__', ctx=ast.Load()),
                        args=[ast.Constant(value=name)],
                        keywords=[],
                    ),
                )
            )
        else:
            # `import a.b.c as alias` binds the deepest submodule. We
            # must walk the attribute chain because __import__('a.b.c')
            # returns `a`, not `a.b.c`.
            value = ast.Call(
                func=ast.Name(id='__import__', ctx=ast.Load()),
                args=[ast.Constant(value=name)],
                keywords=[],
            )
            for part in name.split('.')[1:]:
                value = ast.Attribute(
                    value=value, attr=part, ctx=ast.Load(),
                )
            stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=as_name, ctx=ast.Store())],
                    value=value,
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
    # Sets the loop helper's pending_continue flag and returns True.
    # The truthy return value still short-circuits the surrounding Or
    # chain (so `if cond: continue\nrest` skips `rest` for that
    # iteration). The flag is what lets us cross lambda boundaries: a
    # `continue` inside a try body lambda becomes invisible to the
    # outer for-body Or chain unless something explicitly checks
    # loop.pending_continue, which parse_try does.
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='do_continue',
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
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
