import _ast
import ast

from ..frame import Frame
from ..helpers import func_helper_name
from .utils import add_deco, strip_arg_annotations
from .pep695 import _wrap_pep695_def
from .dispatch import parse_stmts


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
                                            func=ast.Name(id=sub_frame.get_helper_name(func_helper_name), ctx=ast.Load()),
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
        from ..gen_compile import compile_generator
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
    sub_frame.boxed_names = getattr(stmt, '_boxed_names', None)
    return gen_func(stmt, sub_frame)


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
    if is_async_gen:
        # Wrap every user-level `yield V` (and `yield from X`) in
        # _UserYield(...) so the _AsyncGenWrapper.__anext__ coroutine
        # can distinguish user yields from intermediate yields that
        # come out of `await` (which we lower to `yield from
        # _await_iter(x)` further down). Done before await rewriting
        # so we can tell them apart.
        _wrap_user_yields(stmt.body, frame)
    # Lower async comprehensions sitting at the top of a stmt
    # (Return / Assign / Expr) into explicit async-for loops over
    # an accumulator. Doesn't handle async comp nested inside larger
    # expressions — those stay an unimplemented edge case.
    stmt.body = _lower_async_comps(stmt.body)
    stmt.body = _lower_async_constructs(stmt.body)
    _rewrite_await_in_body(stmt.body, frame)
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
    for attr in ('_use_legacy_return', '_box_helper_var',
                 '_gen_self_alias', '_gen_self_names'):
        if hasattr(stmt, attr):
            setattr(fdef, attr, getattr(stmt, attr))
    from ..gen_compile import compile_generator
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


def _is_async_comp(expr):
    if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return any(g.is_async for g in expr.generators)
    return False


def _lower_async_comp_to_stmts(comp, target_var):
    """Lower a list/set/dict/generator comprehension that uses
    `async for` somewhere into a sequence of statements that build
    the collection in `target_var`. The comp's generators may mix
    sync and async for; `async for` becomes a real async-for stmt,
    sync `for` stays sync; ifs become If guards. Nested comps inside
    the elt are not lowered here (they'd need their own pass)."""

    if isinstance(comp, ast.ListComp):
        init = ast.Assign(
            targets=[ast.Name(id=target_var, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        emit = lambda elt: ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target_var, ctx=ast.Load()),
                    attr='append', ctx=ast.Load(),
                ),
                args=[elt], keywords=[],
            )
        )
    elif isinstance(comp, ast.SetComp):
        init = ast.Assign(
            targets=[ast.Name(id=target_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='set', ctx=ast.Load()),
                args=[], keywords=[],
            ),
        )
        emit = lambda elt: ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target_var, ctx=ast.Load()),
                    attr='add', ctx=ast.Load(),
                ),
                args=[elt], keywords=[],
            )
        )
    elif isinstance(comp, ast.DictComp):
        init = ast.Assign(
            targets=[ast.Name(id=target_var, ctx=ast.Store())],
            value=ast.Dict(keys=[], values=[]),
        )
        emit = lambda kv: ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id=target_var, ctx=ast.Load()),
                slice=kv[0], ctx=ast.Store(),
            )],
            value=kv[1],
        )
    else:
        # GeneratorExp with async — Python's GeneratorExp doesn't
        # accept async-for in a sync function. Lower to a list (most
        # common consumer) — semantics differ slightly (eager vs
        # lazy) but it's the closest we can do without a custom
        # async-genexp class.
        init = ast.Assign(
            targets=[ast.Name(id=target_var, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        emit = lambda elt: ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target_var, ctx=ast.Load()),
                    attr='append', ctx=ast.Load(),
                ),
                args=[elt], keywords=[],
            )
        )

    if isinstance(comp, ast.DictComp):
        elt_payload = (comp.key, comp.value)
    else:
        elt_payload = comp.elt

    # Build nested for/async-for loops from generators.
    inner = [emit(elt_payload)]
    for gen in reversed(comp.generators):
        # if guards
        for if_ in reversed(gen.ifs):
            inner = [ast.If(test=if_, body=inner, orelse=[])]
        if gen.is_async:
            loop = ast.AsyncFor(
                target=gen.target, iter=gen.iter,
                body=inner, orelse=[],
            )
        else:
            loop = ast.For(
                target=gen.target, iter=gen.iter,
                body=inner, orelse=[],
            )
        inner = [loop]

    return [init] + inner


class _AsyncCompLifter(ast.NodeTransformer):
    """Lift async comprehensions embedded anywhere inside an expression
    into fresh pre-stmts. Doesn't descend into nested function/class/
    lambda bodies (those have their own lowering pass)."""

    def __init__(self, fresh):
        self._fresh = fresh
        self.pre_stmts: list = []

    def visit_ListComp(self, node):
        return self._lift_if_async(node)

    def visit_SetComp(self, node):
        return self._lift_if_async(node)

    def visit_DictComp(self, node):
        return self._lift_if_async(node)

    def visit_GeneratorExp(self, node):
        return self._lift_if_async(node)

    def _lift_if_async(self, node):
        self.generic_visit(node)
        if not _is_async_comp(node):
            return node
        tmp = self._fresh()
        self.pre_stmts.extend(_lower_async_comp_to_stmts(node, tmp))
        return ast.Name(id=tmp, ctx=ast.Load())

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node


def _lower_async_comps(body):
    """Walk body and lift async comprehensions out of any expression
    position into preceding statements."""
    out = []
    counter = [0]

    def fresh():
        counter[0] += 1
        return f'_acomp_{counter[0]}'

    def lift_expr(value):
        """Return (pre_stmts, new_value). pre_stmts may be empty."""
        lifter = _AsyncCompLifter(fresh)
        new_value = lifter.visit(value)
        return lifter.pre_stmts, new_value

    for stmt in body:
        # Statements with an expression value — lift from that value.
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            pre, stmt.value = lift_expr(stmt.value)
            out.extend(pre)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.Assign):
            pre, stmt.value = lift_expr(stmt.value)
            out.extend(pre)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.AugAssign):
            pre, stmt.value = lift_expr(stmt.value)
            out.extend(pre)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            pre, stmt.value = lift_expr(stmt.value)
            out.extend(pre)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.Expr):
            pre, stmt.value = lift_expr(stmt.value)
            out.extend(pre)
            out.append(stmt)
            continue
        # Compound stmts: recurse into bodies.
        if isinstance(stmt, ast.If):
            pre, stmt.test = lift_expr(stmt.test)
            out.extend(pre)
            stmt.body = _lower_async_comps(stmt.body)
            stmt.orelse = _lower_async_comps(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            pre, stmt.iter = lift_expr(stmt.iter)
            out.extend(pre)
            stmt.body = _lower_async_comps(stmt.body)
            stmt.orelse = _lower_async_comps(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.While):
            stmt.body = _lower_async_comps(stmt.body)
            stmt.orelse = _lower_async_comps(stmt.orelse)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.Try):
            stmt.body = _lower_async_comps(stmt.body)
            for h in stmt.handlers:
                h.body = _lower_async_comps(h.body)
            stmt.orelse = _lower_async_comps(stmt.orelse)
            stmt.finalbody = _lower_async_comps(stmt.finalbody)
            out.append(stmt)
            continue
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body = _lower_async_comps(stmt.body)
            out.append(stmt)
            continue
        out.append(stmt)
    return out


def _wrap_user_yields(body, frame):
    """Wrap every `yield V` and `yield from X` in the generator body
    with _UserYield(...) so the async-generator anext coroutine can
    distinguish them from yields produced by `await` lowering. Doesn't
    descend into nested def / class / lambda."""
    user_yield_name = frame.get_helper_name('_UserYield')
    class _R(ast.NodeTransformer):
        def visit_Yield(self, node):
            self.generic_visit(node)
            v = node.value if node.value is not None else ast.Constant(value=None)
            return ast.Yield(
                value=ast.Call(
                    func=ast.Name(id=user_yield_name, ctx=ast.Load()),
                    args=[v],
                    keywords=[],
                )
            )

        def visit_YieldFrom(self, node):
            # `yield from X` in an async generator iterates X and
            # yields each value as a user-level yield. We can't keep
            # YieldFrom directly because the wrapper would only see
            # the iter values without our marker. Lower it to a for
            # loop with explicit yield.
            self.generic_visit(node)
            # Use a fresh-ish name; we don't have a name_provider in
            # this transform's scope, so use one tied to id(node).
            it_var = '_yfrom_' + str(id(node))[-6:]
            elt = '_yfrom_e_' + str(id(node))[-6:]
            # Emit as a list of stmts wrapped via a synthetic
            # marker the caller will splice. Returning a list of
            # statements from a NodeTransformer requires returning
            # an Expr containing a synthetic indicator — instead,
            # we replace YieldFrom with a yielded Subscript of
            # ((iter_setup_expr_or_pass), yield_loop, None)[2] which
            # is awkward. Simpler: keep it as a YieldFrom for now
            # and wrap the value at runtime via a generator
            # comprehension inside the YieldFrom.
            return ast.YieldFrom(
                value=ast.GeneratorExp(
                    elt=ast.Call(
                        func=ast.Name(id=user_yield_name, ctx=ast.Load()),
                        args=[ast.Name(id='_v', ctx=ast.Load())],
                        keywords=[],
                    ),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id='_v', ctx=ast.Store()),
                            iter=node.value,
                            ifs=[],
                            is_async=0,
                        )
                    ],
                )
            )

        def visit_FunctionDef(self, node): return node
        def visit_AsyncFunctionDef(self, node): return node
        def visit_ClassDef(self, node): return node
        def visit_Lambda(self, node): return node

    r = _R()
    for i, s in enumerate(body):
        body[i] = r.visit(s)


def _rewrite_await_in_body(body, frame):
    """Walk the function body, replacing `await x` with
    `yield from _await_iter(x)` (the latter wrapped in YieldFrom so
    the existing generator pipeline sees a yield)."""
    await_iter_name = frame.get_helper_name('_await_iter')
    class _R(ast.NodeTransformer):
        def visit_Await(self, node):
            self.generic_visit(node)
            return ast.YieldFrom(
                value=ast.Call(
                    func=ast.Name(id=await_iter_name, ctx=ast.Load()),
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
