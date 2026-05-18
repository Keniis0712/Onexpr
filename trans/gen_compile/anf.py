import ast

from .locals import _MiniFrame, _free_user_names


# ---------------------------------------------------------------------
# A-normal form — lift yield-in-expression to its own statement

class _ANFLifter(ast.NodeTransformer):
    """For every Yield / YieldFrom that appears in a sub-expression
    position (i.e. not as the entire .value of an Expr / Assign), lift
    it to a fresh `_yt_<n> = yield ...` assignment placed before the
    enclosing statement.

    We do this only inside a generator function body — caller passes
    the body list and we rewrite in place, returning a new list with
    extra prefix statements.

    Importantly, this lifter only descends into expression contexts.
    Compound statements (If / For / While) have their bodies handled
    by the outer anf_transform recursion; visiting them here would
    double-lift any yields in those sub-bodies."""

    def __init__(self, name_provider):
        self._name = name_provider
        self._prefix = []

    def visit_Yield(self, node):
        # Generic visit recurses into the yield's own value first.
        node.value = self.visit(node.value) if node.value else None
        # Lift this yield to a temp.
        tmp = self._name()
        self._prefix.append(
            ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=node,
            )
        )
        return ast.Name(id=tmp, ctx=ast.Load())

    def visit_YieldFrom(self, node):
        node.value = self.visit(node.value)
        tmp = self._name()
        self._prefix.append(
            ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=node,
            )
        )
        return ast.Name(id=tmp, ctx=ast.Load())

    def visit_NamedExpr(self, node):
        # Walrus inside an expression:
        #     ... (x := expr) ...
        # We don't lift it to a separate Assign because that would
        # break short-circuit semantics: a walrus in an `if` test or
        # `while` test gets re-evaluated each iteration, but a lifted
        # Assign before the loop fires only once. Instead we tell the
        # state-machine self-rewriter to leave both the target Name
        # and any references to that name alone, by treating the
        # walrus target as a send-local rather than a self.<name>
        # box. collect_user_locals already skips NamedExpr, so the
        # name simply isn't boxed and the walrus stays as-is.
        node.value = self.visit(node.value)
        return node

    def visit_Lambda(self, node):
        return node  # don't descend

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    # Compound statements: only visit their direct *expressions*
    # (test/iter/...), not their bodies — the outer anf_transform
    # is responsible for the bodies.
    def visit_If(self, node):
        node.test = self.visit(node.test)
        return node

    def visit_For(self, node):
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        return node

    def visit_AsyncFor(self, node):
        return self.visit_For(node)

    def visit_While(self, node):
        node.test = self.visit(node.test)
        return node

    def visit_With(self, node):
        return node  # phase 1 doesn't support try/with in generator

    def visit_AsyncWith(self, node):
        return node

    def visit_Try(self, node):
        return node


def _is_bare_yield_stmt(stmt):
    """True if stmt is `yield X` or `yield from X` as a top-level
    statement (i.e. an Expr wrapping a Yield/YieldFrom). Such yields
    don't need ANF lifting — they're already at the right level."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
    )


def _is_bare_yield_assign(stmt):
    """True if stmt is `x = yield ...` — also OK without lifting."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
    )


def anf_transform(body: list, name_provider, all_locals=None) -> list:
    """Walk a generator body, returning a new list of statements where
    every Yield / YieldFrom appears either as a top-level expression
    statement or as the entire RHS of a simple-name assignment.

    Recurses into If / For / While bodies but stops at nested function
    / class / lambda — those have their own scopes.

    `all_locals`, if given, is the full set of names that will be
    boxed onto self in the final state-machine class (computed by
    collect_user_locals on the original body). It's used only to
    dehydrate them as send-local variables right before a nested
    def/class so the nested scope's closure can reach them; without
    this, a `def adder(x): return base + x` inside a generator with
    `base` boxed would see no `base` at all (it's only on self)."""
    out = []

    def lift(stmt):
        # If this stmt is already in good form, leave its yield alone
        # but recurse into nested bodies.
        if _is_bare_yield_stmt(stmt) or _is_bare_yield_assign(stmt):
            return [stmt]

        lifter = _ANFLifter(name_provider)
        new_stmt = lifter.visit(stmt)
        return lifter._prefix + [new_stmt]

    for stmt in body:
        if isinstance(stmt, ast.Match):
            from ..match_patterns import compile_match
            fake_frame = _MiniFrame(name_provider)
            flat = compile_match(stmt, fake_frame)
            _mark_match_origin(flat)
            out.extend(anf_transform(flat, name_provider, all_locals))
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested def/class: emit dehydrate-from-self stmts first
            # so the nested scope's closure can capture boxed user
            # locals as cell variables. Then the def itself, then a
            # rebind that boxes the new function/class onto self.
            if all_locals:
                free = _free_user_names(stmt, all_locals)
                for name in sorted(free):
                    lhs = ast.Name(id=name, ctx=ast.Store())
                    lhs._gen_no_self = True
                    out.append(
                        ast.Assign(
                            targets=[lhs],
                            value=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=name,
                                ctx=ast.Load(),
                            ),
                        )
                    )
            out.append(stmt)
            rhs = ast.Name(id=stmt.name, ctx=ast.Load())
            rhs._gen_no_self = True
            out.append(
                ast.Assign(
                    targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
                    value=rhs,
                )
            )
            continue
        if isinstance(stmt, ast.If):
            if getattr(stmt, '_from_match', False):
                # Match-origin If: recurse into bodies but don't lift
                # the test, which contains short-circuit walrus
                # captures we must preserve.
                stmt.body = anf_transform(stmt.body, name_provider, all_locals)
                stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
                out.append(stmt)
                continue
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.For):
            # Tuple/list unpack target in a generator's for loop:
            # rewrite `for a, b in iter:` to
            #   `for _tmp in iter: a, b = _tmp; <body>`
            # so the state machine only ever sees a simple Name target.
            if not isinstance(stmt.target, ast.Name):
                tmp = name_provider()
                unpack = ast.Assign(
                    targets=[stmt.target],
                    value=ast.Name(id=tmp, ctx=ast.Load()),
                )
                stmt.body = [unpack] + stmt.body
                stmt.target = ast.Name(id=tmp, ctx=ast.Store())
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            # The .iter expression itself can contain yield — lift it.
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.While):
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.Try):
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            for h in stmt.handlers:
                if h.name is None:
                    # Bare `except E:` — give it a synthetic name so
                    # bare `raise` inside the handler can refer to it
                    # (parse_raise on the send method has no exc stack
                    # because it's a fresh function frame).
                    h.name = name_provider()
                _rewrite_bare_raise_to_named(h.body, h.name)
                h.body = anf_transform(h.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            stmt.finalbody = anf_transform(stmt.finalbody, name_provider, all_locals)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.With):
            # `with X as v: BODY` (possibly with multiple items) is
            # equivalent to nested withs. We lower to PEP 343 form
            # using try/except/finally so the existing CFG try
            # handling carries the with through yield boundaries.
            lowered = _lower_with_to_try(stmt, name_provider)
            out.extend(anf_transform(lowered, name_provider, all_locals))
            continue
        out.extend(lift(stmt))

    return out


def _rewrite_bare_raise_to_named(body: list, name: str):
    """In-place: every bare `raise` (no exc) in `body` becomes
    `raise <name>`. Doesn't descend into nested function/class/lambda
    or into another except handler (which would have its own name to
    bind to)."""
    class _R(ast.NodeTransformer):
        def visit_Raise(self, node):
            if node.exc is None:
                return ast.Raise(
                    exc=ast.Name(id=name, ctx=ast.Load()),
                    cause=node.cause,
                )
            return node

        def visit_FunctionDef(self, node): return node
        def visit_AsyncFunctionDef(self, node): return node
        def visit_ClassDef(self, node): return node
        def visit_Lambda(self, node): return node
        def visit_Try(self, node):
            # Body and finally still see this name as the active
            # exception. except handlers shadow with their own name
            # (bare raise inside a nested handler refers to that
            # nested except's exception, not ours).
            node.body = [self.visit(s) for s in node.body]
            node.orelse = [self.visit(s) for s in node.orelse]
            node.finalbody = [self.visit(s) for s in node.finalbody]
            # Don't descend into node.handlers — they re-enter
            # _rewrite_bare_raise_to_named with their own name in the
            # outer pass.
            return node

    r = _R()
    for i, s in enumerate(body):
        body[i] = r.visit(s)


def _lower_with_to_try(stmt: ast.With, name_provider) -> list:
    """Rewrite `with X as v: BODY` to PEP 343-equivalent try/except/
    finally so the generator CFG can carry the with across yields.

    For multiple items `with X, Y: BODY`, recurse: outer with X, inner
    with Y over BODY.
    """
    if len(stmt.items) > 1:
        first, *rest = stmt.items
        inner = ast.With(items=rest, body=stmt.body)
        return _lower_with_to_try(
            ast.With(items=[first], body=[inner]),
            name_provider,
        )
    item = stmt.items[0]
    mgr = name_provider()
    exc_flag = name_provider()
    e_name = name_provider()

    # mgr = ctx_expr
    setup = [
        ast.Assign(
            targets=[ast.Name(id=mgr, ctx=ast.Store())],
            value=item.context_expr,
        ),
    ]
    # __enter__ — bind to as-name if any.
    enter_call = ast.Call(
        func=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id='type', ctx=ast.Load()),
                args=[ast.Name(id=mgr, ctx=ast.Load())],
                keywords=[],
            ),
            attr='__enter__', ctx=ast.Load(),
        ),
        args=[ast.Name(id=mgr, ctx=ast.Load())],
        keywords=[],
    )
    if item.optional_vars is not None:
        setup.append(ast.Assign(
            targets=[item.optional_vars],
            value=enter_call,
        ))
    else:
        setup.append(ast.Expr(value=enter_call))
    # exc_flag = True
    setup.append(ast.Assign(
        targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
        value=ast.Constant(value=True),
    ))

    # except BaseException as e:
    #     exc_flag = False
    #     if not type(mgr).__exit__(mgr, type(e), e, e.__traceback__):
    #         raise
    handler_body = [
        ast.Assign(
            targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
            value=ast.Constant(value=False),
        ),
        ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='type', ctx=ast.Load()),
                            args=[ast.Name(id=mgr, ctx=ast.Load())],
                            keywords=[],
                        ),
                        attr='__exit__', ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=mgr, ctx=ast.Load()),
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
                    ],
                    keywords=[],
                ),
            ),
            body=[ast.Raise(
                exc=ast.Name(id=e_name, ctx=ast.Load()),
                cause=None,
            )],
            orelse=[],
        ),
    ]
    # finally:
    #     if exc_flag:
    #         type(mgr).__exit__(mgr, None, None, None)
    final_body = [
        ast.If(
            test=ast.Name(id=exc_flag, ctx=ast.Load()),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id='type', ctx=ast.Load()),
                                args=[ast.Name(id=mgr, ctx=ast.Load())],
                                keywords=[],
                            ),
                            attr='__exit__', ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(id=mgr, ctx=ast.Load()),
                            ast.Constant(value=None),
                            ast.Constant(value=None),
                            ast.Constant(value=None),
                        ],
                        keywords=[],
                    )
                ),
            ],
            orelse=[],
        ),
    ]
    try_node = ast.Try(
        body=stmt.body,
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


def _mark_match_origin(nodes):
    """Tag every If statement reachable from these nodes (without
    crossing nested function/class/lambda scopes) as match-origin so
    anf_transform skips lifting their tests."""

    class _M(ast.NodeVisitor):
        def visit_If(self, node):
            node._from_match = True
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)

        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

    m = _M()
    for n in nodes:
        m.visit(n)


# Import _self_name here for use in anf_transform (dehydrate block)
def _self_name(ctx=None):
    """Create a `self` Name marked so _SelfRewriter does not treat it
    as a user-level boxed local."""
    n = ast.Name(id='self', ctx=ctx if ctx is not None else ast.Load())
    n._gen_no_self = True
    return n
