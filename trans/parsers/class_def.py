import _ast
import ast

from ..frame import Frame
from ..helpers import func_helper_name
from .utils import add_deco
from .pep695 import _wrap_pep695_def
from .dispatch import parse_stmts


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

        def visit_NamedExpr(self, node):
            # `(y := expr)` at class-body scope binds y as a class
            # attribute. Pick up the target so the metaclass dict gets it.
            if isinstance(node.target, ast.Name):
                add(node.target.id)
            self.visit(node.value)

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


def _has_zero_arg_super(body: list) -> bool:
    """True iff the class body contains a zero-arg super() call at any
    depth (including inside method bodies). These need the __class__
    cell mechanism to work."""
    class _V(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name)
                    and node.func.id == 'super'
                    and len(node.args) == 0
                    and len(node.keywords) == 0):
                self.found = True
            self.generic_visit(node)
    v = _V()
    for s in body:
        v.visit(s)
        if v.found:
            return True
    return False


def _shadowed_in_class_body(body: list) -> list:
    """Names that the class body reads BEFORE it stores them.

    CPython class bodies have LOAD_NAME semantics — a read first
    consults the class namespace, then the enclosing scope, then
    builtins. So `class K: x = G; G = 'shadow'` reads outer G for x
    even though the body later defines its own G.

    We compile the class body into a lambda, where Python's LOAD_FAST
    rule treats any name with a store-anywhere as a function-local
    from the very start, and pre-store reads raise UnboundLocalError.
    To recover the LOAD_NAME-ish behaviour we lift each shadowed
    name into a keyword default of the lambda: `lambda *, G=G: ...`.
    The default expression evaluates in the *enclosing* scope at
    lambda-creation time (so it picks up the outer G), and the
    parameter is a regular local that the later store can overwrite.

    Returns the list of names in source order. Skips names that
    aren't read before they're stored (no shadowing problem) and
    names that are only stored.

    Doesn't descend into nested function / class / lambda bodies —
    they have their own scopes.
    """
    seen_store = set()
    needs_lift = []  # preserves source order
    seen_needs = set()

    def get_targets(t):
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            out = []
            for e in t.elts:
                out.extend(get_targets(e))
            return out
        if isinstance(t, ast.Starred):
            return get_targets(t.value)
        return []

    class _W(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # The function's name is a class-body store. The body
            # itself isn't visited (it has its own scope), but the
            # decorators, defaults, and annotations DO evaluate at
            # class-body level — so their reads count here.
            for d in node.decorator_list:
                self.visit(d)
            for d in node.args.defaults:
                self.visit(d)
            for d in node.args.kw_defaults:
                if d is not None:
                    self.visit(d)
            for group in (node.args.posonlyargs, node.args.args, node.args.kwonlyargs):
                for a in group:
                    if a.annotation is not None:
                        self.visit(a.annotation)
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                self.visit(node.args.kwarg.annotation)
            if node.returns is not None:
                self.visit(node.returns)
            seen_store.add(node.name)

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            # decorators / bases evaluate at outer-class-body level
            for d in node.decorator_list:
                self.visit(d)
            for b in node.bases:
                self.visit(b)
            for kw in node.keywords:
                self.visit(kw.value)
            seen_store.add(node.name)

        def visit_Lambda(self, node):
            pass

        def visit_Assign(self, node):
            self.visit(node.value)
            for t in node.targets:
                for n in get_targets(t):
                    seen_store.add(n)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id not in seen_store:
                if node.target.id not in seen_needs:
                    needs_lift.append(node.target.id)
                    seen_needs.add(node.target.id)
            self.visit(node.value)
            for n in get_targets(node.target):
                seen_store.add(n)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit(node.value)
            self.visit(node.annotation)
            for n in get_targets(node.target):
                seen_store.add(n)

        def visit_For(self, node):
            self.visit(node.iter)
            for n in get_targets(node.target):
                seen_store.add(n)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_AsyncFor(self, node):
            self.visit_For(node)

        def visit_With(self, node):
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    for n in get_targets(item.optional_vars):
                        seen_store.add(n)
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node):
            self.visit_With(node)

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                seen_store.add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_Import(self, node):
            for alias in node.names:
                seen_store.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                seen_store.add(alias.asname if alias.asname else alias.name)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                if node.id not in seen_store and node.id not in seen_needs:
                    needs_lift.append(node.id)
                    seen_needs.add(node.id)

        def visit_NamedExpr(self, node):
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                seen_store.add(node.target.id)

    w = _W()
    for s in body:
        w.visit(s)
    return needs_lift


def _function_body_free_names(fdef) -> set:
    """Names that a (def|async def|lambda) body reads from an
    enclosing scope. Approximate — we treat any Name(Load) that
    isn't a local store, parameter, comprehension target, or
    nested-scope owner as free."""
    if isinstance(fdef, ast.Lambda):
        body_iter = [fdef.body]
        params = fdef.args
    else:
        body_iter = fdef.body
        params = fdef.args

    locals_ = set()
    for group in (params.posonlyargs, params.args, params.kwonlyargs):
        for a in group:
            locals_.add(a.arg)
    if params.vararg is not None:
        locals_.add(params.vararg.arg)
    if params.kwarg is not None:
        locals_.add(params.kwarg.arg)

    free = set()

    def get_targets(t):
        if isinstance(t, ast.Name):
            return [t.id]
        if isinstance(t, (ast.Tuple, ast.List)):
            out = []
            for e in t.elts:
                out.extend(get_targets(e))
            return out
        if isinstance(t, ast.Starred):
            return get_targets(t.value)
        return []

    class _W(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            locals_.add(node.name)
        def visit_AsyncFunctionDef(self, node):
            locals_.add(node.name)
        def visit_ClassDef(self, node):
            locals_.add(node.name)
        def visit_Lambda(self, node):
            # Don't recurse — names captured into nested lambdas
            # are still free in this scope unless this scope binds
            # them too. We only want this scope's reads/stores.
            pass

        def visit_Assign(self, node):
            self.visit(node.value)
            for t in node.targets:
                for n in get_targets(t):
                    locals_.add(n)

        def visit_AugAssign(self, node):
            self.visit(node.target)  # AugAssign reads first
            self.visit(node.value)
            for n in get_targets(node.target):
                locals_.add(n)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit(node.value)
            for n in get_targets(node.target):
                locals_.add(n)

        def visit_For(self, node):
            self.visit(node.iter)
            for n in get_targets(node.target):
                locals_.add(n)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_AsyncFor(self, node):
            self.visit_For(node)

        def visit_With(self, node):
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    for n in get_targets(item.optional_vars):
                        locals_.add(n)
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node):
            self.visit_With(node)

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                locals_.add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_Import(self, node):
            for alias in node.names:
                locals_.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                locals_.add(alias.asname if alias.asname else alias.name)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                if node.id not in locals_:
                    free.add(node.id)

        def visit_NamedExpr(self, node):
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                locals_.add(node.target.id)

    w = _W()
    for s in body_iter:
        w.visit(s)
    return free


def _rewrite_loads_in_function(fdef, name_map):
    """Rewrite Name(Load, id=k) to Name(Load, id=name_map[k]) inside
    fdef's body, only for names that aren't shadowed by a local store
    in the same scope. Doesn't descend into nested function / lambda /
    class scopes (they have their own free-var resolution)."""
    if not name_map:
        return

    # Compute this scope's local stores so we don't rewrite a Load
    # that resolves locally.
    if isinstance(fdef, ast.Lambda):
        body_iter = [fdef.body]
        params = fdef.args
    else:
        body_iter = fdef.body
        params = fdef.args

    local = set()
    for group in (params.posonlyargs, params.args, params.kwonlyargs):
        for a in group:
            local.add(a.arg)
    if params.vararg is not None:
        local.add(params.vararg.arg)
    if params.kwarg is not None:
        local.add(params.kwarg.arg)

    def collect_targets(t):
        if isinstance(t, ast.Name):
            local.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                collect_targets(e)
        elif isinstance(t, ast.Starred):
            collect_targets(t.value)

    class _Pre(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            local.add(node.name)
        def visit_AsyncFunctionDef(self, node):
            local.add(node.name)
        def visit_ClassDef(self, node):
            local.add(node.name)
        def visit_Lambda(self, node): pass
        def visit_Assign(self, node):
            for t in node.targets:
                collect_targets(t)
            self.visit(node.value)
        def visit_AugAssign(self, node):
            collect_targets(node.target)
            self.visit(node.value)
        def visit_AnnAssign(self, node):
            collect_targets(node.target)
            if node.value is not None:
                self.visit(node.value)
        def visit_For(self, node):
            collect_targets(node.target)
            self.visit(node.iter)
            for s in node.body + node.orelse:
                self.visit(s)
        def visit_AsyncFor(self, node):
            self.visit_For(node)
        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars is not None:
                    collect_targets(item.optional_vars)
                self.visit(item.context_expr)
            for s in node.body:
                self.visit(s)
        def visit_AsyncWith(self, node):
            self.visit_With(node)
        def visit_ExceptHandler(self, node):
            if node.name is not None:
                local.add(node.name)
            for s in node.body:
                self.visit(s)
        def visit_Import(self, node):
            for alias in node.names:
                local.add(alias.asname if alias.asname else alias.name.split('.')[0])
        def visit_ImportFrom(self, node):
            for alias in node.names:
                local.add(alias.asname if alias.asname else alias.name)
        def visit_NamedExpr(self, node):
            if isinstance(node.target, ast.Name):
                local.add(node.target.id)
            self.visit(node.value)

    pre = _Pre()
    for s in body_iter:
        pre.visit(s)

    class _Rewriter(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return node  # don't recurse
        def visit_AsyncFunctionDef(self, node):
            return node
        def visit_ClassDef(self, node):
            return node
        def visit_Lambda(self, node):
            return node
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in name_map and node.id not in local:
                return ast.Name(id=name_map[node.id], ctx=node.ctx)
            return node

    rw = _Rewriter()
    for i, s in enumerate(body_iter):
        new = rw.visit(s)
        if isinstance(fdef, ast.Lambda):
            fdef.body = new
        else:
            fdef.body[i] = new


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

    # Detect class-body names the user reads before they're stored.
    # CPython class bodies have LOAD_NAME semantics — a read first
    # consults the class namespace, then the enclosing scope, then
    # builtins — so `class K: x = G; G = 'shadow'` reads outer G
    # for x even though the body later defines its own G. Our
    # lambda-based representation would treat such names as locals
    # from the start (LOAD_FAST) and raise UnboundLocalError. Lift
    # those names as keyword-only defaults of the body lambda; the
    # default expression evaluates in the *enclosing* scope at
    # lambda-creation time, so the read picks up the outer value.
    shadowed = _shadowed_in_class_body(cls_body)
    # Names that the nonlocal pass boxed onto the helper (try-clause
    # writes) shouldn't be lifted — their reads are rewritten to
    # helper._b_<name>, not to a plain Name.
    if shadowed:
        boxed = getattr(stmt, '_class_boxed_names', None) or set()
        shadowed = [n for n in shadowed if n not in boxed]

    # Methods inside a class body shouldn't capture sibling class
    # members (CPython: function defs in a class body don't see the
    # class namespace; their free var lookups go straight to the
    # enclosing module/function scope). Our lambda-based class body
    # binds methods as lambda locals, so a nested method's lambda
    # would closure-capture them — wrong. Find each method's free
    # vars that intersect class-body locals; rewrite those reads to
    # use a stable shadow alias `_outer_<name>` that's introduced at
    # class-body lambda entry by the kwonly default mechanism.
    class_locals = set(_collect_class_body_names(cls_body))
    method_outer_needs = set()
    for s in cls_body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            free = _function_body_free_names(s)
            method_outer_needs |= (free & class_locals)
    # Don't rewrite names that are also class-body-boxed (those are
    # already routed via helper._b_<name> on the class-body helper,
    # not via plain Name reads).
    if method_outer_needs:
        boxed = getattr(stmt, '_class_boxed_names', None) or set()
        method_outer_needs -= boxed
    # Pick alias names for each. Use the temp_var allocator so we
    # avoid colliding with anything reserved.
    method_outer_alias = {}
    for n in sorted(method_outer_needs):
        method_outer_alias[n] = sub_frame.get_temp_var()
    # Rewrite method bodies in place: any free Name(Load, id=n) gets
    # renamed to Name(Load, id=alias).
    if method_outer_alias:
        for s in cls_body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _rewrite_loads_in_function(s, method_outer_alias)
        # Add the alias names to the class body lambda's kwonly
        # defaults so they capture outer-scope values at lambda-
        # creation time.
        for orig_name, alias in method_outer_alias.items():
            if alias not in shadowed:
                shadowed.append(alias)
        # The defaults need to evaluate as the *original* outer
        # name, not the alias (alias doesn't exist outside). Build
        # a parallel default expressions list keyed by alias.
        shadow_default_for = {alias: orig for orig, alias in method_outer_alias.items()}
    else:
        shadow_default_for = {}

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
    extra_kwargs = []
    for kwd in stmt.keywords:
        if kwd.arg == 'metaclass':
            metaclass = kwd.value
        else:
            # Forward other keywords (tag=..., **kw, etc.) to
            # _make_class so they reach metaclass.__prepare__ /
            # __init_subclass__ / metaclass(...) the same way Python
            # would. PEP 487's __init_subclass__ relies on these.
            extra_kwargs.append(kwd)

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
    # PEP 3135 / zero-arg super(): if the class body contains any
    # super() zero-arg calls, add `__class__=None` as a kwonly
    # default on the body lambda. This makes the Python compiler
    # create a `__class__` cell that all nested method lambdas can
    # closure-capture. _make_class fills the cell with the actual
    # class object after construction.
    needs_class_cell = _has_zero_arg_super(stmt.body)
    if needs_class_cell and '__class__' not in shadowed:
        shadowed = list(shadowed) + ['__class__']
        shadow_default_for = dict(shadow_default_for)
        shadow_default_for['__class__'] = '__class__'
        # The default `__class__=__class__` would NameError if
        # __class__ isn't defined in the enclosing scope. Use None
        # as the initial value; _make_class fills it in after.
        shadow_default_for['__class__'] = None  # sentinel: emit Constant(None)

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
                kwonlyargs=[ast.arg(arg=n, annotation=None) for n in shadowed],
                kw_defaults=[
                    ast.Constant(value=None)
                    if shadow_default_for.get(n) is None and n in shadow_default_for
                    else ast.Name(
                        id=shadow_default_for.get(n, n),
                        ctx=ast.Load(),
                    )
                    for n in shadowed
                ],
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
        keywords=extra_kwargs,
    )

    return [
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(stmt.name, stmt.decorator_list, construct),
        )
    ]
