"""Pre-pass to handle `nonlocal` declarations.

Transform-time we generate lambdas, and a lambda has no way to write
back into the surrounding lambda's locals. So nonlocal can't be expressed
directly. Instead we *box* every nonlocal-targeted variable: the owning
function gets a shared object stored in its locals, and every read/write
of that name (in the owner and in any nested function that can see it)
goes through that object's attributes.

The owner of a nonlocal name is the nearest enclosing function whose
body binds it — Python's compile-time rule.

Two passes over the module:

1. Pre-scan: for each FunctionDef collect its bound names and its
   `nonlocal` declarations. Walk the function-nesting tree and resolve
   each declaration to its owner; record the boxed name set on the
   owner.

2. Rewrite: walk the whole module, threading a `lookup(name)` closure
   through the scope tree. At every Name reference, lookup tells us
   either (helper_var, attr) — meaning rewrite to Attribute(...) — or
   None — meaning the name resolves locally and we leave it alone.
   Each enclosing scope wraps the lookup: a function adds its own
   boxed names as positive hits and its other bindings as shadows;
   a lambda or comprehension adds its parameters/targets as shadows.
"""

import ast


# Prefix attached to the user-facing variable name when it's stored on
# the owner's _FuncHelper. Without it, a user variable called `value` or
# `returned` would clash with the helper's own fields (used by the
# return machinery) — _FuncHelper.value is overwritten by do_return at
# function exit, so a boxed `value = ...` would later read back wrong.
_BOX_ATTR_PREFIX = '_b_'


def _bound_names_in_function_body(fdef) -> set:
    """Names bound directly in this function's body — parameters,
    assignment targets, for-loop targets, with-as targets, except-as
    targets, nested def/class names. Does NOT recurse into nested
    function/class bodies (those are separate scopes)."""
    names = set()
    for arg_group in (fdef.args.posonlyargs, fdef.args.args, fdef.args.kwonlyargs):
        for a in arg_group:
            names.add(a.arg)
    if fdef.args.vararg is not None:
        names.add(fdef.args.vararg.arg)
    if fdef.args.kwarg is not None:
        names.add(fdef.args.kwarg.arg)

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)
        # Subscript/Attribute targets don't bind a new name in this scope.

    class _Walker(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)

        def visit_AsyncFunctionDef(self, node):
            names.add(node.name)

        def visit_ClassDef(self, node):
            names.add(node.name)

        def visit_Lambda(self, node):
            pass  # lambda body is its own scope

        def visit_ListComp(self, node):
            pass  # comprehensions are their own scope (3.x)

        def visit_SetComp(self, node):
            pass

        def visit_DictComp(self, node):
            pass

        def visit_GeneratorExp(self, node):
            pass

        def visit_Assign(self, node):
            for t in node.targets:
                add_target(t)
            self.visit(node.value)

        def visit_AugAssign(self, node):
            add_target(node.target)
            self.visit(node.value)

        def visit_AnnAssign(self, node):
            add_target(node.target)
            if node.value is not None:
                self.visit(node.value)

        def visit_For(self, node):
            add_target(node.target)
            self.visit(node.iter)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_AsyncFor(self, node):
            add_target(node.target)
            self.visit(node.iter)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars is not None:
                    add_target(item.optional_vars)
                self.visit(item.context_expr)
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node):
            self.visit_With(node)

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                names.add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_Import(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)

        def visit_Global(self, node):
            # `global x` inside a function means x is NOT a local of
            # this function. Subtract it later — but we don't have a
            # convenient negative-set here. For now, names of `global`
            # variables are still reported as "bound in this scope" so
            # they shadow outer bindings. Refine if real code needs it.
            pass

        def visit_Nonlocal(self, node):
            # Same caveat as Global. `nonlocal x` means x resolves to an
            # outer scope, but for shadowing-detection purposes we leave
            # it alone here — _resolve_nonlocals will set up the box
            # correctly anyway.
            pass

    walker = _Walker()
    for stmt in fdef.body:
        walker.visit(stmt)

    # `global x` and `nonlocal x` declarations mean x is NOT a local of
    # this function — it resolves to an outer scope. Strip them so the
    # nonlocal-resolution lookup keeps walking outward.
    declared_global = set()
    declared_nonlocal = set()

    class _Decls(ast.NodeVisitor):
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

        def visit_Global(self, node):
            for n in node.names:
                declared_global.add(n)

        def visit_Nonlocal(self, node):
            for n in node.names:
                declared_nonlocal.add(n)

    decl_walker = _Decls()
    for stmt in fdef.body:
        decl_walker.visit(stmt)

    return names - declared_global - declared_nonlocal


def _collect_direct_nonlocal_decls(fdef) -> set:
    """Names declared `nonlocal` directly in fdef's body (not in nested
    functions)."""
    decls = set()

    class _V(ast.NodeVisitor):
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

        def visit_Nonlocal(self, node):
            for n in node.names:
                decls.add(n)

    v = _V()
    for stmt in fdef.body:
        v.visit(stmt)
    return decls


def _annotate_bound_names(tree):
    class _A(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            node._bound_names = _bound_names_in_function_body(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            node._bound_names = _bound_names_in_function_body(node)
            self.generic_visit(node)

    _A().visit(tree)


def _resolve_owners(tree) -> dict:
    """Return {id(FunctionDef): set(boxed_names)} — for each function,
    which of its locals are referenced by `nonlocal` in some descendant.
    """
    boxed = {}

    def visit(node, func_stack):
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_func:
            decls = _collect_direct_nonlocal_decls(node)
            for name in decls:
                # Walk outward to find the owner.
                for outer in reversed(func_stack):
                    if name in outer._bound_names:
                        boxed.setdefault(id(outer), set()).add(name)
                        break
                # If no owner (invalid Python), silently drop.
            new_stack = func_stack + [node]
            for child in ast.iter_child_nodes(node):
                visit(child, new_stack)
        elif isinstance(node, ast.ClassDef):
            # Class body is a scope but doesn't show up in nonlocal
            # resolution chains.
            for child in ast.iter_child_nodes(node):
                visit(child, func_stack)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, func_stack)

    visit(tree, [])
    return boxed


def _comprehension_targets(generators) -> set:
    """Names bound by the `for X in iter` clauses of a comprehension."""
    names = set()

    def add(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add(e)
        elif isinstance(t, ast.Starred):
            add(t.value)

    for g in generators:
        add(g.target)
    return names


def _lambda_arg_names(args) -> set:
    names = set()
    for g in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in g:
            names.add(a.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _rewrite(tree, boxed, helper_name_for):
    def visit(node, lookup):
        if node is None:
            return None

        if isinstance(node, ast.Name):
            hit = lookup(node.id)
            if hit is not None:
                helper_var, attr = hit
                return ast.Attribute(
                    value=ast.Name(id=helper_var, ctx=ast.Load()),
                    attr=_BOX_ATTR_PREFIX + attr,
                    ctx=node.ctx,
                )
            return node

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound = node._bound_names
            boxset = boxed.get(id(node), set())
            local_helper = helper_name_for(node) if boxset else None

            def new_lookup(name, _parent=lookup, _bound=bound, _box=boxset, _h=local_helper):
                if name in _box:
                    return (_h, name)
                if name in _bound:
                    return None
                return _parent(name)

            # Decorators and defaults evaluate in the OUTER scope.
            node.decorator_list = [visit(d, lookup) for d in node.decorator_list]
            node.args.defaults = [visit(d, lookup) for d in node.args.defaults]
            node.args.kw_defaults = [visit(d, lookup) for d in node.args.kw_defaults]
            # Body in NEW scope.
            node.body = [visit(s, new_lookup) for s in node.body]
            # returns annotation: enclosing scope. We don't care since
            # gen_func strips annotations, but rewrite for completeness.
            node.returns = visit(node.returns, lookup)
            return node

        if isinstance(node, ast.Lambda):
            args_bound = _lambda_arg_names(node.args)

            def new_lookup(name, _parent=lookup, _bound=args_bound):
                if name in _bound:
                    return None
                return _parent(name)

            node.args.defaults = [visit(d, lookup) for d in node.args.defaults]
            node.args.kw_defaults = [visit(d, lookup) for d in node.args.kw_defaults]
            node.body = visit(node.body, new_lookup)
            return node

        if isinstance(node, ast.ClassDef):
            # Class body is a scope but is invisible to nested function
            # nonlocal-resolution. For boxed-name lookup we treat it as
            # transparent: a Name reference in a class body resolves up
            # through the enclosing functions just like a function body.
            node.decorator_list = [visit(d, lookup) for d in node.decorator_list]
            node.bases = [visit(b, lookup) for b in node.bases]
            for kw in node.keywords:
                kw.value = visit(kw.value, lookup)
            node.body = [visit(s, lookup) for s in node.body]
            return node

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            comp_bound = _comprehension_targets(node.generators)

            def new_lookup(name, _parent=lookup, _bound=comp_bound):
                if name in _bound:
                    return None
                return _parent(name)

            # First generator's iter runs in OUTER scope.
            new_gens = []
            for i, gen in enumerate(node.generators):
                if i == 0:
                    gen.iter = visit(gen.iter, lookup)
                else:
                    gen.iter = visit(gen.iter, new_lookup)
                gen.target = visit(gen.target, new_lookup)
                gen.ifs = [visit(x, new_lookup) for x in gen.ifs]
                new_gens.append(gen)
            node.generators = new_gens
            node.elt = visit(node.elt, new_lookup)
            return node

        if isinstance(node, ast.DictComp):
            comp_bound = _comprehension_targets(node.generators)

            def new_lookup(name, _parent=lookup, _bound=comp_bound):
                if name in _bound:
                    return None
                return _parent(name)

            new_gens = []
            for i, gen in enumerate(node.generators):
                if i == 0:
                    gen.iter = visit(gen.iter, lookup)
                else:
                    gen.iter = visit(gen.iter, new_lookup)
                gen.target = visit(gen.target, new_lookup)
                gen.ifs = [visit(x, new_lookup) for x in gen.ifs]
                new_gens.append(gen)
            node.generators = new_gens
            node.key = visit(node.key, new_lookup)
            node.value = visit(node.value, new_lookup)
            return node

        # Generic walk.
        for field, old in ast.iter_fields(node):
            if isinstance(old, list):
                setattr(node, field, [
                    visit(v, lookup) if isinstance(v, ast.AST) else v
                    for v in old
                ])
            elif isinstance(old, ast.AST):
                setattr(node, field, visit(old, lookup))
        return node

    def top_lookup(name):
        return None

    visit(tree, top_lookup)


def apply_nonlocal_pass(tree, name_provider) -> None:
    """Rewrite `tree` in place so that nonlocal-targeted names go
    through their owner's box. `name_provider` is a zero-arg callable
    returning a fresh temp variable name; we use it to allocate a
    helper-var name for each owner function."""
    _annotate_bound_names(tree)
    boxed = _resolve_owners(tree)

    if not boxed:
        return  # nothing to do

    helper_name = {}

    def helper_name_for(func_node):
        key = id(func_node)
        if key not in helper_name:
            helper_name[key] = name_provider()
            func_node._box_helper_var = helper_name[key]
        return helper_name[key]

    _rewrite(tree, boxed, helper_name_for)
