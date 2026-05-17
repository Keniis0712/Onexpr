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

Generator functions (def whose body contains yield / yield from
directly) are skipped by both passes — compile_generator handles its
own boxing via self attributes on the state-machine class, so adding
nonlocal-pass boxing on top would leave dangling helper-var
references inside the rewritten send method.
"""

import ast


def _is_generator_func(fdef) -> bool:
    """True iff fdef goes through gen_compile.compile_generator.
    That happens for: (a) regular def whose body uses yield / yield
    from directly, (b) any AsyncFunctionDef (we lower coroutines to
    a state machine too, regardless of whether they yield)."""
    if isinstance(fdef, ast.AsyncFunctionDef):
        return True
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


def _collect_try_clause_assigns(fdef) -> set:
    """Names assigned inside any try clause directly within fdef's body
    (body / except / else / finally), but not inside nested function
    or class bodies. These need to be boxed because each try clause is
    rewritten into its own lambda — assignments inside that lambda
    can't propagate to the enclosing scope without a shared box."""
    names = set()

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    class _ClauseWalker(ast.NodeVisitor):
        """Walks a try clause body collecting bound names. Stops at
        nested def/class/lambda boundaries (those have their own
        scopes that handle their own boxing)."""
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

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
            self.visit_For(node)

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
            # Don't collect node.name — that's handled as a lambda
            # parameter by parse_try, not as a boxed local. Walk into
            # the body so real assignments inside the handler are
            # collected.
            for s in node.body:
                self.visit(s)

        def visit_Import(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)

    class _TryFinder(ast.NodeVisitor):
        """Finds every try OR with statement directly in fdef's body
        (skipping nested function/class/lambda bodies, which would be
        other owners' problems). For each, walks the clauses with
        _ClauseWalker so the assigned names get boxed."""
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

        def visit_Try(self, node):
            cw = _ClauseWalker()
            for stmt_list in (node.body, node.orelse, node.finalbody):
                for s in stmt_list:
                    cw.visit(s)
            for handler in node.handlers:
                cw.visit(handler)
            # Recurse into the try clauses for nested try statements.
            for stmt_list in (node.body, node.orelse, node.finalbody):
                for s in stmt_list:
                    self.visit(s)
            for handler in node.handlers:
                for s in handler.body:
                    self.visit(s)

        def visit_TryStar(self, node):
            self.visit_Try(node)

        def visit_With(self, node):
            cw = _ClauseWalker()
            for s in node.body:
                cw.visit(s)
            # Note: `with ctx as x:` binds x as a regular outer-scope
            # assignment (parse_with emits an Assign at the with's
            # outer level, not inside the body lambda), so x doesn't
            # need boxing.
            # Recurse into the body for nested try/with.
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node):
            self.visit_With(node)

    finder = _TryFinder()
    for stmt in fdef.body:
        finder.visit(stmt)
    return names


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
    which of its locals must be boxed.

    Two reasons a local needs boxing:
    1. Some descendant function declared it `nonlocal` (the original
       Python semantics).
    2. The function contains a `try` statement and the local is
       assigned inside one of the try clauses. Each try clause is
       rewritten into its own lambda, so without a box the assignment
       is lost when the lambda returns.
    """
    boxed = {}

    def visit(node, func_stack):
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if is_func and _is_generator_func(node):
            # Generator function: compile_generator handles its own
            # boxing through the state-machine class. Don't add this
            # node to the stack and don't recurse into its body — but
            # we still need to walk it so any nested non-generator
            # function inside the generator gets its own analysis. We
            # pass an empty-ish func_stack from this point so a
            # nonlocal declaration inside a deeper non-generator
            # function whose target lives in the generator's body
            # silently fails to find an owner (which is correct — that
            # binding is on self, not in any enclosing lambda). For
            # most user code this is a no-op.
            for child in ast.iter_child_nodes(node):
                visit(child, func_stack)
        elif is_func:
            decls = _collect_direct_nonlocal_decls(node)
            for name in decls:
                # Walk outward to find the owner.
                for outer in reversed(func_stack):
                    if name in outer._bound_names:
                        boxed.setdefault(id(outer), set()).add(name)
                        break
                # If no owner (invalid Python), silently drop.

            # Box every local that's assigned inside a try clause in
            # this function — the try clauses become their own lambdas
            # so writes need a shared container.
            for name in _collect_try_clause_assigns(node):
                if name in node._bound_names:
                    boxed.setdefault(id(node), set()).add(name)

            new_stack = func_stack + [node]
            for child in ast.iter_child_nodes(node):
                visit(child, new_stack)
        elif isinstance(node, ast.ClassDef):
            # Class body has the same try-clause-lambda problem as a
            # function body — assignments inside the try clause are
            # otherwise lost when the per-clause lambda returns. Box
            # them on the class body's helper. parse_class_def will
            # emit reverse-copies before `return locals()` so the
            # class namespace ends up with the right names.
            class_boxed = set()
            for name in _collect_try_clause_assigns(node):
                class_boxed.add(name)
            if class_boxed:
                boxed[id(node)] = class_boxed
                # Stash on the node so parse_class_def can emit
                # the reverse-copies.
                node._class_boxed_names = class_boxed
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
            if _is_generator_func(node):
                # Generator function: skip body rewrite. Decorators and
                # defaults still evaluate in the outer scope, so rewrite
                # those. Body is left intact for compile_generator —
                # nested non-generator functions inside the body don't
                # see boxed names from this layer (which is fine: they
                # close over send-locals dehydrated by gen_compile).
                node.decorator_list = [visit(d, lookup) for d in node.decorator_list]
                node.args.defaults = [visit(d, lookup) for d in node.args.defaults]
                node.args.kw_defaults = [visit(d, lookup) for d in node.args.kw_defaults]
                node.returns = visit(node.returns, lookup)
                return node

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
            # transparent in the parent direction: a Name reference in
            # a class body resolves up through the enclosing functions
            # just like a function body. But the class itself can also
            # be an owner — try-clause assignments inside the class
            # body get boxed on the class's own helper (so per-clause
            # lambdas can write through to a shared container, and
            # parse_class_def reverse-copies them into locals before
            # `return locals()`).
            class_boxset = boxed.get(id(node), set())
            class_helper = helper_name_for(node) if class_boxset else None

            def class_lookup(name, _parent=lookup, _box=class_boxset, _h=class_helper):
                if name in _box:
                    return (_h, name)
                return _parent(name)

            node.decorator_list = [visit(d, lookup) for d in node.decorator_list]
            node.bases = [visit(b, lookup) for b in node.bases]
            for kw in node.keywords:
                kw.value = visit(kw.value, lookup)
            node.body = [visit(s, class_lookup) for s in node.body]
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


def apply_nonlocal_pass(tree, name_provider, module_helper_var=None, top_frame=None) -> None:
    """Rewrite `tree` in place so that nonlocal-targeted names go
    through their owner's box. `name_provider` is a zero-arg callable
    returning a fresh temp variable name; we use it to allocate a
    helper-var name for each owner function.

    `top_frame`, if given, is treated as the module-level Frame; we add
    every name assigned inside a top-level `try` clause to its
    `global_vars` so parse_assign rewrites those writes to
    `globals().__setitem__('name', value)`. Without this, the
    assignment lands in the per-clause lambda's own locals and never
    surfaces to the module scope.

    `module_helper_var` is retained for compatibility but is no longer
    used by this pass — module-level try-clause assigns now go through
    globals() instead of a helper-attribute box.
    """
    _annotate_bound_names(tree)
    boxed = _resolve_owners(tree)

    # Module-level try-clause assignments: route through globals() so
    # they become real module attributes (visible to subsequent code
    # AND to importers of the transformed module).
    if top_frame is not None:
        for name in _collect_module_top_try_assigns(tree):
            if name not in top_frame.global_vars:
                top_frame.global_vars.append(name)

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


def _collect_module_top_try_assigns(tree) -> set:
    """Names assigned inside a try clause that sits directly in the
    module body (not inside a function or class)."""
    names = set()

    class _Walker(ast.NodeVisitor):
        # Reuse the same logic _ClauseWalker uses, but we can't share
        # the closure cleanly so we inline a minimal version here.
        def __init__(self):
            self.in_try = 0

        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

        def visit_Try(self, node):
            self.in_try += 1
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)
            for s in node.finalbody:
                self.visit(s)
            for h in node.handlers:
                for s in h.body:
                    self.visit(s)
            self.in_try -= 1

        def visit_TryStar(self, node):
            self.visit_Try(node)

        def visit_With(self, node):
            self.in_try += 1
            for s in node.body:
                self.visit(s)
            self.in_try -= 1

        def visit_AsyncWith(self, node):
            self.visit_With(node)

        def _add_target(self, t):
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    self._add_target(e)
            elif isinstance(t, ast.Starred):
                self._add_target(t.value)

        def visit_Assign(self, node):
            if self.in_try:
                for t in node.targets:
                    self._add_target(t)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            if self.in_try:
                self._add_target(node.target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if self.in_try:
                self._add_target(node.target)
            self.generic_visit(node)

    _Walker().visit(tree)
    return names
