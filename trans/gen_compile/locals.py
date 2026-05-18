import ast


# ---------------------------------------------------------------------
# Locals discovery

def _free_user_names(node, all_locals: set) -> set:
    """Return the subset of `all_locals` that's referenced (Load ctx)
    inside `node` (a nested def/class). Descends into the entire
    subtree because we want to dehydrate any name the nested scope
    might close over, including in deeply nested expressions."""
    found = set()

    class _V(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id in all_locals:
                found.add(n.id)

    _V().visit(node)
    return found


class _MiniFrame:
    """Adapter so compile_match (which expects a Frame-like object with
    get_temp_var) can be called from the generator pipeline."""

    def __init__(self, name_provider):
        self.get_temp_var = name_provider


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


def collect_user_locals(body: list, args: ast.arguments) -> set:
    """Names that the user reads or writes at top level inside the
    generator body, plus the function's parameters. Used to decide
    what to box on self.

    Doesn't recurse into nested function/class/lambda — those have
    their own scope. Names declared `nonlocal` or `global` are
    excluded: those resolve up the closure / module scope and are
    rewritten to `outer_helper._b_<name>` / `globals()[<name>]` by
    the nonlocal pre-pass before SelfRewriter runs."""
    names = set()

    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            names.add(a.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)

    # Collect direct nonlocal / global declarations — those names must
    # NOT be boxed on self.
    nonlocal_or_global = set()
    class _DeclWalker(ast.NodeVisitor):
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass
        def visit_Nonlocal(self, node):
            for n in node.names:
                nonlocal_or_global.add(n)
        def visit_Global(self, node):
            for n in node.names:
                nonlocal_or_global.add(n)
    dw = _DeclWalker()
    for stmt in body:
        dw.visit(stmt)

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    class _Walker(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)
        def visit_AsyncFunctionDef(self, node):
            names.add(node.name)
        def visit_ClassDef(self, node):
            names.add(node.name)
        def visit_Lambda(self, node):
            pass

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                names.add(node.name)
            for s in node.body:
                self.visit(s)

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

        def visit_Name(self, node):
            # Reads also matter — they tell us this name is a local
            # (vs. a module-level reference).
            if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                # A pure Load could be a global; we can't tell statically
                # without a full scope analysis. To stay safe we DON'T
                # add Load-only names; only Store / Del. This means a
                # generator that only reads a closure variable doesn't
                # accidentally box that name.
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)

        def visit_NamedExpr(self, node):
            # Walrus targets ARE boxed in the generator. _SelfRewriter
            # rewrites the walrus to a setattr-based tuple expression
            # so the binding survives across yield boundaries.
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            self.visit(node.value)

        def visit_Import(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)

    walker = _Walker()
    for stmt in body:
        walker.visit(stmt)

    # Remove names declared nonlocal / global — those resolve up the
    # closure / module scope, not on self.
    names -= nonlocal_or_global

    return names
