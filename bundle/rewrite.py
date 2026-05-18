"""AST rewriter for individual modules in a bundle.

Walks one module's AST and:
  - redirects every reference to a module-top-level name to `_mod.<name>`
  - rewrites every internal `import` / `from ... import ...` to go through
    `_bnd_load(<dotted>)` + `sys.modules`
  - leaves external imports (stdlib / 3rd-party) alone, except the names
    they bind get mirrored onto `_mod.<bound>`

Driven from emit.emit_module_function.
"""
from __future__ import annotations

import ast

from .discover import parent_pkg, resolve_relative
from .scope import ScopeInfo


class ModuleRewriter(ast.NodeTransformer):
    """Rewrite a module's AST so that module-level names are attribute accesses
    on a local `_mod` (types.ModuleType instance), and bundle-internal imports
    go through `_bnd_load`.

    Active only inside the function-wrapped body of one module.
    """

    def __init__(self, modname: str, kind: str, source: str,
                 internal_modules: set[str], scope: ScopeInfo):
        self.modname = modname
        self.kind = kind
        self.pkg = parent_pkg(modname, kind)
        self.source = source
        self.internal = internal_modules
        self.scope_stack: list[ScopeInfo] = [scope]
        self._tmp_counter = 0

    # -- scope tracking -------------------------------------------------------

    @property
    def scope(self) -> ScopeInfo:
        return self.scope_stack[-1]

    def _enter(self, node: ast.AST) -> ScopeInfo | None:
        child = self.scope.child_for(node)
        if child is not None:
            self.scope_stack.append(child)
        return child

    def _leave(self, child: ScopeInfo | None) -> None:
        if child is not None:
            self.scope_stack.pop()

    def _tmp(self, hint: str = "t") -> str:
        self._tmp_counter += 1
        return f"_bnd_{hint}_{self._tmp_counter}"

    # -- helpers --------------------------------------------------------------

    def _is_internal(self, dotted: str) -> bool:
        # internal if dotted matches any registered module name, or is a
        # prefix that has registered descendants (i.e. an internal package).
        if dotted in self.internal:
            return True
        prefix = dotted + "."
        return any(m.startswith(prefix) for m in self.internal)

    def _module_attr(self, name: str, ctx: ast.expr_context) -> ast.Attribute:
        return ast.Attribute(value=ast.Name(id="_mod", ctx=ast.Load()),
                             attr=name, ctx=ctx)

    def _load(self, dotted: str) -> ast.Call:
        return ast.Call(
            func=ast.Name(id="_bnd_load", ctx=ast.Load()),
            args=[ast.Constant(value=dotted)],
            keywords=[])

    # -- visitors -------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.scope.resolves_to_module(node.id) and node.id != "_mod":
            return ast.copy_location(self._module_attr(node.id, node.ctx), node)
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        # `global x` in a function -> already encoded in symtable; the function's
        # uses of x are rewritten to _mod.x by visit_Name. We can drop the
        # `global` declaration since there's no real global being shadowed.
        return ast.Pass()

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        return node

    # nested-scope traversal must update scope_stack
    def visit_FunctionDef(self, node):
        return self._visit_funclike(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit_funclike(node)

    def visit_Lambda(self, node):
        return self._visit_funclike(node)

    def _visit_funclike(self, node):
        # decorators + defaults are evaluated in the *outer* scope
        node.decorator_list = [self.visit(d) for d in getattr(node, "decorator_list", [])]
        if isinstance(node.args, ast.arguments):
            node.args.defaults = [self.visit(d) for d in node.args.defaults]
            node.args.kw_defaults = [self.visit(d) if d is not None else None
                                     for d in node.args.kw_defaults]
            # annotations on params evaluated in outer scope
            for a in (*node.args.args, *node.args.posonlyargs, *node.args.kwonlyargs):
                if a.annotation is not None:
                    a.annotation = self.visit(a.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                node.args.vararg.annotation = self.visit(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                node.args.kwarg.annotation = self.visit(node.args.kwarg.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            node.returns = self.visit(node.returns)

        child = self._enter(node)
        try:
            if isinstance(node, ast.Lambda):
                node.body = self.visit(node.body)
            else:
                node.body = [self.visit(s) for s in node.body]
        finally:
            self._leave(child)
        return node

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [ast.keyword(arg=kw.arg, value=self.visit(kw.value))
                         for kw in node.keywords]
        child = self._enter(node)
        try:
            node.body = [self.visit(s) for s in node.body]
        finally:
            self._leave(child)
        return node

    def visit_ListComp(self, node):
        return self._visit_comp(node)

    def visit_SetComp(self, node):
        return self._visit_comp(node)

    def visit_DictComp(self, node):
        return self._visit_comp(node)

    def visit_GeneratorExp(self, node):
        return self._visit_comp(node)

    def _visit_comp(self, node):
        # the *outermost* iter is evaluated in the enclosing scope
        outer_iter = self.visit(node.generators[0].iter)
        child = self._enter(node)
        try:
            new_gens = []
            for i, gen in enumerate(node.generators):
                it = outer_iter if i == 0 else self.visit(gen.iter)
                target = self.visit(gen.target)
                ifs = [self.visit(c) for c in gen.ifs]
                new_gens.append(ast.comprehension(target=target, iter=it,
                                                  ifs=ifs, is_async=gen.is_async))
            node.generators = new_gens
            if isinstance(node, ast.DictComp):
                node.key = self.visit(node.key)
                node.value = self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
        finally:
            self._leave(child)
        return node

    # -- import rewriting -----------------------------------------------------

    def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for alias in node.names:
            top = alias.name.split(".")[0]
            if not self._is_internal(top):
                out.append(ast.Import(names=[alias]))
                continue
            # internal: ensure all packages along the chain are loaded
            parts = alias.name.split(".")
            for i in range(1, len(parts) + 1):
                out.append(ast.Expr(self._load(".".join(parts[:i]))))
            bound_name = alias.asname or top
            target_dotted = alias.name if alias.asname else top
            # _mod.<bound> = sys.modules['<target_dotted>']
            out.append(ast.Assign(
                targets=[self._module_attr(bound_name, ast.Store())],
                value=ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id="sys", ctx=ast.Load()),
                                        attr="modules", ctx=ast.Load()),
                    slice=ast.Constant(value=target_dotted),
                    ctx=ast.Load())))
        return [ast.copy_location(s, node) for s in out]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.stmt]:
        target = resolve_relative(self.pkg, node.level, node.module)
        if not target or not self._is_internal(target.split(".")[0]):
            # external (stdlib / 3rd-party). leave alone, but module-level
            # names it binds should still be reflected on _mod, which we
            # achieve by storing into _mod.<name> via a tiny rewrite.
            return self._external_from(node)

        stmts: list[ast.stmt] = []
        # ensure target loaded
        stmts.append(ast.Expr(self._load(target)))
        src_tmp = self._tmp("src")
        stmts.append(ast.Assign(
            targets=[ast.Name(id=src_tmp, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Attribute(value=ast.Name(id="sys", ctx=ast.Load()),
                                    attr="modules", ctx=ast.Load()),
                slice=ast.Constant(value=target),
                ctx=ast.Load())))

        for alias in node.names:
            if alias.name == "*":
                # runtime expansion using __all__ or non-underscore names
                stmts.append(ast.Expr(ast.Call(
                    func=ast.Name(id="_bnd_star", ctx=ast.Load()),
                    args=[ast.Name(id="_mod", ctx=ast.Load()),
                          ast.Name(id=src_tmp, ctx=ast.Load())],
                    keywords=[])))
                continue
            bound = alias.asname or alias.name
            # if the source package has a submodule of this name, ensure it
            # is loaded so that getattr works (matches 'from pkg import sub').
            sub_dotted = f"{target}.{alias.name}"
            if self._is_internal(sub_dotted):
                stmts.append(ast.Expr(self._load(sub_dotted)))
            stmts.append(ast.Assign(
                targets=[self._module_attr(bound, ast.Store())],
                value=ast.Attribute(value=ast.Name(id=src_tmp, ctx=ast.Load()),
                                    attr=alias.name, ctx=ast.Load())))
        return [ast.copy_location(s, node) for s in stmts]

    def _external_from(self, node: ast.ImportFrom) -> list[ast.stmt]:
        # We still want the *bound names* to live on _mod (so `globals()`
        # equivalence holds). Rewrite:
        #   from os.path import join as J, exists
        # -> _bnd_t = __import__('os.path', fromlist=['join','exists'])
        #    _mod.J = _bnd_t.join; _mod.exists = _bnd_t.exists
        # For `from os import *` external — rare, but: use _bnd_star.
        if node.level != 0:
            # relative import that didn't resolve to internal — let it raise as Python would
            return [node]
        names = [a.name for a in node.names if a.name != "*"]
        has_star = any(a.name == "*" for a in node.names)
        tmp = self._tmp("ext")
        out: list[ast.stmt] = []
        out.append(ast.Assign(
            targets=[ast.Name(id=tmp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load()),
                args=[ast.Constant(value=node.module or "")],
                keywords=[ast.keyword(arg="fromlist",
                                       value=ast.List(elts=[ast.Constant(value=n) for n in names] or [ast.Constant(value="*")],
                                                      ctx=ast.Load()))])))
        if has_star:
            out.append(ast.Expr(ast.Call(
                func=ast.Name(id="_bnd_star", ctx=ast.Load()),
                args=[ast.Name(id="_mod", ctx=ast.Load()),
                      ast.Name(id=tmp, ctx=ast.Load())],
                keywords=[])))
        for a in node.names:
            if a.name == "*":
                continue
            bound = a.asname or a.name
            out.append(ast.Assign(
                targets=[self._module_attr(bound, ast.Store())],
                value=ast.Attribute(value=ast.Name(id=tmp, ctx=ast.Load()),
                                    attr=a.name, ctx=ast.Load())))
        return [ast.copy_location(s, node) for s in out]
