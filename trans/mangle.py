"""Optional mangling pass that runs *before* add_helper / parse_stmts.

Driven by a tag set:

    helper    — mangle the runtime helper class / member names. Done
                later by add_helper itself; this module just records
                the tag so add_helper can pick it up.
    toplevel  — module-top-level def/class names → temp_N, plus their
                references everywhere they're loaded.
    imports   — names introduced by `import X` / `from M import N` /
                `as alias` → temp_N.
    locals    — function parameters + local assignments + lambda
                params + comprehension targets → temp_N.
    methods   — class method names + class-body self.X / cls.X field
                names → temp_N. Attribute reads on the *outside*
                (foo_instance.X) get the same rewrite — but only when
                the name X was actually defined inside a ClassDef in
                this source file. External APIs (e.g. list.append)
                stay intact.
    attrs     — every Attribute.attr (other than dunders) → temp_N.
                This breaks any attribute that wasn't reflected in
                the AST as a definition (stdlib API names like
                .append, .send, .throw). Use only when the resulting
                file is fully self-contained.

The pass uses symtable for scope resolution so that:
    def outer(): x = 1; def inner(): return x
yields a single mangled name shared by both occurrences (inner's `x`
is `free`, resolved to outer's binding).

Names always preserved: dunders (__init__, __class__, …), builtins
(print, len, …), and string literals — including format-spec attr
names inside f-strings, which are sometimes mistaken for code.
"""
from __future__ import annotations

import ast
import builtins
import symtable

VALID_TAGS = {
    'none', 'helper', 'toplevel', 'imports', 'locals', 'methods',
    'attrs', 'safe', 'all',
}
SAFE_TAGS = {'helper', 'toplevel', 'imports', 'locals', 'methods'}
ALL_TAGS = SAFE_TAGS | {'attrs'}


def expand_tags(spec: str) -> set[str]:
    """Parse a CSV tag spec into a normalised tag set.

    `spec` is the value of --replace-name; aliases none/safe/all
    expand into their underlying tags. Returns {} for 'none' or empty.
    """
    if not spec:
        return set()
    out: set[str] = set()
    for raw in spec.split(','):
        t = raw.strip()
        if not t or t == 'none':
            continue
        if t == 'safe':
            out |= SAFE_TAGS
            continue
        if t == 'all':
            out |= ALL_TAGS
            continue
        if t not in VALID_TAGS:
            raise SystemExit(
                f'unknown --replace-name tag: {t!r} '
                f'(valid: {sorted(VALID_TAGS)})'
            )
        out.add(t)
    # `attrs` rewrites every Attribute.attr — that includes class
    # method definitions inside ClassDef, which only get their name
    # rebound when `methods` is on. Imply it.
    if 'attrs' in out:
        out.add('methods')
    return out


# ---------------------------------------------------------------------------
# Scope-aware name pool
# ---------------------------------------------------------------------------

_BUILTINS = set(dir(builtins))


def _is_safe_name(name: str) -> bool:
    """Names we never mangle: dunders + builtins."""
    if name.startswith('__') and name.endswith('__'):
        return True
    if name in _BUILTINS:
        return True
    return False


class _NamePool:
    """Allocates fresh temp_N names while honouring `reserved` (the
    union of the user's existing identifiers + names we want to leave
    alone). The same pool is used by Frame later, so the counters
    must agree — we accept an external counter callable."""

    def __init__(self, reserved: set[str], allocator):
        self.reserved = set(reserved)
        self.allocator = allocator  # callable returning the next name
        self.cache: dict = {}  # binding_id -> mangled name

    def get_for(self, binding_id) -> str:
        if binding_id not in self.cache:
            name = self.allocator()
            self.cache[binding_id] = name
            self.reserved.add(name)
        return self.cache[binding_id]


# ---------------------------------------------------------------------------
# Symtable wrapper
# ---------------------------------------------------------------------------

def _walk_symtable(table: symtable.SymbolTable, parents: tuple = ()):
    """Yield (scope_path, table) for every scope, depth-first."""
    yield parents, table
    for c in table.get_children():
        yield from _walk_symtable(c, parents + (table,))


def _scope_key(table: symtable.SymbolTable) -> tuple:
    """A hashable identifier for a symtable scope. (type, name, lineno)."""
    return (table.get_type(), table.get_name(), table.get_lineno())


def _resolve_binding(scope_path: tuple, table: symtable.SymbolTable,
                     name: str) -> tuple | None:
    """Walk up parent scopes looking for the scope that *defines* `name`
    (is_local without being free). Returns (scope_key, name) or None
    if `name` is a builtin / undefined / global at module level.

    For names declared `global x`, the defining scope is the module.
    For names declared `nonlocal x`, it's the nearest enclosing
    function scope that has `x` as a local.
    """
    # Walk from current scope outward.
    chain = list(scope_path) + [table]
    for i in range(len(chain) - 1, -1, -1):
        t = chain[i]
        try:
            sym = t.lookup(name)
        except KeyError:
            continue
        if sym.is_local() and not sym.is_free():
            return (_scope_key(t), name)
        # 'global x' jumps to module scope
        if sym.is_global() and chain:
            return (_scope_key(chain[0]), name)
    return None


# ---------------------------------------------------------------------------
# Walker that performs the rewrite
# ---------------------------------------------------------------------------

class _Rewriter(ast.NodeTransformer):
    def __init__(self, src: str, tags: set[str], pool: _NamePool,
                 type_map=None):
        self.src = src
        # `attrs` rewrites every Attribute.attr; without `methods`
        # the class-body method definitions wouldn't be renamed
        # consistently with their call sites. Imply it.
        if 'attrs' in tags:
            tags = tags | {'methods'}
        self.tags = tags
        # Optional type_map (trans.infer.TypeMap) — when present,
        # `attrs` mode skips Attribute.attr accesses whose receiver
        # is stdlib, and only renames user-class attributes.
        self.type_map = type_map
        self.pool = pool
        self.module_table = symtable.symtable(src, '<onexpr-mangle>', 'exec')
        self.scope_stack: list[symtable.SymbolTable] = [self.module_table]

        # Class-defined attribute names (for `methods` tag): every
        # name that was assigned via `self.X = ...` or appears as a
        # method/class-attr inside *some* ClassDef in the source. The
        # set is enough — we don't track which class, because (a)
        # one symbol-name can mean only one thing in Python's dot
        # syntax, (b) inheritance breaks any narrower invariant.
        self.class_member_names: set[str] = set()
        if 'methods' in self.tags or 'attrs' in self.tags:
            self._collect_class_members()

        # Pre-collect names to mangle, keyed by binding id.
        self.binding_to_mangle: dict[tuple, bool] = {}
        self._collect_bindings()

        # Map for class-member names: name → mangled name. Built after
        # _collect_bindings, before scope traversal, so the cache is
        # primed for both `binding_to_mangle` lookups (in class scope)
        # and Attribute.attr rewrites.
        self.member_map: dict[str, str] = {}
        for n in self.class_member_names:
            if _is_safe_name(n):
                continue
            self.member_map[n] = self.pool.allocator()

        # For every class-scope binding whose name is in member_map,
        # pre-seed the pool cache so visiting the method's FunctionDef
        # produces the same mangled name as `inst.<method>` references
        # will (member_map drives those).
        for bid, want in self.binding_to_mangle.items():
            if not want:
                continue
            scope_key, name = bid
            if scope_key[0] == symtable.SymbolTableType.CLASS and name in self.member_map:
                self.pool.cache[bid] = self.member_map[name]

    # -- pass 1: collect ----------------------------------------------------

    def _collect_bindings(self):
        for scope_path, table in _walk_symtable(self.module_table):
            ttype = table.get_type()
            for sym in table.get_symbols():
                name = sym.get_name()
                if _is_safe_name(name):
                    continue
                # Skip 'self'/'cls' params — they get mangled if locals
                # is in tags, but it's surprising and breaks readability
                # without much win. Keep them.
                if name in ('self', 'cls'):
                    continue
                bid = (_scope_key(table), name)
                if bid in self.binding_to_mangle:
                    continue
                # Decide based on tags.
                want = False
                if ttype == 'module' and sym.is_local():
                    if sym.is_imported() and 'imports' in self.tags:
                        want = True
                    elif (
                        sym.is_assigned() or sym.is_namespace()
                    ) and 'toplevel' in self.tags:
                        # Module-level binding via `def`/`class`/assign.
                        want = True
                elif ttype == 'function' and sym.is_local() and not sym.is_free():
                    # Function parameters keep their names: callers
                    # may pass them by keyword (`f(arg=...)`) and
                    # there's no general way to rewrite the keyword
                    # at every call site. `locals` only mangles
                    # genuinely local (non-parameter) bindings.
                    if sym.is_parameter():
                        pass
                    elif sym.is_imported():
                        # Imports inside a function are governed by
                        # the `imports` tag, not `locals`.
                        if 'imports' in self.tags:
                            want = True
                    elif 'locals' in self.tags:
                        want = True
                elif ttype == 'class' and sym.is_local():
                    # Class-body bindings: method names + class attrs.
                    # When 'methods' is on we also want to rename the
                    # method as a binding in class scope (so e.g.
                    # `def greet(self):` becomes `def temp_X(self):`
                    # AND every `self.greet` / `inst.greet` access
                    # picks up the same mangled name via member_map.
                    if 'methods' in self.tags and sym.is_assigned():
                        self.class_member_names.add(name)
                        want = True
                self.binding_to_mangle[bid] = want

    def _collect_class_members(self):
        """Walk the AST collecting names that any ClassDef in this file
        defines. A name 'X' qualifies if any of:
          - a method `def X(self, ...)` inside a ClassDef
          - a class-level assign `X = ...` inside a ClassDef
          - `self.X = ...` or `cls.X = ...` inside a ClassDef
        """
        tree = ast.parse(self.src)
        class _Coll(ast.NodeVisitor):
            def __init__(self, out: set):
                self.out = out
                self.depth = 0
            def visit_ClassDef(self, node):
                self.depth += 1
                for s in node.body:
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not _is_safe_name(s.name):
                            self.out.add(s.name)
                    if isinstance(s, ast.Assign):
                        for tgt in s.targets:
                            if isinstance(tgt, ast.Name) and not _is_safe_name(tgt.id):
                                self.out.add(tgt.id)
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                        if not _is_safe_name(s.target.id):
                            self.out.add(s.target.id)
                self.generic_visit(node)
                self.depth -= 1
            def visit_Attribute(self, node):
                # self.X = ... or cls.X = ... (Store ctx)
                if (
                    self.depth > 0
                    and isinstance(node.ctx, ast.Store)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ('self', 'cls')
                    and not _is_safe_name(node.attr)
                ):
                    self.out.add(node.attr)
                self.generic_visit(node)
        _Coll(self.class_member_names).visit(tree)

    # -- pass 2: rewrite ----------------------------------------------------

    @property
    def scope(self) -> symtable.SymbolTable:
        return self.scope_stack[-1]

    def _scope_path_to_current(self) -> tuple:
        return tuple(self.scope_stack[:-1])

    def _maybe_mangle_name(self, name: str) -> str:
        """Look up `name` in the current scope chain and return its
        mangled form if its binding is marked for renaming."""
        if _is_safe_name(name) or name in ('self', 'cls'):
            return name
        bid = _resolve_binding(
            self._scope_path_to_current(), self.scope, name,
        )
        if bid is None:
            return name
        if not self.binding_to_mangle.get(bid, False):
            return name
        return self.pool.get_for(bid)

    def visit_Name(self, node):
        new = self._maybe_mangle_name(node.id)
        if new != node.id:
            node.id = new
        return node

    def visit_Nonlocal(self, node):
        # Each name in `nonlocal x, y` is a *reference* to the
        # enclosing function's binding. Resolve it the same way
        # visit_Name does (walk up parent scopes) and rewrite to the
        # mangled name.
        node.names = [self._maybe_mangle_name(n) for n in node.names]
        return node

    def visit_Global(self, node):
        # `global x` references the module-scope binding of x.
        node.names = [self._maybe_mangle_name(n) for n in node.names]
        return node

    def visit_ExceptHandler(self, node):
        # `except E as e:` binds `e` in the enclosing function/module
        # scope. Visit the type expr / body normally; rewrite the
        # binding name through the same scope-resolution path so it
        # matches the body's references to `e`.
        if node.type is not None:
            node.type = self.visit(node.type)
        if node.name is not None:
            new = self._maybe_mangle_name(node.name)
            if new != node.name:
                node.name = new
        node.body = [self.visit(s) for s in node.body]
        return node

    # Match patterns (PEP 634). Capture-style patterns bind names in
    # the enclosing scope, so they need the same mangle as `for`
    # targets / except-handler names.

    def visit_MatchAs(self, node):
        if node.pattern is not None:
            node.pattern = self.visit(node.pattern)
        if node.name is not None:
            new = self._maybe_mangle_name(node.name)
            if new != node.name:
                node.name = new
        return node

    def visit_MatchStar(self, node):
        if node.name is not None:
            new = self._maybe_mangle_name(node.name)
            if new != node.name:
                node.name = new
        return node

    def visit_MatchMapping(self, node):
        node.keys = [self.visit(k) for k in node.keys]
        node.patterns = [self.visit(p) for p in node.patterns]
        if node.rest is not None:
            new = self._maybe_mangle_name(node.rest)
            if new != node.rest:
                node.rest = new
        return node

    def visit_MatchClass(self, node):
        # `case Cls(field=pat)` — kwd_attrs lists field names to
        # match against the matched object's attributes. If methods
        # remapped those attribute names, the pattern's kwd_attrs
        # have to follow.
        node.cls = self.visit(node.cls)
        node.patterns = [self.visit(p) for p in node.patterns]
        node.kwd_patterns = [self.visit(p) for p in node.kwd_patterns]
        new_attrs = []
        for a in node.kwd_attrs:
            if 'methods' in self.tags and a in self.member_map:
                new_attrs.append(self.member_map[a])
            elif 'attrs' in self.tags and not _is_safe_name(a):
                if a not in self.member_map:
                    self.member_map[a] = self.pool.allocator()
                new_attrs.append(self.member_map[a])
            else:
                new_attrs.append(a)
        node.kwd_attrs = new_attrs
        return node

    def visit_arg(self, node):
        # Function parameter: a binding in the enclosing function scope.
        if 'locals' not in self.tags:
            return node
        if _is_safe_name(node.arg) or node.arg in ('self', 'cls'):
            return node
        bid = (_scope_key(self.scope), node.arg)
        if self.binding_to_mangle.get(bid, False):
            node.arg = self.pool.get_for(bid)
        return node

    def visit_FunctionDef(self, node):
        return self._visit_funclike(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit_funclike(node)

    def visit_Lambda(self, node):
        return self._visit_funclike(node, lambda_node=True)

    def _visit_funclike(self, node, lambda_node=False):
        # Decorators / defaults / annotations evaluate in the *outer*
        # scope. Visit them with the enclosing scope still on top.
        if not lambda_node:
            node.decorator_list = [self.visit(d) for d in node.decorator_list]
        for d in node.args.defaults:
            self.visit(d)
        for d in node.args.kw_defaults:
            if d is not None:
                self.visit(d)
        if not lambda_node and node.returns is not None:
            self.visit(node.returns)
        for a in (
            *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs
        ):
            if a.annotation is not None:
                self.visit(a.annotation)
        if node.args.vararg and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)

        # Look up the symtable child BEFORE renaming the function,
        # because _symtable_child matches by the original name.
        child = self._symtable_child(node)

        # Rename the function itself in the outer scope.
        if not lambda_node:
            outer_bid = (_scope_key(self.scope), node.name)
            if self.binding_to_mangle.get(outer_bid, False):
                node.name = self.pool.get_for(outer_bid)

        # Push the function's own scope.
        if child is not None:
            self.scope_stack.append(child)
        try:
            # Args (param.arg renaming)
            for a in (
                *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs
            ):
                self.visit(a)
            if node.args.vararg is not None:
                self.visit(node.args.vararg)
            if node.args.kwarg is not None:
                self.visit(node.args.kwarg)
            # Body
            if lambda_node:
                node.body = self.visit(node.body)
            else:
                node.body = [self.visit(s) for s in node.body]
        finally:
            if child is not None:
                self.scope_stack.pop()
        return node

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [
            ast.keyword(arg=kw.arg, value=self.visit(kw.value))
            for kw in node.keywords
        ]

        # Look up the symtable child BEFORE renaming the class.
        child = self._symtable_child(node)

        # Rename class itself.
        outer_bid = (_scope_key(self.scope), node.name)
        if self.binding_to_mangle.get(outer_bid, False):
            node.name = self.pool.get_for(outer_bid)

        if child is not None:
            self.scope_stack.append(child)
        try:
            node.body = [self.visit(s) for s in node.body]
            # `__slots__ = [...]` enumerates the attribute names the
            # class is allowed to set; if `methods` / `attrs` mangles
            # those, the literal strings need to follow.
            self._rewrite_slots(node)
        finally:
            if child is not None:
                self.scope_stack.pop()
        return node

    def _rewrite_slots(self, classdef: ast.ClassDef) -> None:
        if not self.member_map:
            return
        for stmt in classdef.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for tgt in stmt.targets:
                if not isinstance(tgt, ast.Name):
                    continue
                # __slots__ / __match_args__ enumerate attribute names
                # the class binds; if `methods` mangles those, the
                # string literals here have to follow.
                if tgt.id in ('__slots__', '__match_args__'):
                    stmt.value = self._rewrite_slots_value(stmt.value)

    def _rewrite_slots_value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            new = self.member_map.get(node.value)
            if new is not None:
                return ast.Constant(value=new)
            return node
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            node.elts = [self._rewrite_slots_value(e) for e in node.elts]
            return node
        if isinstance(node, ast.Dict):
            node.keys = [self._rewrite_slots_value(k) for k in node.keys]
            return node
        return node

    def _symtable_child(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = ('function', node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            key = ('class', node.name, node.lineno)
        elif isinstance(node, ast.Lambda):
            key = ('function', 'lambda', node.lineno)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            key = ('function', {
                ast.ListComp: 'listcomp', ast.SetComp: 'setcomp',
                ast.DictComp: 'dictcomp',
            }[type(node)], node.lineno)
        elif isinstance(node, ast.GeneratorExp):
            key = ('function', 'genexpr', node.lineno)
        else:
            return None
        # FunctionDef.name was mangled above; symtable uses the original.
        # We need to look up by what symtable saw, but we already
        # mutated node.name. Use the FunctionDef object's id hash —
        # too fragile. Easier: capture symtable mapping up front,
        # before any rewrite. (Done in __init__; here we just look
        # up by lineno + type — names match because symtable was
        # built on the *original* source, and we walk the AST in the
        # same order.) For now look up directly via children:
        for c in self.scope.get_children():
            ck = (c.get_type(), c.get_name(), c.get_lineno())
            if ck == key:
                return c
        return None

    # -- comprehension scopes ---------------------------------------------

    def visit_ListComp(self, node):
        return self._visit_comp(node)

    def visit_SetComp(self, node):
        return self._visit_comp(node)

    def visit_DictComp(self, node):
        return self._visit_comp(node)

    def visit_GeneratorExp(self, node):
        return self._visit_comp(node)

    def _visit_comp(self, node):
        # The *outermost* iter is evaluated in the enclosing scope.
        node.generators[0].iter = self.visit(node.generators[0].iter)
        child = self._symtable_child(node)
        if child is not None:
            self.scope_stack.append(child)
        try:
            for i, gen in enumerate(node.generators):
                if i > 0:
                    gen.iter = self.visit(gen.iter)
                gen.target = self.visit(gen.target)
                gen.ifs = [self.visit(c) for c in gen.ifs]
            if isinstance(node, ast.DictComp):
                node.key = self.visit(node.key)
                node.value = self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
        finally:
            if child is not None:
                self.scope_stack.pop()
        return node

    # -- imports -----------------------------------------------------------

    def visit_Import(self, node):
        if 'imports' in self.tags:
            for alias in node.names:
                # `import X.Y` binds the *top-level* X; later code
                # writes `X.Y.attr`. Renaming via `import X.Y as M`
                # would change the binding's identity (M = X.Y),
                # invalidating the X.Y.attr access pattern. Skip
                # dotted imports without an explicit asname — and
                # demote the binding so visit_Name doesn't rewrite
                # references to X either.
                if '.' in alias.name and alias.asname is None:
                    bound = alias.name.split('.')[0]
                    bid = (_scope_key(self.scope), bound)
                    self.binding_to_mangle[bid] = False
                    continue
                bound = alias.asname if alias.asname else alias.name.split('.')[0]
                if _is_safe_name(bound):
                    continue
                bid = (_scope_key(self.scope), bound)
                if self.binding_to_mangle.get(bid, False):
                    new = self.pool.get_for(bid)
                    alias.asname = new
        return node

    def visit_ImportFrom(self, node):
        if 'imports' in self.tags:
            for alias in node.names:
                if alias.name == '*':
                    continue
                bound = alias.asname if alias.asname else alias.name
                if _is_safe_name(bound):
                    continue
                bid = (_scope_key(self.scope), bound)
                if self.binding_to_mangle.get(bid, False):
                    new = self.pool.get_for(bid)
                    alias.asname = new
        return node

    # -- attributes --------------------------------------------------------

    def visit_Attribute(self, node):
        # Visit the value first (could itself contain attrs / names).
        node.value = self.visit(node.value)
        if _is_safe_name(node.attr):
            return node
        if 'attrs' in self.tags:
            # If mypy says the receiver is a user class → mangle.
            # If it says stdlib AND the attr name is not a member of
            # any user class → skip. This guards against a class of
            # mypy mistakes (rebinding `a = 1; a = MyClass(...)` ends
            # up keeping the int type) without trusting mypy beyond
            # what we can cross-check via the AST.
            if self.type_map is not None:
                info = self.type_map.lookup(
                    node.lineno, node.col_offset, node.attr,
                )
                # Receiver is definitely stdlib OR mypy gave up (Any) —
                # in either case, only mangle when the attr name is
                # also known as a member of some user class. This
                # protects stdlib API names like list.append while
                # still catching e.g. self.field on an unannotated
                # class.
                if info.kind in ('stdlib', 'any') and node.attr not in self.class_member_names:
                    return node
                if info.kind == 'user':
                    if node.attr not in self.member_map:
                        self.member_map[node.attr] = self.pool.allocator()
                    node.attr = self.member_map[node.attr]
                    return node
                # Fall through:
                #   - 'unknown' or no entry: mypy didn't classify this
                #     receiver. Be conservative and only rename when
                #     the attr is a known user-class member.
                #   - 'stdlib' / 'any' with attr in class_member_names:
                #     mangle (collision; the fact that we know a user
                #     class names this attr means the access is
                #     ambiguous, so renaming keeps the obfuscation
                #     consistent).
                if info.kind in ('stdlib', 'any'):
                    # attr IS in class_member_names — mangle.
                    if node.attr not in self.member_map:
                        self.member_map[node.attr] = self.pool.allocator()
                    node.attr = self.member_map[node.attr]
                    return node
                # 'unknown': only mangle when attr is a known
                # user-class member.
                if node.attr in self.class_member_names:
                    if node.attr not in self.member_map:
                        self.member_map[node.attr] = self.pool.allocator()
                    node.attr = self.member_map[node.attr]
                return node
            # No type map at all (mypy not installed): name-only
            # heuristic — rewrite EVERY non-dunder attribute.
            if node.attr not in self.member_map:
                self.member_map[node.attr] = self.pool.allocator()
            node.attr = self.member_map[node.attr]
        elif 'methods' in self.tags and node.attr in self.member_map:
            node.attr = self.member_map[node.attr]
        return node


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_mangle(tree: ast.AST, src: str, tags: set[str], pool: _NamePool,
                 type_map=None) -> ast.AST:
    """Run the configured user-code mangle pass over `tree`.

    `tree` must be the AST that was parsed from `src` — symtable is
    rebuilt from `src` (CPython's symtable C API needs source, not
    AST).

    `pool` allocates fresh names from the same Frame counter so the
    later passes don't collide. Returns the (mutated) tree.

    `type_map` is an optional `trans.infer.TypeMap` (built by mypy
    when available); used to refine `attrs` so stdlib API names
    aren't renamed.
    """
    if not tags - {'helper'}:
        # Nothing for this pass to do — only the helper tag is set,
        # which add_helper handles itself.
        return tree
    rewriter = _Rewriter(src, tags, pool, type_map=type_map)
    rewriter.visit(tree)
    return tree
