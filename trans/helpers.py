import ast
import os

from .passes import NodePresenceDetector


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def _load_runtime_module(filename: str) -> list:
    """Read a sibling runtime/<filename>.py, parse it, and return its
    top-level statement list. The file is processed at transform time
    (not imported), so its statements get woven into the user's tree
    and pass through the full onexpr transformation.

    Strips docstrings and type annotations on the way out — helpers are
    internal scaffolding, the user-facing output doesn't need those."""
    path = os.path.join(os.path.dirname(__file__), 'runtime', filename)
    with open(path, encoding='utf-8') as f:
        src = f.read()
    body = ast.parse(src).body
    # Drop module-level docstring (if any).
    body = _strip_leading_docstring(body)
    for node in body:
        _strip_annotations_and_docs(node)
    return body


def _strip_leading_docstring(stmts: list) -> list:
    """Return `stmts` with its leading string-literal Expr removed (the
    docstring of a module / class / function body)."""
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:]
    return stmts


def _strip_annotations_and_docs(node: ast.AST) -> None:
    """Walk `node` in place, deleting:
      - parameter / return type annotations on every def/lambda
      - leading docstrings inside class / function bodies
      - bare `x: T` (AnnAssign without value) — it produces no behavior;
        for `x: T = v` we drop the annotation but keep the assignment

    Only sensible for helper code we synthesize. User code should never
    pass through this — annotations may be load-bearing (FastAPI,
    dataclasses, etc.)."""
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = n.args
            for group in (args.posonlyargs, args.args, args.kwonlyargs):
                for a in group:
                    a.annotation = None
            if args.vararg is not None:
                args.vararg.annotation = None
            if args.kwarg is not None:
                args.kwarg.annotation = None
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n.returns = None
                n.body = _strip_leading_docstring(n.body)
                n.body = _drop_bare_annassigns(n.body)
        elif isinstance(n, ast.ClassDef):
            n.body = _strip_leading_docstring(n.body)
            n.body = _drop_bare_annassigns(n.body)


def _drop_bare_annassigns(stmts: list) -> list:
    """Replace `x: T = v` with `x = v`, drop bare `x: T`."""
    out = []
    for s in stmts:
        if isinstance(s, ast.AnnAssign):
            if s.value is None:
                continue
            out.append(ast.Assign(
                targets=[ast.Name(id=s.target.id, ctx=ast.Store())]
                if isinstance(s.target, ast.Name) else [s.target],
                value=s.value,
            ))
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def _has_pep695(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.TypeAlias):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, 'type_params', None):
                return True
    return False


def _has_async_generator(tree: ast.AST) -> bool:
    class _V(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Lambda(self, node): pass
        def visit_Yield(self, node): self.found = True
        def visit_YieldFrom(self, node): self.found = True

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            v = _V()
            for s in node.body:
                v.visit(s)
                if v.found:
                    return True
    return False


def _has_generator_function(tree: ast.AST) -> bool:
    class _V(ast.NodeVisitor):
        def __init__(self): self.found = False
        def visit_Lambda(self, node): pass
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_Yield(self, node): self.found = True
        def visit_YieldFrom(self, node): self.found = True

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            v = _V()
            for s in node.body:
                v.visit(s)
                if v.found:
                    return True
    return False


def _has_bare_raise(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is None:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper name constants (used by parsers / gen_compile)
# ---------------------------------------------------------------------------

for_helper_name   = '_ForHelper'
while_helper_name = '_WhileHelper'
func_helper_name  = '_FuncHelper'
try_helper_name   = '_TryHelper'

LIB_TYPES       = '_types'
LIB_COLLECTIONS = '_collections'
LIB_INSPECT     = '_inspect'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry:
#   file      – runtime/<file>.py to load
#   names     – top-level names exported (used for dedup / ordering)
#   condition – callable(tree, presence) -> bool; None means always
#   deps      – list of entry keys that must come before this one
#   legacy    – if True, mark every FunctionDef/ClassDef with _use_legacy_return
#
# Topological injection order is derived from deps.

_REGISTRY = {
    # ---- always-needed ----
    'func_helper': {
        'file': 'func_helper.py',
        'names': ['_FuncHelper'],
        'condition': None,
        'deps': [],
        'legacy': True,
    },
    'for_helper': {
        'file': 'for_helper.py',
        'names': ['_ForHelper'],
        # Always inject: _make_class uses it, and it's tiny.
        'condition': None,
        'deps': ['func_helper'],
        'legacy': True,
    },
    'while_helper': {
        'file': 'while_helper.py',
        'names': ['_WhileHelper'],
        'condition': lambda tree, p: (
            ast.While in p
            or ast.With in p
            or ast.AsyncWith in p
            or _has_generator_function(tree)
        ),
        'deps': ['func_helper'],
        'legacy': True,
    },
    'del_local': {
        'file': 'del_local.py',
        'names': ['_del_local'],
        'condition': lambda tree, p: ast.Delete in p,
        'deps': [],
        'legacy': False,
    },
    'make_class': {
        'file': 'make_class.py',
        'names': ['_CELL_EMPTY', '_make_class'],
        # Always inject: every ClassDef emits a _make_class call.
        'condition': None,
        'deps': ['func_helper', 'for_helper'],
        'legacy': False,
    },
    'gen_sentinel': {
        'file': 'gen_sentinel.py',
        'names': ['_GEN_DONE_SENTINEL'],
        'condition': lambda tree, p: _has_generator_function(tree),
        'deps': [],
        'legacy': False,
    },
    'try_helper': {
        'file': 'try_helper.py',
        'names': ['_TryHelper'],
        'condition': lambda tree, p: (
            ast.Try in p
            or ast.TryStar in p
            or ast.With in p
            or ast.AsyncWith in p
            or _has_generator_function(tree)
            or _has_bare_raise(tree)
        ),
        'deps': ['func_helper', 'for_helper', 'while_helper'],
        'legacy': False,
    },
    'await_iter': {
        'file': 'await_iter.py',
        'names': ['_await_iter'],
        'condition': lambda tree, p: ast.AsyncFunctionDef in p,
        'deps': [],
        'legacy': False,
    },
    'async_gen': {
        'file': 'async_gen.py',
        'names': [
            '_UserYield', '_AsyncGenWrapper',
            '_async_gen_anext', '_async_gen_asend',
            '_async_gen_athrow', '_async_gen_aclose',
        ],
        'condition': lambda tree, p: _has_async_generator(tree),
        'deps': [],
        'legacy': False,
    },
    'typealias': {
        'file': 'typealias.py',
        'names': ['_LazyAlias', '_TAProxy'],
        'condition': lambda tree, p: _has_pep695(tree),
        'deps': [],
        'legacy': False,
    },
    'inspect_patch': {
        'file': 'inspect_patch.py',
        'names': ['_onexpr_patched_has_code_flag'],
        'condition': lambda tree, p: _has_generator_function(tree),
        'deps': [],
        'legacy': False,
    },
}


def _topo_sort(keys: list[str]) -> list[str]:
    """Return keys in dependency order (deps before dependents).
    Deps are pulled in transitively even if their own condition was False."""
    result = []
    visited = set()

    def visit(k):
        if k in visited:
            return
        visited.add(k)
        for dep in _REGISTRY[k]['deps']:
            visit(dep)   # always follow deps, regardless of keys_set
        result.append(k)

    for k in keys:
        visit(k)
    return result


def _mark_legacy_return(node: ast.AST) -> None:
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n._use_legacy_return = True


# ---------------------------------------------------------------------------
# Helper member names — mangle-eligible identifiers that live INSIDE the
# helper sources. Method names, self-attribute names, the boxing-prefix.
# Listed centrally so add_helper can build a rename map driven by
# --replace-name.
#
# What's safe to mangle: identifiers used only between helper code and
# parser-emitted code (we control both ends). What's NOT in this list:
# Python protocol dunders (__init__, __aiter__, ...) and stdlib API names
# (list.append, generator.send, asyncgen.asend, ...). Those reach
# external code and renaming them would crash.
HELPER_MEMBERS = [
    # _FuncHelper
    'returned', 'value', 'do_return',
    # _ForHelper / _WhileHelper shared
    'iterable', 'stopped', 'func_helper', 'last_yielded', 'was_iterated',
    'pending_continue', 'stop', 'do_continue', 'ended', 'cond',
    # _TryHelper
    '_loop', '_exc_stack', '_get_loop', 'guarded', 'dispatch',
    'dispatch_star', 'with_block',
    # _AsyncGenWrapper internal
    '_g',
    # _LazyAlias internal
    '_thunk', '_evaluated', '_cached',
    # gen_compile state-machine self-fields (set/read both in emit.py
    # and in user code translated against them)
    'state', '_exc', '_sent', '_stopping_via_return',
]
HELPER_BOX_PREFIX = '_b_'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_helper(tree: ast.AST, top_frame, replace_name: str = 'none'):
    """Insert the runtime helper classes/functions at the top of `tree`.

    `top_frame` carries the reserved-name set and receives the
    `helper_names` / `helper_members` / `helper_box_prefix` maps this
    function builds. Any helper name that collides with a name the
    user uses is rewritten to a fresh temp_N in the loaded AST, and
    parsers consult `frame.get_helper_name(...)` /
    `frame.get_helper_member(...)` to discover the renamed identifier
    later.

    `replace_name`:
      - 'none' (default): only collision-driven renames; identity for
        everything else.
      - 'global': every helper class name + every member identifier in
        HELPER_MEMBERS + the box prefix gets a fresh temp_N name. Use
        when the bundle is intended as a self-contained / obfuscated
        artifact (no external introspection).
    """
    detector = NodePresenceDetector()
    detector.visit(tree)
    presence = detector.presence

    # Determine which helpers are needed.
    needed = []
    for key, entry in _REGISTRY.items():
        cond = entry['condition']
        if cond is None or cond(tree, presence):
            needed.append(key)

    ordered = _topo_sort(needed)

    # Build the rename map. Any helper-exported name colliding with a
    # name the user already uses (collected by collect_user_names) gets
    # remapped to a fresh temp_N. Even helpers whose names don't
    # collide get an identity entry, so frame.get_helper_name() returns
    # the correct emitted name uniformly.
    reserved = top_frame.reserved_names
    mangle_all = (replace_name == 'global')
    rename_map: dict[str, str] = {}
    for key in ordered:
        for orig in _REGISTRY[key]['names']:
            if orig in rename_map:
                continue
            if mangle_all or orig in reserved:
                rename_map[orig] = top_frame.get_temp_var()
            else:
                rename_map[orig] = orig
    top_frame.helper_names = rename_map

    # Member rename map — only diverges from identity when mangle is on.
    member_map: dict[str, str] = {}
    for orig in HELPER_MEMBERS:
        if mangle_all:
            member_map[orig] = top_frame.get_temp_var()
        else:
            member_map[orig] = orig
    top_frame.helper_members = member_map

    if mangle_all:
        top_frame.helper_box_prefix = top_frame.get_temp_var() + '_'
    else:
        top_frame.helper_box_prefix = HELPER_BOX_PREFIX

    # Load and (optionally) mark each helper's stmts. Apply the rename
    # so the helper class/function defs and their internal references
    # all use the chosen identifier.
    to_insert: list[ast.stmt] = []
    for key in ordered:
        entry = _REGISTRY[key]
        stmts = _load_runtime_module(entry['file'])
        for s in stmts:
            _rename_helper_refs(s, rename_map, member_map)
        if entry['legacy']:
            for s in stmts:
                _mark_legacy_return(s)
        to_insert.extend(stmts)

    # Coroutine registration: when AsyncFunctionDef is present, register
    # GeneratorType as a Coroutine so our fake coroutines are awaitable.
    if ast.AsyncFunctionDef in presence:
        to_insert.append(ast.Import(names=[
            ast.alias(name='types',          asname=LIB_TYPES),
            ast.alias(name='collections.abc', asname=LIB_COLLECTIONS),
            ast.alias(name='inspect',         asname=LIB_INSPECT),
        ]))
        to_insert.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id=LIB_COLLECTIONS, ctx=ast.Load()),
                    attr='Coroutine', ctx=ast.Load(),
                ),
                attr='register', ctx=ast.Load(),
            ),
            args=[ast.Attribute(
                value=ast.Name(id=LIB_TYPES, ctx=ast.Load()),
                attr='GeneratorType', ctx=ast.Load(),
            )],
            keywords=[],
        )))

    # Module-level _FuncHelper instance (the top-level "function" helper).
    to_insert.append(ast.Assign(
        targets=[ast.Name(id=top_frame.func_helper_var, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id=rename_map[func_helper_name], ctx=ast.Load()),
            args=[], keywords=[],
        ),
    ))

    tree.body = to_insert + tree.body


def _rename_helper_refs(node: ast.AST, rename_map: dict[str, str],
                        member_map: dict[str, str] | None = None) -> None:
    """Walk `node` and rename identifiers that map in `rename_map` /
    `member_map`.

    `rename_map` covers helper class / function names (top-level
    bindings the parsers reference by Name).

    `member_map` covers helper class members (method names, self-attr
    names). Applies to:
      - FunctionDef.name / AsyncFunctionDef.name *inside a ClassDef*
        (method definitions)
      - Attribute.attr — only when the receiver is known to be a
        helper (`self.X` inside a helper method, or `_FuncHelper.X`
        / `_TryHelper.X` / etc. via the renamed class). External
        receivers (asyncio loops, stdlib types, generators returned
        from user code) keep their attribute names so the runtime
        bridge to stdlib still works.
      - keyword.arg (`Foo(returned=True)` — kw goes to the helper's
        __init__)
      - arg.arg (constructor parameters that share a name with the
        attribute they're assigned to, like `func_helper`)
    """
    members = member_map or {}
    # Receivers whose attribute access we know targets helper
    # internals: `self` (inside any helper method), the param-name
    # convention `func_helper` / `loop_helper` (parameters of
    # _TryHelper.dispatch that always carry helper instances), and
    # any helper class name that lives in rename_map.
    helper_receivers = (
        {'self', 'func_helper', 'loop_helper'}
        | set(rename_map.keys()) | set(rename_map.values())
    )
    # Field names that, when used as `<obj>.<field>`, are known to
    # return a helper instance. Used to walk transitive chains like
    # `self.func_helper.returned` — outer `.returned` should mangle.
    helper_field_names = {'func_helper', 'loop_helper'}

    def _is_helper_receiver(expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in helper_receivers
        if isinstance(expr, ast.Attribute):
            return expr.attr in helper_field_names
        return False
    # We need to know whether a FunctionDef sits directly inside a
    # ClassDef to rename its name (a top-level FunctionDef in a runtime
    # module is, e.g., _del_local — already handled by rename_map).
    class_body_funcs: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.ClassDef):
            for s in n.body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_body_funcs.add(id(s))

    # Track ast.Call nodes whose function is a helper class so we can
    # rewrite their keyword.arg names (helper-class kw args correspond
    # to helper member identifiers; stdlib calls don't).
    helper_call_kw: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in helper_receivers:
                for kw in n.keywords:
                    helper_call_kw.add(id(kw))

    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            new = rename_map.get(n.id)
            if new is None:
                # Inside helper bodies, parameter names that match a
                # member entry (e.g. `iterable`, `func_helper`) need to
                # track the renamed `arg.arg` so `self.iterable =
                # iterable` keeps working. Safe here because this
                # walker only ever runs on runtime helper sources.
                new = members.get(n.id)
            if new is not None and new != n.id:
                n.id = new
        elif isinstance(n, ast.ClassDef):
            new = rename_map.get(n.name)
            if new is not None and new != n.name:
                n.name = new
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if id(n) in class_body_funcs:
                new = members.get(n.name)
            else:
                new = rename_map.get(n.name)
            if new is not None and new != n.name:
                n.name = new
        elif isinstance(n, ast.Attribute):
            # Only rewrite when we can prove the receiver is a helper.
            # `self.X` and `<HelperClass>.X` qualify; `loop.stop()` /
            # `g.send()` / `mgr.__exit__()` don't. Transitive chains
            # like `self.func_helper.returned` also qualify because
            # `func_helper` / `loop_helper` are known helper-instance
            # fields.
            if _is_helper_receiver(n.value):
                new = members.get(n.attr)
                if new is not None and new != n.attr:
                    n.attr = new
        elif isinstance(n, ast.keyword):
            if n.arg is not None and id(n) in helper_call_kw:
                new = members.get(n.arg)
                if new is not None and new != n.arg:
                    n.arg = new
        elif isinstance(n, ast.arg):
            new = members.get(n.arg)
            if new is not None and new != n.arg:
                n.arg = new
