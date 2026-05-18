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
    and pass through the full onexpr transformation."""
    path = os.path.join(os.path.dirname(__file__), 'runtime', filename)
    with open(path, encoding='utf-8') as f:
        src = f.read()
    return ast.parse(src).body


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
# Main entry point
# ---------------------------------------------------------------------------

def add_helper(tree: ast.AST, top_frame):
    """Insert the runtime helper classes/functions at the top of `tree`.

    `top_frame` carries the reserved-name set and receives the
    `helper_names` map this function builds. Any helper name that
    collides with a name the user uses is rewritten to a fresh temp_N
    in the loaded AST, and parsers consult `frame.get_helper_name(...)`
    to discover the renamed identifier later.
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
    rename_map: dict[str, str] = {}
    for key in ordered:
        for orig in _REGISTRY[key]['names']:
            if orig in rename_map:
                continue
            if orig in reserved:
                rename_map[orig] = top_frame.get_temp_var()
            else:
                rename_map[orig] = orig
    top_frame.helper_names = rename_map

    # Load and (optionally) mark each helper's stmts. Apply the rename
    # so the helper class/function defs and their internal references
    # all use the chosen identifier.
    to_insert: list[ast.stmt] = []
    for key in ordered:
        entry = _REGISTRY[key]
        stmts = _load_runtime_module(entry['file'])
        for s in stmts:
            _rename_helper_refs(s, rename_map)
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


def _rename_helper_refs(node: ast.AST, rename_map: dict[str, str]) -> None:
    """Walk `node` and rename any identifier that maps in `rename_map`.

    Targets:
      - ast.Name (load and store) — the helper's own definition site
        (`class _FuncHelper:` shows up as ClassDef, not Name) AND every
        intra-helper reference (`_TryHelper.guarded(...)`,
        `_async_gen_anext(self._g)`, etc.)
      - ast.ClassDef.name / ast.FunctionDef.name — the definitions
      - ast.AsyncFunctionDef.name — generator-protocol coroutine forwarders

    Doesn't touch attribute names (`obj.foo`) or string constants —
    those are intentional public surface, never helper identifiers.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            new = rename_map.get(n.id)
            if new is not None and new != n.id:
                n.id = new
        elif isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            new = rename_map.get(n.name)
            if new is not None and new != n.name:
                n.name = new
