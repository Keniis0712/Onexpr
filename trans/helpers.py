import ast
import os

from .passes import NodePresenceDetector


def _load_runtime_module(filename: str) -> list:
    """Read a sibling runtime/<filename>.py, parse it, and return its
    top-level statement list. The file is processed at transform time
    (not imported), so its statements get woven into the user's tree
    and pass through the full onexpr transformation."""
    path = os.path.join(os.path.dirname(__file__), 'runtime', filename)
    with open(path, encoding='utf-8') as f:
        src = f.read()
    return ast.parse(src).body


try_helper_name = '_TryHelper'


def _get_try_helper_body() -> list:
    # Always re-parse — see _get_core_helper_body for why caching the
    # parsed copy doesn't work.
    return _load_runtime_module('try_helper.py')


def _get_typealias_body() -> list:
    return _load_runtime_module('typealias.py')


def _has_pep695(tree: ast.AST) -> bool:
    """True iff the tree uses PEP 695 syntax: `type X = ...`,
    `def f[T](...)`, or `class C[T]:`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.TypeAlias):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if getattr(node, 'type_params', None):
                return True
    return False


def _has_async_generator(tree: ast.AST) -> bool:
    """True iff the tree contains an AsyncFunctionDef whose body uses
    yield / yield from directly. Such functions are async generators
    (PEP 525) and need a different shape than plain async def."""
    class _V(ast.NodeVisitor):
        def __init__(self):
            self.found = False
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
    """True iff the tree contains a FunctionDef whose body uses
    yield / yield from directly (not in a nested function), or any
    AsyncFunctionDef (which we lower to a generator state machine)."""
    class _V(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Lambda(self, node):
            pass
        def visit_Yield(self, node):
            self.found = True
        def visit_YieldFrom(self, node):
            self.found = True

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


for_helper_name = '_ForHelper'
while_helper_name = '_WhileHelper'
func_helper_name = '_FuncHelper'

LIB_TYPES = '_types'
LIB_COLLECTIONS = '_collections'
LIB_INSPECT = '_inspect'


def _get_core_helper_body() -> list:
    """Top-level statements of trans/runtime/core.py: the three helper
    class definitions. Each ClassDef (and every method nested inside)
    is annotated with `_use_legacy_return = True`, so parse_class_def /
    parse_function_def take the older `(value, True)` tuple-and-[0]
    return convention and avoid referring to _FuncHelper while
    _FuncHelper is itself being defined."""
    # Always re-parse — parse_class_def mutates the body (appends
    # `return locals()`), so reusing a single parsed copy across
    # transforms accumulates injected returns.
    body = _load_runtime_module('core.py')
    kept = []
    for stmt in body:
        if isinstance(stmt, ast.ClassDef) and stmt.name in (
            func_helper_name, for_helper_name, while_helper_name,
            '_AsyncGenWrapper', '_UserYield',
        ):
            if stmt.name in (func_helper_name, for_helper_name, while_helper_name):
                _mark_legacy_return(stmt)
            kept.append(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            # Module-level helper functions (e.g. _del_local). These
            # appear after the helper classes, so they CAN reference
            # _FuncHelper through the regular transform path.
            kept.append(stmt)
        elif isinstance(stmt, ast.Assign):
            # Module-level Assigns in core.py (e.g. the
            # types.coroutine wrap on _async_gen_anext). Keep them so
            # they get spliced in alongside their owning function.
            kept.append(stmt)
    return kept


def _mark_legacy_return(node: ast.AST) -> None:
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n._use_legacy_return = True


def add_helper(tree: ast.AST, top_func_helper_var: str):
    # Get all used node types
    detector = NodePresenceDetector()
    detector.visit(tree)

    # The try-helper runtime contains for-loops and while-loops, so
    # injecting it pulls in _ForHelper / _WhileHelper as a transitive
    # dependency. `with` also uses the try-helper (its with_block).
    # _make_class also has a for-loop in its body (to walk bases),
    # and we always inject _make_class.
    needs_try_runtime = (
        ast.Try in detector.presence
        or ast.TryStar in detector.presence
        or ast.With in detector.presence
        or ast.AsyncWith in detector.presence
        # Generator state machines wrap their dispatch in a
        # try/except so user except clauses inside a generator body
        # can dispatch on caught exceptions.
        or _has_generator_function(tree)
    )
    # _make_class is always injected (it instantiates every emitted
    # ClassDef, including the helpers themselves and the generator
    # state-machine class). Its body uses a for-loop, so we need
    # _ForHelper unconditionally.
    needs_for = True
    needs_while = (
        ast.While in detector.presence
        or needs_try_runtime
        or _has_generator_function(tree)  # state machine uses while True
    )

    # Pull in the three core helper classes; each is annotated for the
    # legacy return convention so transforming them doesn't reference
    # _FuncHelper before it's bound. Module-level helper functions
    # (e.g. _del_local) are pulled in on demand based on what the user
    # code uses.
    core_body = _get_core_helper_body()
    core_by_name = {}
    core_extras = []
    for s in core_body:
        if hasattr(s, 'name'):
            core_by_name[s.name] = s
        else:
            core_extras.append(s)

    to_insert = [core_by_name[func_helper_name]]
    if needs_for:
        to_insert.append(core_by_name[for_helper_name])
    if needs_while:
        to_insert.append(core_by_name[while_helper_name])
    if ast.Delete in detector.presence and '_del_local' in core_by_name:
        to_insert.append(core_by_name['_del_local'])
    # _make_class is needed by every ClassDef parse_class_def emits,
    # including the helper classes we always inject. Always include it.
    if '_make_class' in core_by_name:
        to_insert.append(core_by_name['_make_class'])

    # Generator state machines need a sentinel to distinguish "iterator
    # exhausted" from "iterator yielded None" — built via next(it,
    # sentinel). Define it module-level so every generator class can
    # see it.
    if _has_generator_function(tree):
        to_insert.append(
            ast.Assign(
                targets=[ast.Name(id='_GEN_DONE_SENTINEL', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='object', ctx=ast.Load()),
                    args=[], keywords=[],
                ),
            )
        )

    if ast.AsyncFunctionDef in detector.presence:
        to_insert.append(ast.Import(
            names=[
                ast.alias(
                    name='types',
                    asname=LIB_TYPES,
                ),
                ast.alias(
                    name='collections.abc',
                    asname=LIB_COLLECTIONS,
                ),
                ast.alias(
                    name='inspect',
                    asname=LIB_INSPECT,
                )
            ]
        ))
        if '_await_iter' in core_by_name:
            to_insert.append(core_by_name['_await_iter'])
        to_insert.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id=LIB_COLLECTIONS, ctx=ast.Load()),
                            attr='Coroutine',
                            ctx=ast.Load()
                        ),
                        attr='register',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id=LIB_TYPES, ctx=ast.Load()),
                            attr='GeneratorType',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[],
                )
            )
        )

        # If the user has an `async def` with `yield` (an async
        # generator), pull in _AsyncGenWrapper + _async_gen_anext +
        # the types.coroutine wrap assign.
        if _has_async_generator(tree):
            for n in ('_UserYield', '_AsyncGenWrapper', '_async_gen_anext'):
                if n in core_by_name:
                    to_insert.append(core_by_name[n])
            # The Assign that wraps _async_gen_anext via types.coroutine
            # lives in core_extras (it's an Assign, not a def).
            for s in core_extras:
                # All extras are tied to async-gen plumbing right now,
                # so include them here.
                to_insert.append(s)

    to_insert.append(
        ast.Assign(
            targets=[ast.Name(id=top_func_helper_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=func_helper_name, ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
        )
    )

    # Inject the try/except runtime helper if the user code uses `try`.
    # Unlike the for/while/func helpers (whose class definitions are
    # exec'd as a string blob to avoid chicken-and-egg with _FuncHelper),
    # the try helper is a regular set of top-level statements that goes
    # through the full onexpr transformation along with user code.
    if needs_try_runtime:
        to_insert = to_insert + _get_try_helper_body()

    if _has_pep695(tree):
        to_insert = to_insert + _get_typealias_body()

    tree.body = to_insert + tree.body
