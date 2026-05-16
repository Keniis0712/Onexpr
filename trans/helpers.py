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
            func_helper_name, for_helper_name, while_helper_name
        ):
            _mark_legacy_return(stmt)
            kept.append(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            # Module-level helper functions (e.g. _del_local). These
            # appear after the helper classes, so they CAN reference
            # _FuncHelper through the regular transform path.
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
    )
    needs_for = (
        ast.For in detector.presence
        or needs_try_runtime
        or ast.ClassDef in detector.presence  # _make_class iterates bases
    )
    needs_while = ast.While in detector.presence or needs_try_runtime

    # Pull in the three core helper classes; each is annotated for the
    # legacy return convention so transforming them doesn't reference
    # _FuncHelper before it's bound. Module-level helper functions
    # (e.g. _del_local) are pulled in on demand based on what the user
    # code uses.
    core_body = _get_core_helper_body()
    core_by_name = {s.name: s for s in core_body}

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
        to_insert.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=LIB_COLLECTIONS, ctx=ast.Load()),
                                attr='abc',
                                ctx=ast.Load()
                            ),
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

    tree.body = to_insert + tree.body
