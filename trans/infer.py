"""Optional mypy-based type inference.

Used by the mangle pass to make the `attrs` tag selective:
  - `obj.append` where obj is `list` → leave as-is (stdlib API)
  - `obj.greet` where obj is a user-defined class → mangle

If mypy isn't installed, `analyze()` returns None and the mangle pass
falls back to its name-only heuristic.

This module isolates every mypy import inside `analyze()` so the rest
of trans/ can be loaded without mypy on PYTHONPATH.
"""
from __future__ import annotations

import ast
import os
import tempfile
from dataclasses import dataclass


@dataclass
class AttrInfo:
    """What the type checker says about one `obj.attr` access.

    `kind` is one of:
      'user'    obj is an instance of a class defined in *this* source.
      'stdlib'  obj's type lives in the standard library / third-party
                code (not in this file). Don't mangle the attribute.
      'any'     mypy gave up on the receiver's type (no annotation,
                untyped imported library, ...). Treat conservatively:
                let the caller decide based on whether `attr` is also
                known as a user-class member.
      'unknown' nothing useful.
    """
    kind: str
    user_class: str | None = None  # the class name, when kind == 'user'


class TypeMap:
    """Lookup table from `(line, col, attr_name)` to AttrInfo.

    The position triple matches stdlib `ast.Attribute` (lineno,
    col_offset, attr) — mypy's `MemberExpr.line / column / name`
    coincide with those, so this is the right key.
    """

    def __init__(self, entries: dict[tuple[int, int, str], AttrInfo],
                 user_classes: set[str]):
        self._entries = entries
        self._user_classes = user_classes

    def lookup(self, lineno: int, col_offset: int, attr: str) -> AttrInfo:
        return self._entries.get((lineno, col_offset, attr),
                                 AttrInfo(kind='unknown'))

    @property
    def user_classes(self) -> set[str]:
        return self._user_classes


def analyze_package(pkg_root, package: str) -> dict | None:
    """Run mypy over an entire package and return
    {module_dotted: TypeMap}, or None if mypy isn't installed.

    Importantly this analyses the whole tree at once so relative
    imports / cross-module class references resolve. The per-module
    TypeMap entries can then drive a per-module mangle pass.
    """
    try:
        from mypy.build import build  # type: ignore
        from mypy.main import process_options  # type: ignore
        from mypy.modulefinder import BuildSource  # type: ignore
        from mypy.nodes import (  # type: ignore
            Expression, Statement, MemberExpr, NameExpr, MypyFile,
        )
        from mypy.types import Instance, AnyType  # type: ignore
    except ImportError:
        return None

    from pathlib import Path
    pkg_root = Path(pkg_root)
    pkg_dir = pkg_root / package
    if not pkg_dir.is_dir():
        return None

    # Build the source list: walk pkg_dir for *.py.
    sources_list = []
    file_to_module: dict = {}
    for path in sorted(pkg_dir.rglob('*.py')):
        rel = path.relative_to(pkg_root)
        # Convert path to dotted module name. `__init__.py` →
        # parent dir name.
        parts = list(rel.with_suffix('').parts)
        if parts[-1] == '__init__':
            parts = parts[:-1]
        if not parts:
            continue
        dotted = '.'.join(parts)
        sources_list.append(BuildSource(str(path), dotted, None))
        file_to_module[str(path)] = dotted

    if not sources_list:
        return None

    try:
        _sources, options = process_options([str(s.path) for s in sources_list])
    except SystemExit:
        return None
    options.export_types = True
    options.preserve_asts = True
    options.incremental = False
    options.show_traceback = False

    try:
        result = build(sources_list, options)
    except Exception:
        return None
    if result is None or result.types is None:
        return None

    out: dict = {}
    for dotted, _ in [(s.module, s) for s in sources_list]:
        state = result.graph.get(dotted)
        if state is None or state.tree is None:
            continue
        out[dotted] = _build_typemap_from_state(
            state, result.types, dotted,
            MemberExpr, NameExpr, MypyFile, Instance, AnyType,
            Expression, Statement,
        )
    return out


def _build_typemap_from_state(state, types_map, module_name,
                              MemberExpr, NameExpr, MypyFile,
                              Instance, AnyType, Expression, Statement) -> 'TypeMap':
    """Shared between analyze() and analyze_package()."""
    user_node_ids: set[int] = set()
    all_member_exprs: list = []
    seen: set[int] = set()

    def walk(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, (Expression, Statement)):
            user_node_ids.add(id(n))
        if isinstance(n, MemberExpr):
            all_member_exprs.append(n)
        for k in dir(n):
            if k.startswith('_'):
                continue
            try:
                v = getattr(n, k, None)
            except Exception:
                continue
            if isinstance(v, (Expression, Statement)):
                walk(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, (Expression, Statement)):
                        walk(x)

    walk(state.tree)

    entries: dict = {}
    user_classes: set = set()
    try:
        user_tree = ast.parse(state.source) if hasattr(state, 'source') and state.source else None
    except Exception:
        user_tree = None
    if user_tree is None:
        # Fall back to walking mypy AST for ClassDef
        from mypy.nodes import ClassDef as _MCD  # type: ignore
        for n in state.tree.defs if hasattr(state.tree, 'defs') else ():
            if isinstance(n, _MCD):
                user_classes.add(n.name)
    else:
        for n in ast.walk(user_tree):
            if isinstance(n, ast.ClassDef):
                user_classes.add(n.name)

    for expr in all_member_exprs:
        recv = expr.expr
        recv_node = getattr(recv, 'node', None)
        if recv_node is not None and isinstance(recv_node, MypyFile):
            entries[(expr.line, expr.column, expr.name)] = AttrInfo(kind='stdlib')
            continue
        recv_ty = types_map.get(recv)
        if recv_ty is None:
            continue
        kind: str
        cls_name: str | None = None
        if isinstance(recv_ty, Instance):
            # User-class iff the class lives in the same package
            # we're analysing. We compare module_name prefix —
            # both `pkg.api.MyClass` and `pkg.MyClass` count.
            if recv_ty.type.module_name == module_name or (
                '.' in module_name and recv_ty.type.module_name.startswith(
                    module_name.rsplit('.', 1)[0] + '.'
                )
            ) or recv_ty.type.module_name.startswith(
                module_name.split('.', 1)[0] + '.'
            ) or recv_ty.type.module_name == module_name.split('.', 1)[0]:
                kind = 'user'
                cls_name = recv_ty.type.name
            else:
                kind = 'stdlib'
        elif isinstance(recv_ty, AnyType):
            kind = 'any'
        else:
            kind = 'unknown'
        entries[(expr.line, expr.column, expr.name)] = AttrInfo(
            kind=kind, user_class=cls_name,
        )
    return TypeMap(entries, user_classes)


def analyze(src: str, *, module_name: str = '_onexpr_probe') -> TypeMap | None:
    """Run mypy over `src` and build a TypeMap.

    Returns None if mypy isn't installed or the analysis fails (we
    don't want type errors in user code to abort the whole obfuscation
    — fall back to the name-only heuristic instead).
    """
    try:
        from mypy.build import build  # type: ignore
        from mypy.main import process_options  # type: ignore
        from mypy.modulefinder import BuildSource  # type: ignore
        from mypy.nodes import (  # type: ignore
            Expression, Statement, MemberExpr, NameExpr, MypyFile,
        )
        from mypy.types import Instance, AnyType  # type: ignore
    except ImportError:
        return None

    # mypy needs the source on disk (BuildSource(path, module, src) is
    # supposed to bypass that, but in practice options come from the
    # path arg too — so we write a temp file and clean up afterwards).
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8',
        ) as f:
            tmp = f.name
            f.write(src)

        try:
            sources, options = process_options([tmp])
        except SystemExit:
            return None
        options.export_types = True
        options.preserve_asts = True
        options.incremental = False
        options.show_traceback = False
        # Don't let mypy's exit code on errors abort us.
        options.warn_unused_ignores = False
        options.disable_error_code = ['misc']

        sources = [BuildSource(tmp, module_name, src)]
        try:
            result = build(sources, options)
        except Exception:
            return None
        if result is None or result.types is None:
            return None

        state = result.graph.get(module_name)
        if state is None or state.tree is None:
            return None

        # Walk the user file's mypy AST collecting every node id; we
        # use this to filter result.types (which is global) AND to
        # find every MemberExpr we want to classify (some module
        # accesses like the inner `os.path` in `os.path.sep` aren't
        # in result.types but are still relevant).
        user_node_ids: set[int] = set()
        all_member_exprs: list = []
        seen: set[int] = set()

        def walk(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            if isinstance(n, (Expression, Statement)):
                user_node_ids.add(id(n))
            if isinstance(n, MemberExpr):
                all_member_exprs.append(n)
            # mypyc-compiled nodes don't have __dict__; iterate via
            # dir + getattr.
            for k in dir(n):
                if k.startswith('_'):
                    continue
                try:
                    v = getattr(n, k, None)
                except Exception:
                    continue
                if isinstance(v, (Expression, Statement)):
                    walk(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, (Expression, Statement)):
                            walk(x)

        walk(state.tree)

        types_map = result.types

        entries: dict[tuple[int, int, str], AttrInfo] = {}
        user_classes: set[str] = set()

        # Collect class names defined in this source for fast checks.
        # We also feed them by walking the stdlib AST.
        try:
            user_tree = ast.parse(src)
        except SyntaxError:
            return None
        for n in ast.walk(user_tree):
            if isinstance(n, ast.ClassDef):
                user_classes.add(n.name)

        for expr in all_member_exprs:
            # Receiver might be a module (e.g. `asyncio.run` — the
            # NameExpr `asyncio` resolves to a MypyFile, and
            # types_map has no entry for it; or `os.path.sep` where
            # the receiver `os.path` is itself a MemberExpr that
            # resolves to a MypyFile). Treat any module receiver as
            # stdlib: we don't mangle attributes on imported modules.
            recv = expr.expr
            recv_node = getattr(recv, 'node', None)
            if recv_node is not None and isinstance(recv_node, MypyFile):
                entries[(expr.line, expr.column, expr.name)] = AttrInfo(
                    kind='stdlib',
                )
                continue

            recv_ty = types_map.get(recv)
            if recv_ty is None:
                continue
            kind: str
            cls_name: str | None = None
            if isinstance(recv_ty, Instance):
                if recv_ty.type.module_name == module_name:
                    kind = 'user'
                    cls_name = recv_ty.type.name
                else:
                    kind = 'stdlib'
            elif isinstance(recv_ty, AnyType):
                # No annotation / library without stubs / explicitly
                # `Any` — let the caller treat this conservatively.
                kind = 'any'
            else:
                # Union, Callable, NoneType etc.
                kind = 'unknown'
            entries[(expr.line, expr.column, expr.name)] = AttrInfo(
                kind=kind, user_class=cls_name,
            )

        return TypeMap(entries, user_classes)

    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
