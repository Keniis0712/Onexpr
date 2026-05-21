"""Module discovery and name resolution for the bundler."""
from __future__ import annotations

import ast
import os
from pathlib import Path


def discover_modules(root: Path, package: str) -> dict[str, tuple[str, Path]]:
    """Walk <root>/<package> and return {dotted_name: (kind, path)}.

    kind is "package" (for __init__.py) or "module".
    """
    mods: dict[str, tuple[str, Path]] = {}
    pkg_root = root / package
    if not pkg_root.is_dir():
        raise SystemExit(f"package root not found: {pkg_root}")
    for dirpath, dirnames, filenames in os.walk(pkg_root):
        rel = Path(dirpath).relative_to(root)
        dotted_dir = ".".join(rel.parts)
        # ignore __pycache__ etc
        dirnames[:] = [d for d in dirnames if not d.startswith("__pycache__")]
        if "__init__.py" in filenames:
            mods[dotted_dir] = ("package", Path(dirpath) / "__init__.py")
        for fn in filenames:
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            stem = fn[:-3]
            mods[f"{dotted_dir}.{stem}"] = ("module", Path(dirpath) / fn)
    return mods


def mangle(dotted: str) -> str:
    return dotted.replace(".", "_")


def resolve_relative(current_pkg: str, level: int, module: str | None) -> str:
    """`from .x import y` inside pkg.sub -> resolve_relative('pkg.sub', 1, 'x')."""
    if level == 0:
        return module or ""
    parts = current_pkg.split(".")
    if level > len(parts):
        raise SyntaxError(f"relative import beyond top-level: level={level} pkg={current_pkg}")
    base = parts[: len(parts) - level + 1]
    if module:
        base.append(module)
    return ".".join(base)


def parent_pkg(modname: str, kind: str) -> str:
    if kind == "package":
        return modname
    return modname.rpartition(".")[0]


def collect_module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module top level. Used for symtable cross-check / fallback."""
    names: set[str] = set()

    def add_target(t: ast.expr) -> None:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                add_target(elt)
        elif isinstance(t, ast.Starred):
            add_target(t.value)
        # Attribute / Subscript targets don't bind a top-level name

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                add_target(tgt)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            add_target(node.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name.split(".")[0]
                names.add(local)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    add_target(item.optional_vars)
        elif isinstance(node, ast.Try):
            for h in node.handlers:
                if h.name:
                    names.add(h.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.TypeAlias):
            # PEP 695: `type X[T] = ...`
            if isinstance(node.name, ast.Name):
                names.add(node.name.id)
        elif isinstance(node, ast.Expr):
            # Top-level expression statements may contain walrus
            # bindings: `(x := 1)` at module top level binds x.
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    names.add(sub.target.id)
    return names
