"""Bundle emission: turn the per-module AST rewrites into a final
single-file Python program.

The flow:
  1. discover_modules(root, package) finds every internal .py
  2. for each non-entry module, emit_module_function builds a
     `def _M_pkg_x(_mod): ...` function plus a `_bnd_register(...)` call
  3. emit_entry rewrites the entry module so its imports go through
     `_bnd_load`, keeping the entry's name bindings as bundle globals
     (so `__name__ == "__main__"` is natural for it)
  4. build glues everything together with the runtime preamble
"""
from __future__ import annotations

import ast
import symtable
import sys
from pathlib import Path

from .discover import (
    collect_module_level_names,
    discover_modules,
    mangle,
    resolve_relative,
)
from .rewrite import ModuleRewriter
from .runtime import RUNTIME
from .scope import ScopeInfo


def emit_module_function(modname: str, kind: str, source: str,
                         internal: set[str]) -> tuple[str, ast.FunctionDef]:
    """Build the `def _M_xxx(_mod): ...` function for a module."""
    tree = ast.parse(source, filename=f"<bundle:{modname}>")
    module_names = collect_module_level_names(tree)
    # dunders are pre-set on _mod by _bnd_load; redirect references to them too
    module_names |= {"__name__", "__package__", "__file__", "__doc__",
                     "__loader__", "__spec__", "__path__"}
    # symtable wants the same source
    table = symtable.symtable(source, f"<bundle:{modname}>", "exec")
    scope = ScopeInfo(table, module_names)
    rewriter = ModuleRewriter(modname, kind, source, internal, scope)

    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        out = rewriter.visit(stmt)
        items = out if isinstance(out, list) else [out]
        new_body.extend(items)
        # top-level def/class: mirror name onto _mod and fix __module__
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = stmt.name
            new_body.append(ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id="_mod", ctx=ast.Load()),
                                       attr=nm, ctx=ast.Store())],
                value=ast.Name(id=nm, ctx=ast.Load())))
            new_body.append(ast.Try(
                body=[ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id=nm, ctx=ast.Load()),
                                           attr="__module__", ctx=ast.Store())],
                    value=ast.Constant(value=modname))],
                handlers=[ast.ExceptHandler(type=ast.Name(id="Exception", ctx=ast.Load()),
                                            name=None, body=[ast.Pass()])],
                orelse=[], finalbody=[]))
            new_body.append(ast.Delete(targets=[ast.Name(id=nm, ctx=ast.Del())]))

    # ensure module-level dunders are set on _mod even if user code reads them
    # before assignment (e.g. through a function call). They're already set by
    # _bnd_load before init runs, so nothing to do here.

    fn = ast.FunctionDef(
        name=f"_M_{mangle(modname)}",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="_mod")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=new_body or [ast.Pass()],
        decorator_list=[],
        returns=None,
    )
    ast.fix_missing_locations(fn)
    return ast.unparse(fn), fn


def emit_entry(entry_name: str, source: str, internal: set[str]) -> str:
    """Rewrite the entry module so its imports go through _bnd_load, but keep
    its name bindings as bundle globals (so __name__ == '__main__' is natural)."""
    tree = ast.parse(source, filename=f"<bundle:{entry_name}>")
    out_stmts: list[ast.stmt] = []

    pkg = entry_name.rpartition(".")[0]

    def is_internal(dotted: str) -> bool:
        if dotted in internal:
            return True
        prefix = dotted + "."
        return any(m.startswith(prefix) for m in internal)

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            new_stmts: list[ast.stmt] = []
            for alias in stmt.names:
                if not is_internal(alias.name.split(".")[0]):
                    new_stmts.append(ast.Import(names=[alias]))
                    continue
                parts = alias.name.split(".")
                for i in range(1, len(parts) + 1):
                    new_stmts.append(ast.Expr(
                        ast.Call(func=ast.Name(id="_bnd_load", ctx=ast.Load()),
                                 args=[ast.Constant(value=".".join(parts[:i]))],
                                 keywords=[])))
                bound = alias.asname or parts[0]
                target = alias.name if alias.asname else parts[0]
                new_stmts.append(ast.Assign(
                    targets=[ast.Name(id=bound, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Attribute(value=ast.Name(id="sys", ctx=ast.Load()),
                                            attr="modules", ctx=ast.Load()),
                        slice=ast.Constant(value=target), ctx=ast.Load())))
            out_stmts.extend(ast.copy_location(s, stmt) for s in new_stmts)
        elif isinstance(stmt, ast.ImportFrom):
            target = resolve_relative(pkg, stmt.level, stmt.module)
            if not target or not is_internal(target):
                out_stmts.append(stmt)
                continue
            new_stmts = []
            new_stmts.append(ast.Expr(
                ast.Call(func=ast.Name(id="_bnd_load", ctx=ast.Load()),
                         args=[ast.Constant(value=target)], keywords=[])))
            tmp = "_bnd_entry_src"
            new_stmts.append(ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Attribute(value=ast.Name(id="sys", ctx=ast.Load()),
                                        attr="modules", ctx=ast.Load()),
                    slice=ast.Constant(value=target), ctx=ast.Load())))
            for alias in stmt.names:
                if alias.name == "*":
                    new_stmts.append(ast.Expr(ast.Call(
                        func=ast.Name(id="_bnd_star_globals", ctx=ast.Load()),
                        args=[ast.Name(id=tmp, ctx=ast.Load())],
                        keywords=[])))
                    continue
                bound = alias.asname or alias.name
                sub = f"{target}.{alias.name}"
                if is_internal(sub):
                    new_stmts.append(ast.Expr(
                        ast.Call(func=ast.Name(id="_bnd_load", ctx=ast.Load()),
                                 args=[ast.Constant(value=sub)], keywords=[])))
                new_stmts.append(ast.Assign(
                    targets=[ast.Name(id=bound, ctx=ast.Store())],
                    value=ast.Attribute(value=ast.Name(id=tmp, ctx=ast.Load()),
                                        attr=alias.name, ctx=ast.Load())))
            out_stmts.extend(ast.copy_location(s, stmt) for s in new_stmts)
        else:
            out_stmts.append(stmt)

    new_module = ast.Module(body=out_stmts, type_ignores=[])
    ast.fix_missing_locations(new_module)
    return ast.unparse(new_module)


def _has_dynamic_imports(source: str) -> list[str]:
    warnings = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                full = f"{f.value.id}.{f.attr}"
                if full in ("importlib.import_module", "importlib.reload"):
                    warnings.append(f"line {node.lineno}: {full}() — bundle does not support dynamic loading")
            elif isinstance(f, ast.Name) and f.id == "__import__":
                # __import__ with non-constant arg also iffy
                if node.args and not isinstance(node.args[0], ast.Constant):
                    warnings.append(f"line {node.lineno}: __import__ with dynamic argument")
    return warnings


def build(root: Path, package: str, entry: str, output: Path) -> None:
    """Bundle <root>/<package> into a single Python file at `output`.

    `entry` is the dotted name of the module that runs as `__main__`
    (e.g. `pkg.__main__`).
    """
    mods = discover_modules(root, package)
    if entry not in mods:
        raise SystemExit(f"entry not found: {entry}; known modules:\n  " +
                         "\n  ".join(sorted(mods)))
    internal_mods = set(mods.keys())

    parts: list[str] = []
    parts.append("# generated by onexpr bundle — do not edit")
    parts.append(RUNTIME)
    parts.append("def _bnd_star_globals(src):\n"
                 "    g = globals()\n"
                 "    names = getattr(src, '__all__', None)\n"
                 "    if names is None:\n"
                 "        names = [k for k in vars(src) if not k.startswith('_')]\n"
                 "    for n in names:\n"
                 "        g[n] = getattr(src, n)\n")

    # emit non-entry modules first
    for name in sorted(mods, key=lambda n: (n.count("."), n)):
        if name == entry:
            continue
        kind, path = mods[name]
        src = path.read_text(encoding="utf-8")
        warnings = _has_dynamic_imports(src)
        for w in warnings:
            print(f"warning [{name}] {w}", file=sys.stderr)
        fn_src, _ = emit_module_function(name, kind, src, internal_mods)
        parts.append(fn_src)
        parts.append(f"_bnd_register({name!r}, {kind!r}, _M_{mangle(name)}, {src!r})")

    # entry
    kind, path = mods[entry]
    src = path.read_text(encoding="utf-8")
    warnings = _has_dynamic_imports(src)
    for w in warnings:
        print(f"warning [{entry}] {w}", file=sys.stderr)
    entry_src = emit_entry(entry, src, internal_mods)
    # set entry's __package__ before running, so relative imports in entry work
    entry_pkg = entry.rpartition(".")[0]
    parts.append(f"__package__ = {entry_pkg!r}")
    parts.append(entry_src)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(parts), encoding="utf-8")
