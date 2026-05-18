"""Symbol-table scope tracking for the bundler.

Used by ModuleRewriter to decide which Name nodes refer to the enclosing
module's top-level namespace (and therefore need to be redirected to
`_mod.<name>`) vs. function locals / closures / builtins (which stay
untouched).
"""
from __future__ import annotations

import ast
import symtable


class ScopeInfo:
    """Per-scope info derived from symtable: which names are *resolved* from the
    enclosing module scope (i.e., would do LOAD_GLOBAL / STORE_GLOBAL at runtime).
    """

    def __init__(self, table: symtable.SymbolTable, module_names: set[str]):
        self.table = table
        self.module_names = module_names
        # children indexed by (type, name, lineno) — symtable's identifier
        self.children: dict[tuple[str, str, int], ScopeInfo] = {}
        for c in table.get_children():
            self.children[(c.get_type(), c.get_name(), c.get_lineno())] = ScopeInfo(c, module_names)

    def child_for(self, node: ast.AST) -> ScopeInfo | None:
        """Find the symtable child for an ast node that opens a scope."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = ("function", node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            key = ("class", node.name, node.lineno)
        elif isinstance(node, ast.Lambda):
            key = ("function", "lambda", node.lineno)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            key = ("function", {ast.ListComp: "listcomp", ast.SetComp: "setcomp",
                                ast.DictComp: "dictcomp"}[type(node)], node.lineno)
        elif isinstance(node, ast.GeneratorExp):
            key = ("function", "genexpr", node.lineno)
        else:
            return None
        return self.children.get(key)

    def resolves_to_module(self, name: str) -> bool:
        """Is `name`, when used in this scope, a name we want to redirect to _mod?

        Only names that are *actually bound* at module top level get redirected.
        Built-ins / unresolved free names stay as plain Name (so `print`, `len`,
        etc. work via normal LOAD_GLOBAL → builtins fallback).
        """
        if name not in self.module_names:
            return False
        try:
            sym = self.table.lookup(name)
        except KeyError:
            # name doesn't appear in this scope at all
            return False
        if self.table.get_type() == "module":
            return True
        if sym.is_local() or sym.is_parameter():
            return False
        # nested scope referencing the module-bound name (free or declared global)
        return True
