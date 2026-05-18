import _ast
import ast
from typing import Callable

from ..frame import Frame


def parse_stmt(stmt: ast.stmt, frame: Frame) -> list[ast.expr]:
    exprs = [stmt]

    while not all(isinstance(expr, ast.expr) for expr in exprs):
        new_exprs = []

        for expr in exprs:
            if isinstance(expr, ast.expr):
                new_exprs.append(expr)
                continue
            class_name = expr.__class__.__name__
            nf = name2func()
            assert class_name in nf, f"unknown stmt: {class_name}"

            parse_func = nf[class_name]
            exprs = parse_func(expr, frame)
            if exprs is None:
                raise NotImplementedError(f"Not implemented for {class_name}")

            new_exprs.extend(exprs)

        exprs = new_exprs

    return exprs


def parse_stmts(stmts: list[ast.stmt], frame: Frame) -> _ast.BoolOp:
    exprs = []
    for stmt in stmts:
        expr = parse_stmt(stmt, frame)
        exprs.extend(expr)

    return ast.BoolOp(
        op=ast.Or(),
        values=exprs
    )


def _build_name2func() -> dict[str, Callable]:
    from .func_def import parse_function_def, parse_async_function_def
    from .class_def import parse_class_def
    from .pep695 import parse_type_alias
    from .control_flow import (parse_for, parse_async_for, parse_while,
                                parse_if, parse_with, parse_async_with, parse_match)
    from .exceptions import parse_raise, parse_try, parse_try_star, parse_assert
    from .imports import parse_import, parse_import_from
    from .simple import (parse_return, parse_delete, parse_assign, parse_aug_assign,
                         parse_ann_assign, parse_expr, parse_pass, parse_break,
                         parse_continue, parse_global, parse_nonlocal)

    return {
        "FunctionDef": parse_function_def,
        "AsyncFunctionDef": parse_async_function_def,
        "ClassDef": parse_class_def,
        "Return": parse_return,
        "Delete": parse_delete,
        "Assign": parse_assign,
        "TypeAlias": parse_type_alias,
        "AugAssign": parse_aug_assign,
        "AnnAssign": parse_ann_assign,
        "For": parse_for,
        "AsyncFor": parse_async_for,
        "While": parse_while,
        "If": parse_if,
        "With": parse_with,
        "AsyncWith": parse_async_with,
        "Match": parse_match,
        "Raise": parse_raise,
        "Try": parse_try,
        "TryStar": parse_try_star,
        "Assert": parse_assert,
        "Import": parse_import,
        "ImportFrom": parse_import_from,
        "Global": parse_global,
        "Nonlocal": parse_nonlocal,
        "Expr": parse_expr,
        "Pass": parse_pass,
        "Break": parse_break,
        "Continue": parse_continue,
    }


_name2func: dict[str, Callable] | None = None


def _get_name2func() -> dict[str, Callable]:
    global _name2func
    if _name2func is None:
        _name2func = _build_name2func()
    return _name2func


name2func = _get_name2func
