import _ast
import ast

from ..frame import Frame
from .utils import _binding_target


def parse_import(stmt: ast.Import, frame: Frame) -> list[_ast.AST]:
    stmts = []
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            # `import a.b.c` binds the top-level package `a` to the
            # current scope. __import__('a.b.c') already returns `a`.
            top = name.split('.')[0]
            stmts.append(
                ast.Assign(
                    targets=[_binding_target(frame, top)],
                    value=ast.Call(
                        func=ast.Name(id='__import__', ctx=ast.Load()),
                        args=[ast.Constant(value=name)],
                        keywords=[],
                    ),
                )
            )
        else:
            # `import a.b.c as alias` binds the deepest submodule. We
            # must walk the attribute chain because __import__('a.b.c')
            # returns `a`, not `a.b.c`.
            value = ast.Call(
                func=ast.Name(id='__import__', ctx=ast.Load()),
                args=[ast.Constant(value=name)],
                keywords=[],
            )
            for part in name.split('.')[1:]:
                value = ast.Attribute(
                    value=value, attr=part, ctx=ast.Load(),
                )
            stmts.append(
                ast.Assign(
                    targets=[_binding_target(frame, as_name)],
                    value=value,
                )
            )
    return stmts


def parse_import_from(stmt: ast.ImportFrom, frame: Frame) -> list[_ast.AST]:
    stmts = []
    module_var = frame.get_temp_var()
    # The fromlist is what names we want from the module; passing it
    # is what causes __import__ to load submodules (e.g.
    # `from tkinter import ttk` needs fromlist=['ttk'] so ttk is
    # imported as an attribute of tkinter, not just left missing).
    if len(stmt.names) == 1 and stmt.names[0].name == '*':
        from_list = [ast.Constant(value='*')]
    else:
        from_list = [ast.Constant(value=n.name) for n in stmt.names]
    stmts.append(
        ast.Assign(
            targets=[ast.Name(id=module_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='__import__', ctx=ast.Load()),
                args=[
                    ast.Constant(value=stmt.module),
                ],
                keywords=[
                    ast.keyword(
                        arg='fromlist',
                        value=ast.List(elts=from_list, ctx=ast.Load()),
                    )
                ],
            ),
        )
    )
    if len(stmt.names) == 1 and stmt.names[0].name == '*':
        # `from m import *` — pull in everything in m.__all__ if defined,
        # otherwise every name in dir(m) that doesn't start with '_'.
        # We push into globals() since star-imports are only valid at
        # module level anyway.
        key_var = frame.get_temp_var()
        name_var = frame.get_temp_var()
        names_iter = ast.IfExp(
            test=ast.Call(
                func=ast.Name(id='hasattr', ctx=ast.Load()),
                args=[
                    ast.Name(id=module_var, ctx=ast.Load()),
                    ast.Constant(value='__all__'),
                ],
                keywords=[],
            ),
            body=ast.Attribute(
                value=ast.Name(id=module_var, ctx=ast.Load()),
                attr='__all__',
                ctx=ast.Load(),
            ),
            orelse=ast.ListComp(
                elt=ast.Name(id=name_var, ctx=ast.Load()),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id=name_var, ctx=ast.Store()),
                        iter=ast.Call(
                            func=ast.Name(id='dir', ctx=ast.Load()),
                            args=[ast.Name(id=module_var, ctx=ast.Load())],
                            keywords=[],
                        ),
                        ifs=[
                            ast.UnaryOp(
                                op=ast.Not(),
                                operand=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id=name_var, ctx=ast.Load()),
                                        attr='startswith',
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(value='_')],
                                    keywords=[],
                                ),
                            ),
                        ],
                        is_async=0,
                    ),
                ],
            ),
        )
        update_dict = ast.DictComp(
            key=ast.Name(id=key_var, ctx=ast.Load()),
            value=ast.Call(
                func=ast.Name(id='getattr', ctx=ast.Load()),
                args=[
                    ast.Name(id=module_var, ctx=ast.Load()),
                    ast.Name(id=key_var, ctx=ast.Load()),
                ],
                keywords=[],
            ),
            generators=[
                ast.comprehension(
                    target=ast.Name(id=key_var, ctx=ast.Store()),
                    iter=names_iter,
                    ifs=[],
                    is_async=0,
                ),
            ],
        )
        stmts.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='globals', ctx=ast.Load()),
                            args=[], keywords=[],
                        ),
                        attr='update',
                        ctx=ast.Load(),
                    ),
                    args=[update_dict],
                    keywords=[],
                ),
            )
        )
        return stmts
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            as_name = name
        stmts.append(
            ast.Assign(
                targets=[_binding_target(frame, as_name)],
                value=ast.Attribute(
                    value=ast.Name(id=module_var, ctx=ast.Load()),
                    attr=name,
                    ctx=ast.Load(),
                ),
            )
        )
    return stmts
