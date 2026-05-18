import _ast
import ast

from ..frame import Frame
from .utils import slice_to_callable


def parse_return(stmt: ast.Return, frame: Frame) -> list[_ast.AST]:
    value = stmt.value if stmt.value is not None else ast.Constant(value=None)
    if frame.legacy_return:
        # Old style: produce a (value, True) tuple so the body's Or
        # chain short-circuits truthy. gen_func's `[0]` extraction
        # picks `value` out at the lambda boundary.
        return [
            ast.Tuple(
                elts=[value, ast.Constant(value=True)],
                ctx=ast.Load(),
            )
        ]
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.func_helper_var, ctx=ast.Load()),
                attr=frame.get_helper_member('do_return'),
                ctx=ast.Load(),
            ),
            args=[value],
            keywords=[],
        )
    ]


def parse_delete(stmt: ast.Delete, frame: Frame) -> list[_ast.AST]:
    # Flatten Tuple / List targets: `del a, (b, c)` deletes a, b, c.
    # `del ()` is a no-op (empty tuple has no targets).
    flat = []
    def flatten(t):
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                flatten(e)
        else:
            flat.append(t)
    for t in stmt.targets:
        flatten(t)

    out = []
    for target in flat:
        if isinstance(target, ast.Attribute):
            # del obj.x -> delattr(obj, 'x')
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='delattr', ctx=ast.Load()),
                    args=[
                        target.value,
                        ast.Constant(value=target.attr),
                    ],
                    keywords=[],
                )
            ))
        elif isinstance(target, ast.Subscript):
            # del obj[k] -> obj.__delitem__(k); slice key needs to be
            # built explicitly because `1:3` syntax isn't valid as a
            # call argument.
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=target.value,
                        attr='__delitem__',
                        ctx=ast.Load(),
                    ),
                    args=[slice_to_callable(target.slice)],
                    keywords=[],
                )
            ))
        elif isinstance(target, ast.Name):
            # del x: best-effort. _del_local handles both the module-
            # level case (deletes from globals) and the function/lambda
            # local case (sets the slot to None on Python 3.13+ via the
            # PEP 667 f_locals proxy; on earlier versions the local
            # case is a silent no-op). Either way, the original
            # NameError-on-subsequent-access semantics can't be
            # reproduced — once a slot exists in fast locals there's
            # no Python-level way to unbind it.
            out.append(ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=frame.get_helper_name('_del_local'), ctx=ast.Load()),
                    args=[ast.Constant(value=target.id)],
                    keywords=[],
                )
            ))
        else:
            raise NotImplementedError(f"del target: {type(target).__name__}")
    return out


def parse_assign(stmt: ast.Assign, frame: Frame) -> list[_ast.AST]:
    if len(stmt.targets) != 1:
        # This is look like "a = b = 0"

        # First copy the value to a temp var
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[
                    ast.Name(id=temp_var, ctx=ast.Store())
                ],
                value=stmt.value,
            )
        ]

        # Then copy the temp var to each target
        assigns.extend(
            ast.Assign(
                targets=[target],
                value=ast.Name(id=temp_var, ctx=ast.Load())
            )
            for target in stmt.targets
        )

        return assigns

    target = stmt.targets[0]

    if isinstance(target, (ast.List, ast.Tuple)):
        # This is look like "a, b = c"
        # First materialize the RHS as a list so subscripting works for any
        # iterable (including iterators and instances with __iter__ but no
        # __getitem__).
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='list', ctx=ast.Load()),
                    args=[stmt.value],
                    keywords=[],
                ),
            )
        ]
        # Then get each element from the temp var

        # If there is a starred name, we have to get how many elements before and after it
        names = target.elts
        before_star = None
        starred_name = None

        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                before_star = pos
                # Starred target can be a plain Name or — after the
                # nonlocal pre-pass rewrites a boxed name — an
                # Attribute (helper._b_<n>). Both are handled below
                # via the same Subscript-and-store machinery.
                starred_target = name.value
                break

        if before_star is not None:
            after_star = len(names) - before_star - 1
            upper = ast.Constant(-after_star) if after_star > 0 else ast.Constant(value=None)
            assigns.append(
                ast.Assign(
                    targets=[starred_target],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Slice(
                            lower=ast.Constant(before_star),
                            upper=upper,
                            col_offset=0,
                            end_col_offset=None,
                            end_lineno=None,
                            lineno=0,
                        )
                    )
                )
            )

        # Finally we start to build the assigns
        parse_before_star = True
        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                # We already parsed it
                parse_before_star = False
                continue
            assigns.append(
                ast.Assign(
                    targets=[name],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Constant(pos if parse_before_star else pos - len(names)),
                    )
                )
            )
        return assigns

    if isinstance(target, (ast.Subscript, ast.Attribute)):
        if isinstance(target, ast.Subscript):
            return [
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=target.value,
                            attr='__setitem__',
                            ctx=ast.Load(),
                        ),
                        args=[
                            slice_to_callable(target.slice),
                            stmt.value,
                        ],
                        keywords=[],
                    )
                )
            ]
        # Attribute assignment: use builtin setattr() so it works on both
        # instances and class objects (cls.__setattr__('a', v) fails because
        # __setattr__ on a class is a metaclass method that needs 3 args).
        return [
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='setattr', ctx=ast.Load()),
                    args=[
                        target.value,
                        ast.Constant(target.attr),
                        stmt.value,
                    ],
                    keywords=[],
                )
            )
        ]

    # Now is a simple assign like "a = 1"
    assert isinstance(target, ast.Name), f"unexpected assign target: {ast.dump(target)[:200]}"
    if target.id in frame.global_vars:
        # `global x` was declared in this scope; route writes through
        # globals() so they land in the module dict instead of the
        # generated lambda's locals.
        return [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='globals', ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        attr='__setitem__',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=target.id),
                        stmt.value,
                    ],
                    keywords=[],
                )
            )
        ]
    return [
        ast.Expr(
            value=ast.NamedExpr(
                target=target,
                value=stmt.value,
            )
        )
    ]


def parse_aug_assign(stmt: ast.AugAssign, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Assign(
            targets=[stmt.target],
            value=ast.BinOp(
                left=stmt.target,
                op=stmt.op,
                right=stmt.value
            ),
        )
    ]


def parse_ann_assign(stmt: ast.AnnAssign, frame: Frame) -> list[_ast.AST]:
    # `x: int [= v]` at module level or in a class body writes to
    # __annotations__. Inside a function body, the annotation is
    # ignored at runtime (it's only for static checkers).
    should_annotate = (
        frame.prev is None  # module level
        or frame.is_class_body
    ) and isinstance(stmt.target, ast.Name)

    out = []
    if should_annotate:
        # __annotations__['x'] = int
        out.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='__annotations__', ctx=ast.Load()),
                        attr='__setitem__',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=stmt.target.id),
                        stmt.annotation,
                    ],
                    keywords=[],
                )
            )
        )
    if stmt.value is None:
        # Bare annotation `x: int` with no value. If we wrote to
        # __annotations__, that's all. Otherwise it's a no-op.
        if not out:
            out.append(ast.Constant(value=False))
        return out
    # `x: int = v` -> also do `x = v`.
    out.append(ast.Assign(targets=[stmt.target], value=stmt.value))
    return out


def parse_expr(stmt: ast.Expr, frame: Frame) -> list[_ast.AST]:
    if frame.in_async_def and isinstance(stmt.value, ast.Await):
        value = ast.YieldFrom(value=stmt.value.value)
    else:
        value = stmt.value
    # Use `(value, False)[1]` instead of `value and False` so we don't
    # call __bool__ on the value. Numpy arrays (and a few other types)
    # raise on bool() of multi-element instances; the tuple-and-index
    # form sidesteps that while still evaluating value for its side
    # effects and producing a falsy result for the surrounding Or
    # chain.
    return [
        ast.Subscript(
            value=ast.Tuple(
                elts=[value, ast.Constant(value=False)],
                ctx=ast.Load(),
            ),
            slice=ast.Constant(value=1),
            ctx=ast.Load(),
        )
    ]


def parse_pass(_: ast.Pass, __: Frame) -> list[_ast.AST]:
    return [
        ast.Constant(value=False),
    ]


def parse_break(_: ast.Break, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr=frame.get_helper_member('stop'),
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    ]


def parse_continue(stmt: ast.Continue, frame: Frame) -> list[_ast.AST]:
    # Sets the loop helper's pending_continue flag and returns True.
    # The truthy return value still short-circuits the surrounding Or
    # chain (so `if cond: continue\nrest` skips `rest` for that
    # iteration). The flag is what lets us cross lambda boundaries: a
    # `continue` inside a try body lambda becomes invisible to the
    # outer for-body Or chain unless something explicitly checks
    # loop.pending_continue, which parse_try does.
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr=frame.get_helper_member('do_continue'),
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
    ]


def parse_global(stmt: ast.Global, frame: Frame) -> list[_ast.AST]:
    for name in stmt.names:
        if name not in frame.global_vars:
            frame.global_vars.append(name)
    return [ast.Constant(value=False)]


def parse_nonlocal(stmt: ast.Nonlocal, frame: Frame) -> list[_ast.AST]:
    # The nonlocal pre-pass already rewrote every read/write of these
    # names to go through the owner function's box, so the declaration
    # itself becomes a no-op at this point.
    return [ast.Constant(value=False)]
