import _ast
import ast

from ..frame import Frame
from .utils import add_deco


def _make_typevar_call(tp: ast.AST) -> ast.expr:
    """Build a typing.TypeVar / ParamSpec / TypeVarTuple constructor
    call for a PEP 695 type-param node."""
    typing_mod = ast.Call(
        func=ast.Name(id='__import__', ctx=ast.Load()),
        args=[ast.Constant(value='typing')],
        keywords=[],
    )
    if isinstance(tp, ast.TypeVar):
        kind = 'TypeVar'
        kwargs = []
        # `T: bound` — single bound expression. `T: (a, b)` — constraints.
        if tp.bound is not None:
            if isinstance(tp.bound, ast.Tuple):
                # constraints: pass as positional args after the name
                args = [ast.Constant(value=tp.name)] + list(tp.bound.elts)
            else:
                args = [ast.Constant(value=tp.name)]
                kwargs.append(ast.keyword(arg='bound', value=tp.bound))
        else:
            args = [ast.Constant(value=tp.name)]
        # PEP 695 type vars are implicitly variance-inferring (which
        # also drops the leading ~/+/- in repr); typing.TypeVar's
        # default is invariant.
        kwargs.append(ast.keyword(arg='infer_variance', value=ast.Constant(value=True)))
        # PEP 696 default
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    elif isinstance(tp, ast.ParamSpec):
        kind = 'ParamSpec'
        args = [ast.Constant(value=tp.name)]
        kwargs = []
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    elif isinstance(tp, ast.TypeVarTuple):
        kind = 'TypeVarTuple'
        args = [ast.Constant(value=tp.name)]
        kwargs = []
        if getattr(tp, 'default_value', None) is not None:
            kwargs.append(ast.keyword(arg='default', value=tp.default_value))
    else:
        raise NotImplementedError(f'PEP 695 type param: {type(tp).__name__}')
    return ast.Call(
        func=ast.Attribute(value=typing_mod, attr=kind, ctx=ast.Load()),
        args=args,
        keywords=kwargs,
    )


def _make_pep695_wrapper(
    name: str,
    type_params: list,
    inner_stmts: list,
    return_expr: ast.expr,
    frame: Frame,
) -> list[_ast.AST]:
    """Emit the standard PEP 695 wrapper:

        def <wrapper>():
            T = typing.TypeVar('T')   # for each type param
            ...
            <inner_stmts>             # caller-provided body
            return <return_expr>
        <name> = <wrapper>()

    The wrapper isolates the type vars in its own scope so they don't
    leak into the surrounding namespace, and gives the alias / def /
    class body's RHS access to them via closure."""
    wrapper_name = frame.get_temp_var()
    body = []
    for tp in type_params:
        body.append(
            ast.Assign(
                targets=[ast.Name(id=tp.name, ctx=ast.Store())],
                value=_make_typevar_call(tp),
            )
        )
    body.extend(inner_stmts)
    body.append(ast.Return(value=return_expr))
    return [
        ast.FunctionDef(
            name=wrapper_name,
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], defaults=[],
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_params=[],
        ),
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=wrapper_name, ctx=ast.Load()),
                args=[], keywords=[],
            ),
        ),
    ]


def _wrap_pep695_def(stmt, frame: Frame, is_class: bool) -> list[_ast.AST]:
    """Rewrite a generic def/class declaration (PEP 695) into a wrapper
    function call. The wrapper isolates the type variables, assigns
    __type_params__ on the inner construct, applies decorators, and
    returns the decorated callable/class. The outer assignment binds
    the user's chosen name."""
    type_params = stmt.type_params
    decorators = list(stmt.decorator_list)
    name = stmt.name
    inner_temp = frame.get_temp_var()
    # Inner declaration: same node minus type_params and decorators,
    # bound to a temp name. The decorators get re-applied below after
    # __type_params__ is attached.
    inner_stmt = ast.copy_location(
        type(stmt)(
            **{f: getattr(stmt, f) for f in stmt._fields if f not in ('decorator_list', 'type_params', 'name')},
            name=inner_temp,
            decorator_list=[],
            type_params=[],
        ),
        stmt,
    )
    # Attach __type_params__ tuple
    type_params_tuple = ast.Tuple(
        elts=[ast.Name(id=tp.name, ctx=ast.Load()) for tp in type_params],
        ctx=ast.Load(),
    )
    inner_setup = [
        inner_stmt,
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__type_params__',
                    ctx=ast.Store(),
                )
            ],
            value=type_params_tuple,
        ),
        # Restore the user-visible name on the inner construct (we
        # built it with a temp name to avoid shadowing the outer
        # binding).
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__name__',
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=name),
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=inner_temp, ctx=ast.Load()),
                    attr='__qualname__',
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=name),
        ),
    ]
    if is_class:
        # PEP 695 generic classes get an auto-generated __class_getitem__
        # so `Cls[int]` works. CPython does this in the compiler; we
        # have to emit it explicitly. types.GenericAlias matches what
        # the real PEP 695 path produces (modulo the exact class name
        # of the alias).
        inner_setup.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=inner_temp, ctx=ast.Load()),
                        attr='__class_getitem__',
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='classmethod', ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[
                                    ast.arg(arg='cls', annotation=None),
                                    ast.arg(arg='__p', annotation=None),
                                ],
                                vararg=None,
                                kwonlyargs=[], kw_defaults=[], defaults=[],
                            ),
                            body=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name(id='__import__', ctx=ast.Load()),
                                        args=[ast.Constant(value='types')],
                                        keywords=[],
                                    ),
                                    attr='GenericAlias',
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Name(id='cls', ctx=ast.Load()),
                                    ast.Name(id='__p', ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                        ),
                    ],
                    keywords=[],
                ),
            )
        )
    # Build the decorated return expression: deco_n(...deco_1(inner)...)
    return_expr = add_deco(
        name, decorators, ast.Name(id=inner_temp, ctx=ast.Load())
    )
    return _make_pep695_wrapper(
        name=name,
        type_params=type_params,
        inner_stmts=inner_setup,
        return_expr=return_expr,
        frame=frame,
    )


def parse_type_alias(stmt: ast.TypeAlias, frame: Frame) -> list[_ast.AST]:
    # PEP 695 `type X[T, ...] = expr`. We wrap the construction in a
    # zero-arg function so the type vars stay scoped, then build a
    # _LazyAlias whose `__value__` is computed lazily from a thunk
    # (so RHS forward-references / side-effects match Python's real
    # PEP 695 semantics).
    #
    #     def _wrap():
    #         T = typing.TypeVar('T')
    #         return _LazyAlias('X', lambda: <expr>, type_params=(T,))
    #     X = _wrap()
    type_param_names = [tp.name for tp in stmt.type_params]
    thunk = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=stmt.value,
    )
    type_params_tuple = ast.Tuple(
        elts=[ast.Name(id=n, ctx=ast.Load()) for n in type_param_names],
        ctx=ast.Load(),
    )
    alias_construct = ast.Call(
        func=ast.Name(id='_LazyAlias', ctx=ast.Load()),
        args=[
            ast.Constant(value=stmt.name.id),
            thunk,
        ],
        keywords=[ast.keyword(arg='type_params', value=type_params_tuple)],
    )
    return _make_pep695_wrapper(
        name=stmt.name.id,
        type_params=stmt.type_params,
        inner_stmts=[],
        return_expr=alias_construct,
        frame=frame,
    )
