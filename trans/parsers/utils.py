import _ast
import ast

from ..frame import Frame


def add_deco(name: str, decorators: list[ast.expr], func: _ast.expr) -> ast.expr:
    node = func
    for decorator in reversed(decorators):
        node = ast.Call(
            func=decorator,
            args=[node],
            keywords=[],
        )
    return node


def slice_to_callable(key: ast.expr) -> ast.expr:
    """If key is an ast.Slice (or a tuple containing slices, e.g.
    `arr[::2, ::2]`), convert each slice to a slice(...) call so the
    expression is valid outside subscript brackets. Plain tuple
    elements are left alone — they're already legal in a tuple
    literal."""
    if isinstance(key, ast.Slice):
        return ast.Call(
            func=ast.Name(id='slice', ctx=ast.Load()),
            args=[
                key.lower if key.lower is not None else ast.Constant(value=None),
                key.upper if key.upper is not None else ast.Constant(value=None),
                key.step if key.step is not None else ast.Constant(value=None),
            ],
            keywords=[],
        )
    if isinstance(key, ast.Tuple):
        # Multi-axis subscript: rewrite each slice element. Non-slice
        # elements stay as-is so e.g. `arr[i, ::2]` becomes
        # `(i, slice(None, None, 2))`.
        return ast.Tuple(
            elts=[slice_to_callable(e) for e in key.elts],
            ctx=key.ctx,
        )
    return key


def strip_arg_annotations(args: ast.arguments) -> dict:
    """Lambda doesn't accept annotated parameters; strip them in
    place. Returns a {param_name: annotation_expr} dict so the caller
    can re-attach them to the resulting function's __annotations__.
    The dict preserves Python's parameter order: posonly, args, vararg,
    kwonly, kwarg."""
    annotations = {}
    for group in (args.posonlyargs, args.args):
        for a in group:
            if a.annotation is not None:
                annotations[a.arg] = _normalize_annotation(a.annotation)
                a.annotation = None
    if args.vararg is not None:
        if args.vararg.annotation is not None:
            annotations[args.vararg.arg] = _normalize_annotation(args.vararg.annotation)
            args.vararg.annotation = None
    for a in args.kwonlyargs:
        if a.annotation is not None:
            annotations[a.arg] = _normalize_annotation(a.annotation)
            a.annotation = None
    if args.kwarg is not None:
        if args.kwarg.annotation is not None:
            annotations[args.kwarg.arg] = _normalize_annotation(args.kwarg.annotation)
            args.kwarg.annotation = None
    return annotations


def _normalize_annotation(ann: ast.expr) -> ast.expr:
    """Rewrite a top-level `*X` annotation (PEP 646: `def f(*args: *Ts)`)
    into `typing.Unpack[X]`. The bare `Starred` node is illegal in a
    dict-value position, which is where the annotation ends up when we
    rebuild `__annotations__`. CPython itself stores the annotation as
    `typing.Unpack[Ts]` at runtime, so this is a faithful match."""
    if isinstance(ann, ast.Starred):
        return ast.Subscript(
            value=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='__import__', ctx=ast.Load()),
                    args=[ast.Constant(value='typing')],
                    keywords=[],
                ),
                attr='Unpack', ctx=ast.Load(),
            ),
            slice=ann.value,
            ctx=ast.Load(),
        )
    return ann


def _binding_target(frame: Frame, name: str) -> ast.expr:
    """Build the appropriate Store target for `name` in this frame.

    If the nonlocal pre-pass marked `name` as boxed in the owning
    function (try-clause assigns, nonlocal-targeted), produce an
    Attribute on the helper var (`<helper>._b_<name>`). Otherwise fall
    back to a plain Name. parse_assign and friends turn an
    Attribute-Store into setattr at the lambda level, so this is
    the same form _rewrite would have emitted for an existing Name
    in the original AST.
    """
    boxed = frame.boxed_names
    if boxed is not None and name in boxed and frame.func_helper_var is not None:
        return ast.Attribute(
            value=ast.Name(id=frame.func_helper_var, ctx=ast.Load()),
            attr='_b_' + name, ctx=ast.Store(),
        )
    return ast.Name(id=name, ctx=ast.Store())
