"""Optional strip pass: remove cosmetic AST elements (docstrings,
type annotations, asserts) before the onexpr transform runs.

Driven by a tag set:
    docs         — module / class / function docstrings.
    annotations  — function param + return annotations, plus
                   module-level and function-body AnnAssign
                   annotations. **Class-body AnnAssigns are kept**
                   automatically, because dataclass / pydantic /
                   attrs / Django ORM all read them at class-creation
                   time. Use a ``# obfuscate: keep`` marker to
                   protect anything else.

`strip_asserts` is a separate boolean flag — assert is more
load-bearing than a docstring or annotation (it has runtime
behaviour) so it shouldn't be folded into a casual `--strip=all`.

Markers
-------
``# obfuscate: keep`` on the line *above* a statement, or on the
same line, protects that statement and only that statement (the
scope is non-recursive: marking a class doesn't shield its method
docstrings; mark each method individually).

The marker is detected by a tokenize pass over the original source,
mapping protected lineno → True. A node at line N is protected if
N or N-1 (i.e. the line a same-line / preceding-line marker would
be on) appears in the protection set.

Pipeline position
-----------------
Run AFTER mangle (mangle's mypy step needs annotations to do type
inference) and BEFORE nonlocal_pass / add_helper. None of the later
passes care about docstrings or annotations, so stripping there is
the cleanest place.
"""
from __future__ import annotations

import ast
import io
import tokenize


VALID_STRIP_TAGS = {'none', 'docs', 'annotations', 'all'}
_ALL_STRIP_TAGS = {'docs', 'annotations'}

_KEEP_MARKER = 'obfuscate: keep'


def expand_strip_tags(spec: str) -> set[str]:
    """Parse a CSV of strip tags. ``'none'`` / empty → empty set.
    ``'all'`` expands to docs + annotations."""
    if not spec:
        return set()
    out: set[str] = set()
    for raw in spec.split(','):
        t = raw.strip()
        if not t or t == 'none':
            continue
        if t == 'all':
            out |= _ALL_STRIP_TAGS
            continue
        if t not in VALID_STRIP_TAGS:
            raise SystemExit(
                f'unknown --strip tag: {t!r} '
                f'(valid: {sorted(VALID_STRIP_TAGS)})'
            )
        out.add(t)
    return out


def _collect_keep_lines(src: str) -> set[int]:
    """Tokenize `src` and return the set of line numbers carrying a
    ``# obfuscate: keep`` comment.

    Both same-line (``def foo():  # obfuscate: keep``) and
    preceding-line forms are recognised by the caller — this just
    returns the raw line set; the caller decides which neighbour
    relations count.
    """
    lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            # Comment text is the full `# ...` string.
            body = tok.string.lstrip('#').strip()
            if body == _KEEP_MARKER or body.startswith(_KEEP_MARKER + ' '):
                lines.add(tok.start[0])
    except tokenize.TokenizeError:
        # Malformed source — return an empty set; caller continues
        # without protection.
        pass
    return lines


def _is_protected(node: ast.AST, keep_lines: set[int]) -> bool:
    """Same-line marker (``def foo():  # obfuscate: keep``) lands on
    the same lineno as the def. Preceding-line marker is on N-1
    where the def is on N. We accept either."""
    line = getattr(node, 'lineno', None)
    if line is None:
        return False
    return line in keep_lines or (line - 1) in keep_lines


def _is_string_constant(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _strip_docstring(body: list, keep_lines: set[int]) -> list:
    """Remove the leading docstring (Expr containing a string
    constant) from `body` if it's not protected by a marker."""
    if not body:
        return body
    first = body[0]
    if not _is_string_constant(first):
        return body
    if _is_protected(first, keep_lines):
        return body
    return body[1:]


def _strip_args_annotations(args: ast.arguments) -> None:
    for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        a.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None


def _strip_function_annotations(node, keep_lines: set[int]) -> None:
    """Drop param + return annotations from a FunctionDef /
    AsyncFunctionDef / Lambda. No-op if the function is protected."""
    if _is_protected(node, keep_lines):
        return
    _strip_args_annotations(node.args)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.returns = None


def _strip_annassign(stmt: ast.AnnAssign, keep_lines: set[int],
                     in_class_body: bool) -> ast.stmt | None:
    """Decide what to do with an AnnAssign at the given context.

    Returns the (possibly replacement) stmt, or None if the stmt
    should be removed entirely (only happens when the AnnAssign has
    no value AND we're stripping annotations AND it's not protected
    AND it's NOT in a class body).
    """
    if in_class_body:
        # Class-body AnnAssigns — dataclass / pydantic / attrs /
        # Django ORM read these. Always preserve.
        return stmt
    if _is_protected(stmt, keep_lines):
        return stmt
    if stmt.value is None:
        # `x: int` with no value — pure declaration, can drop.
        return None
    # `x: int = v` — keep the assignment, drop the annotation.
    return ast.copy_location(
        ast.Assign(targets=[stmt.target], value=stmt.value), stmt,
    )


def _walk_stmts(body: list, keep_lines: set[int], tags: set[str],
                strip_asserts: bool, in_class_body: bool = False) -> list:
    """Process a list of stmts in-place, returning the new list."""
    out: list = []
    for stmt in body:
        protected = _is_protected(stmt, keep_lines)
        if strip_asserts and isinstance(stmt, ast.Assert) and not protected:
            continue
        if isinstance(stmt, ast.AnnAssign) and 'annotations' in tags:
            replaced = _strip_annassign(stmt, keep_lines, in_class_body)
            if replaced is None:
                continue
            stmt = replaced
        # Recurse into bodies for nested defs / class / control flow
        _recurse_into_bodies(stmt, keep_lines, tags, strip_asserts)
        out.append(stmt)
    return out


def _recurse_into_bodies(stmt: ast.stmt, keep_lines: set[int],
                         tags: set[str], strip_asserts: bool) -> None:
    """Strip docstrings / annotations / asserts inside `stmt`'s
    nested bodies."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if 'annotations' in tags:
            _strip_function_annotations(stmt, keep_lines)
        # Walk body. Function-body AnnAssigns get stripped via the
        # outer _walk_stmts loop on stmt.body.
        if 'docs' in tags and not _is_protected(stmt, keep_lines):
            stmt.body = _strip_docstring(stmt.body, keep_lines)
        stmt.body = _walk_stmts(
            stmt.body, keep_lines, tags, strip_asserts,
            in_class_body=False,
        )
    elif isinstance(stmt, ast.ClassDef):
        if 'docs' in tags and not _is_protected(stmt, keep_lines):
            stmt.body = _strip_docstring(stmt.body, keep_lines)
        stmt.body = _walk_stmts(
            stmt.body, keep_lines, tags, strip_asserts,
            in_class_body=True,
        )
    elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While,
                           ast.With, ast.AsyncWith, ast.Try, ast.TryStar)):
        # Recurse into all nested-stmt fields.
        if hasattr(stmt, 'body'):
            stmt.body = _walk_stmts(
                stmt.body, keep_lines, tags, strip_asserts,
                in_class_body=False,
            )
        if hasattr(stmt, 'orelse'):
            stmt.orelse = _walk_stmts(
                stmt.orelse, keep_lines, tags, strip_asserts,
                in_class_body=False,
            )
        if hasattr(stmt, 'finalbody'):
            stmt.finalbody = _walk_stmts(
                stmt.finalbody, keep_lines, tags, strip_asserts,
                in_class_body=False,
            )
        if hasattr(stmt, 'handlers'):
            for h in stmt.handlers:
                h.body = _walk_stmts(
                    h.body, keep_lines, tags, strip_asserts,
                    in_class_body=False,
                )
    elif isinstance(stmt, ast.Match):
        for case in stmt.cases:
            case.body = _walk_stmts(
                case.body, keep_lines, tags, strip_asserts,
                in_class_body=False,
            )
    # Lambdas (in expressions) — handled by ast.walk below.


def apply_strip(tree: ast.Module, src: str, tags: set[str],
                strip_asserts: bool = False) -> ast.Module:
    """Run the configured strip pass over `tree`. `src` is the
    original source text — used to find ``# obfuscate: keep``
    markers (we can't recover them from the AST).
    """
    if not tags and not strip_asserts:
        return tree

    keep_lines = _collect_keep_lines(src)

    # Module-level docstring.
    if 'docs' in tags:
        # The protection check for the module-level docstring uses
        # the docstring's own lineno (or line - 1).
        tree.body = _strip_docstring(tree.body, keep_lines)

    # Walk top-level stmts.
    tree.body = _walk_stmts(
        tree.body, keep_lines, tags, strip_asserts, in_class_body=False,
    )

    # Lambdas may carry annotations on their parameters (rare; the
    # syntax only allows them in the same arg form). Walk every
    # lambda anywhere in the tree and clear the annotation field.
    if 'annotations' in tags:
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                _strip_function_annotations(node, keep_lines)
    return tree
