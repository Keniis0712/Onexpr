import ast
import sys

from .frame import Frame
from .helpers import add_helper
from .mangle import apply_mangle, expand_tags, _NamePool
from .nonlocals import apply_nonlocal_pass
from .parsers import parse_stmts
from .passes import SuperTransformer, collect_user_names
from .strip import apply_strip, expand_strip_tags


def parse_root(tree: ast.Module, replace_name: str = 'none',
               src: str | None = None,
               strip: str = 'none',
               strip_asserts: bool = False) -> ast.Module:
    # The transform recurses through the AST several times — for big
    # source files (e.g. CPython's own test_grammar.py with very
    # nested data literals) the default recursion limit can be too
    # low. The transformed AST is deeper still (every stmt becomes a
    # nested BoolOp / Subscript / Lambda chain), so callers that
    # want to ast.unparse() the result need the bumped limit too.
    # Leave the limit raised — restoring it here would break unparse.
    if sys.getrecursionlimit() < 5000:
        sys.setrecursionlimit(5000)
    return _parse_root_inner(
        tree, replace_name=replace_name, src=src,
        strip=strip, strip_asserts=strip_asserts,
    )


def _parse_root_inner(tree: ast.Module, replace_name: str = 'none',
                      src: str | None = None,
                      strip: str = 'none',
                      strip_asserts: bool = False) -> ast.Module:
    tags = expand_tags(replace_name)
    strip_tags = expand_strip_tags(strip)

    # Init the top frame
    top_frame = Frame(None, [], [])
    top_frame.temp_var_num = 0
    top_frame.reserved_names = collect_user_names(tree)

    # Pre-mangle pass for the user-code tags. Must run *before*
    # SuperTransformer / nonlocal / add_helper, so all later passes
    # see the mangled identifiers and stay consistent. Reuses the
    # frame's temp counter so generated mangle names won't collide
    # with subsequent temp_N allocations.
    user_mangle_tags = tags & {'toplevel', 'imports', 'locals', 'methods', 'attrs'}
    if user_mangle_tags:
        if src is None:
            # Fall back to unparsing the tree — symtable needs source.
            src = ast.unparse(tree)
        pool = _NamePool(top_frame.reserved_names, top_frame.get_temp_var)
        type_map = None
        if 'attrs' in user_mangle_tags or 'methods' in user_mangle_tags:
            # Type info disambiguates receivers — `methods` uses it
            # to skip stdlib calls whose attribute happens to share
            # a name with a user-class member; `attrs` needs it to
            # avoid the cliff of mangling every attribute access.
            from .infer import analyze
            type_map = analyze(src)
            if type_map is None and 'attrs' in user_mangle_tags:
                # `attrs` is too aggressive to run without type info —
                # it would mangle every stdlib API call. Refuse rather
                # than producing a broken bundle.
                raise SystemExit(
                    "--replace-name=attrs (or any preset that includes "
                    "it, like `all`) requires mypy to be installed for "
                    "type-driven attribute resolution. Install with "
                    "`pip install mypy`, or use --replace-name=safe."
                )
        apply_mangle(tree, src, user_mangle_tags, pool, type_map=type_map)
        # Refresh reserved set so subsequent temp_N allocation skips
        # whatever the mangler introduced.
        top_frame.reserved_names = collect_user_names(tree)

    # Strip pass — run after mangle (mangle's mypy step needs
    # annotations) but before SuperTransformer / nonlocal. The
    # nonlocal pass and add_helper don't read docstrings or
    # annotations, so removing them here is safe.
    if strip_tags or strip_asserts:
        if src is None:
            src = ast.unparse(tree)
        apply_strip(tree, src, strip_tags, strip_asserts=strip_asserts)

    # Transform "super()" to "super(cls, self)"
    super_trans = SuperTransformer()
    tree = super_trans.visit(tree)

    # Allocate the module-level _FuncHelper var name before running
    # the nonlocal pass — it needs to know the name in case there are
    # top-level try clauses to box.
    top_frame.func_helper_var = top_frame.get_temp_var()

    # Resolve `nonlocal`: walk the tree, find the owner function for
    # each declared name, and rewrite reads/writes to go through the
    # owner's box. Run before add_helper so the injected helper class
    # source isn't itself rewritten.
    apply_nonlocal_pass(
        tree,
        top_frame.get_temp_var,
        module_helper_var=top_frame.func_helper_var,
        top_frame=top_frame,
    )

    add_helper(tree, top_frame, replace_name='global' if 'helper' in tags else 'none')

    expr = parse_stmts(tree.body, top_frame)
    return ast.Module(
        body=[
            ast.Expr(
                expr
            )
        ],
        type_ignores=[],
    )
