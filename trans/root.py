import ast
import sys

from .frame import Frame
from .helpers import add_helper
from .nonlocals import apply_nonlocal_pass
from .parsers import parse_stmts
from .passes import SuperTransformer, collect_user_names


def parse_root(tree: ast.Module) -> ast.Module:
    # The transform recurses through the AST several times — for big
    # source files (e.g. CPython's own test_grammar.py with very
    # nested data literals) the default recursion limit can be too
    # low. The transformed AST is deeper still (every stmt becomes a
    # nested BoolOp / Subscript / Lambda chain), so callers that
    # want to ast.unparse() the result need the bumped limit too.
    # Leave the limit raised — restoring it here would break unparse.
    if sys.getrecursionlimit() < 5000:
        sys.setrecursionlimit(5000)
    return _parse_root_inner(tree)


def _parse_root_inner(tree: ast.Module) -> ast.Module:
    # Init the top frame
    top_frame = Frame(None, [], [])
    top_frame.temp_var_num = 0
    top_frame.reserved_names = collect_user_names(tree)

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

    add_helper(tree, top_frame)

    expr = parse_stmts(tree.body, top_frame)
    return ast.Module(
        body=[
            ast.Expr(
                expr
            )
        ],
        type_ignores=[],
    )
