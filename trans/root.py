import ast

from .frame import Frame
from .helpers import add_helper
from .nonlocals import apply_nonlocal_pass
from .parsers import parse_stmts
from .passes import SuperTransformer, collect_user_names


def parse_root(tree: ast.Module) -> ast.Module:
    # Init the top frame
    top_frame = Frame(None, [], [])
    top_frame.temp_var_num = 0
    top_frame.reserved_names = collect_user_names(tree)

    # Transform "super()" to "super(cls, self)"
    super_trans = SuperTransformer()
    tree = super_trans.visit(tree)

    # Resolve `nonlocal`: walk the tree, find the owner function for
    # each declared name, and rewrite reads/writes to go through the
    # owner's box. Run before add_helper so the injected helper class
    # source isn't itself rewritten.
    apply_nonlocal_pass(tree, top_frame.get_temp_var)

    top_frame.func_helper_var = top_frame.get_temp_var()
    add_helper(tree, top_frame.func_helper_var)

    expr = parse_stmts(tree.body, top_frame)
    return ast.Module(
        body=[
            ast.Expr(
                expr
            )
        ],
        type_ignores=[],
    )
