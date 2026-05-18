import ast

from .ir import TBranch, TYield, TReturn


# ---------------------------------------------------------------------
# Self-rewriting: replace user names with self attributes

def _self_name(ctx=None):
    """Create a `self` Name marked so _SelfRewriter does not treat it
    as a user-level boxed local. We use this everywhere emit code
    references the state-machine instance — without the marker, a
    user-level parameter literally named `self` (any user-defined
    method whose generator body lives inside a class) would cause
    the rewriter to recursively rewrite our own emitted self
    references into self.self.<...>."""
    n = ast.Name(id='self', ctx=ctx if ctx is not None else ast.Load())
    n._gen_no_self = True
    return n


class _SelfRewriter(ast.NodeTransformer):
    """Replace Name reads/writes for boxed names with self.<name>.
    Name(Store) becomes Attribute(self, name, Store) — parse_assign
    will handle that (setattr).

    Doesn't descend into nested function/class/lambda."""

    def __init__(self, boxed: set):
        self.boxed = boxed

    def visit_Name(self, node):
        if getattr(node, '_gen_no_self', False):
            return node
        if node.id in self.boxed:
            return ast.Attribute(
                value=_self_name(ast.Load()),
                attr=node.id,
                ctx=node.ctx,
            )
        return node

    def visit_NamedExpr(self, node):
        # Walrus: rewrite `(name := expr)` to a tuple-pick expression
        # that stores into self.<name> *and* yields it as the value.
        # We can't use `(self.name := expr)` directly (Python rejects
        # walrus on attribute targets), so we synthesize:
        #   (setattr(self, 'name', expr), self.name)[1]
        # Doing it this way makes the binding survive across yield
        # boundaries, which a plain send-local walrus would not. If
        # the target name isn't in our boxed set (e.g. comprehension-
        # local, lambda parameter — neither of which we descend into
        # here, so this branch is mostly defensive), fall back to
        # leaving the walrus as-is.
        target = node.target
        new_value = self.visit(node.value)
        if isinstance(target, ast.Name) and target.id in self.boxed:
            return ast.Subscript(
                value=ast.Tuple(
                    elts=[
                        ast.Call(
                            func=ast.Name(id='setattr', ctx=ast.Load()),
                            args=[
                                _self_name(ast.Load()),
                                ast.Constant(value=target.id),
                                new_value,
                            ],
                            keywords=[],
                        ),
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr=target.id, ctx=ast.Load(),
                        ),
                    ],
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=1),
                ctx=ast.Load(),
            )
        node.value = new_value
        return node

    def visit_Lambda(self, node):
        # don't descend
        return node

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node


def rewrite_block_to_self(blocks: list, boxed: set):
    """In place rewrite of every block's stmts and terminator
    expressions, replacing user Name references with self.<name>."""
    rewriter = _SelfRewriter(boxed)
    for blk in blocks:
        blk.stmts = [rewriter.visit(s) for s in blk.stmts]
        t = blk.terminator
        if isinstance(t, TBranch):
            t.test = rewriter.visit(t.test)
        elif isinstance(t, TYield):
            t.value = rewriter.visit(t.value)
        elif isinstance(t, TReturn):
            t.value = rewriter.visit(t.value)
        # TGoto, TYieldFrom (its iterable is in the prior block's
        # stmts, already rewritten), TForIter (same), TEnd: nothing.
