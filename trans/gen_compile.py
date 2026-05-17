"""Compile a generator function (def with `yield`) into a state-machine
class so onexpr's regular machinery can transform it.

The user writes a generator like:

    def gen():
        for i in range(3):
            yield i
        return 'done'

We compile it to a class whose __next__ is a plain synchronous method
(no yield) — onexpr's existing transform handles that fine. The class
implements the iterator protocol; `iter(gen())` and `yield from gen()`
both work. (We don't fully implement the coroutine .send/.throw
contract — only the read-only iterator side. Generators that try to
receive sent values will see None.)

Pipeline:

  1. CFG construction: walk the body, create a list of basic blocks
     each ending in a terminator (Goto / Branch / Yield / Return /
     ForIter / End). yield is a cut point; if/for/while introduce
     branches and back-edges.

  2. Locals discovery: collect every Name the user reads or writes
     (excluding nested function/class/lambda scopes and excluding
     names from imports/builtins). All of these get boxed onto
     `self.<name>`.

  3. State machine emission: build a ClassDef with __init__ (boxes
     args), __iter__ (returns self), and __next__ (the dispatch
     loop). The dispatch loop is a `while True` containing a chain
     of `if self.state == K:` blocks; each block does the user
     stmts (with names rewritten to self attributes) and then
     handles its terminator.

Phase 1 supports: Assign, AugAssign, AnnAssign, Expr (incl. Yield /
YieldFrom appearing as expression statements), If, For (Name target),
While, Break, Continue, Return, Pass, Import, ImportFrom.
Try / With / nested ClassDef / nested FunctionDef inside the
generator body raise NotImplementedError.

For yield in a non-statement position (e.g. `y = (yield x) + 1`) we
do an A-normal-form lift first: each yield/yield-from in a
sub-expression gets pulled out to a fresh assignment statement.
"""

import ast
import dataclasses
from typing import Optional


# ---------------------------------------------------------------------
# Terminators

@dataclasses.dataclass
class TGoto:
    target: int


@dataclasses.dataclass
class TBranch:
    test: ast.expr
    true: int
    false: int


@dataclasses.dataclass
class TYield:
    value: ast.expr
    next: int


@dataclasses.dataclass
class TYieldFrom:
    iter_var: str        # name of the bound sub-iterator on self
    next: int            # block to enter after sub-iter is exhausted


@dataclasses.dataclass
class TForIter:
    iter_var: str        # name of the bound iterator on self
    target_name: str     # name of the loop variable
    body: int
    after: int


@dataclasses.dataclass
class TReturn:
    value: ast.expr


@dataclasses.dataclass
class TEnd:
    pass


@dataclasses.dataclass
class TReraise:
    """Re-raise the currently-caught exception (saved on self._exc).
    Emitted by the try/except dispatcher's no-handler-matched fall-
    through. The send-level except wrapper keeps a single attribute
    self._exc that holds the active exception while the dispatcher
    decides what to do; if reraise wins, we throw it back out."""
    pass


@dataclasses.dataclass
class TUnreachable:
    """Synthetic terminator for blocks that always exit via an
    explicit `continue` inside their stmts (the try/except
    dispatcher). Emits a defensive raise."""
    pass


@dataclasses.dataclass
class Block:
    id: int
    stmts: list           # body stmts (no control flow, no yield)
    terminator: object    # one of T*
    # When the dispatch loop catches an exception while executing this
    # block, jump to this state instead of re-raising. None means no
    # active try around this block — the exception propagates out of
    # send to the caller.
    exc_handler: int = None

    @property
    def is_terminated(self):
        return self.terminator is not None


# ---------------------------------------------------------------------
# A-normal form — lift yield-in-expression to its own statement

class _ANFLifter(ast.NodeTransformer):
    """For every Yield / YieldFrom that appears in a sub-expression
    position (i.e. not as the entire .value of an Expr / Assign), lift
    it to a fresh `_yt_<n> = yield ...` assignment placed before the
    enclosing statement.

    We do this only inside a generator function body — caller passes
    the body list and we rewrite in place, returning a new list with
    extra prefix statements.

    Importantly, this lifter only descends into expression contexts.
    Compound statements (If / For / While) have their bodies handled
    by the outer anf_transform recursion; visiting them here would
    double-lift any yields in those sub-bodies."""

    def __init__(self, name_provider):
        self._name = name_provider
        self._prefix = []

    def visit_Yield(self, node):
        # Generic visit recurses into the yield's own value first.
        node.value = self.visit(node.value) if node.value else None
        # Lift this yield to a temp.
        tmp = self._name()
        self._prefix.append(
            ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=node,
            )
        )
        return ast.Name(id=tmp, ctx=ast.Load())

    def visit_YieldFrom(self, node):
        node.value = self.visit(node.value)
        tmp = self._name()
        self._prefix.append(
            ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=node,
            )
        )
        return ast.Name(id=tmp, ctx=ast.Load())

    def visit_NamedExpr(self, node):
        # Walrus inside an expression:
        #     ... (x := expr) ...
        # We don't lift it to a separate Assign because that would
        # break short-circuit semantics: a walrus in an `if` test or
        # `while` test gets re-evaluated each iteration, but a lifted
        # Assign before the loop fires only once. Instead we tell the
        # state-machine self-rewriter to leave both the target Name
        # and any references to that name alone, by treating the
        # walrus target as a send-local rather than a self.<name>
        # box. collect_user_locals already skips NamedExpr, so the
        # name simply isn't boxed and the walrus stays as-is.
        node.value = self.visit(node.value)
        return node

    def visit_Lambda(self, node):
        return node  # don't descend

    def visit_FunctionDef(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_ClassDef(self, node):
        return node

    # Compound statements: only visit their direct *expressions*
    # (test/iter/...), not their bodies — the outer anf_transform
    # is responsible for the bodies.
    def visit_If(self, node):
        node.test = self.visit(node.test)
        return node

    def visit_For(self, node):
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        return node

    def visit_AsyncFor(self, node):
        return self.visit_For(node)

    def visit_While(self, node):
        node.test = self.visit(node.test)
        return node

    def visit_With(self, node):
        return node  # phase 1 doesn't support try/with in generator

    def visit_AsyncWith(self, node):
        return node

    def visit_Try(self, node):
        return node


def _is_bare_yield_stmt(stmt):
    """True if stmt is `yield X` or `yield from X` as a top-level
    statement (i.e. an Expr wrapping a Yield/YieldFrom). Such yields
    don't need ANF lifting — they're already at the right level."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
    )


def _is_bare_yield_assign(stmt):
    """True if stmt is `x = yield ...` — also OK without lifting."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, (ast.Yield, ast.YieldFrom))
    )


def anf_transform(body: list, name_provider, all_locals=None) -> list:
    """Walk a generator body, returning a new list of statements where
    every Yield / YieldFrom appears either as a top-level expression
    statement or as the entire RHS of a simple-name assignment.

    Recurses into If / For / While bodies but stops at nested function
    / class / lambda — those have their own scopes.

    `all_locals`, if given, is the full set of names that will be
    boxed onto self in the final state-machine class (computed by
    collect_user_locals on the original body). It's used only to
    dehydrate them as send-local variables right before a nested
    def/class so the nested scope's closure can reach them; without
    this, a `def adder(x): return base + x` inside a generator with
    `base` boxed would see no `base` at all (it's only on self)."""
    out = []

    def lift(stmt):
        # If this stmt is already in good form, leave its yield alone
        # but recurse into nested bodies.
        if _is_bare_yield_stmt(stmt) or _is_bare_yield_assign(stmt):
            return [stmt]

        lifter = _ANFLifter(name_provider)
        new_stmt = lifter.visit(stmt)
        return lifter._prefix + [new_stmt]

    for stmt in body:
        if isinstance(stmt, ast.Match):
            from .match_patterns import compile_match
            fake_frame = _MiniFrame(name_provider)
            flat = compile_match(stmt, fake_frame)
            _mark_match_origin(flat)
            out.extend(anf_transform(flat, name_provider, all_locals))
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested def/class: emit dehydrate-from-self stmts first
            # so the nested scope's closure can capture boxed user
            # locals as cell variables. Then the def itself, then a
            # rebind that boxes the new function/class onto self.
            if all_locals:
                free = _free_user_names(stmt, all_locals)
                for name in sorted(free):
                    lhs = ast.Name(id=name, ctx=ast.Store())
                    lhs._gen_no_self = True
                    out.append(
                        ast.Assign(
                            targets=[lhs],
                            value=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=name,
                                ctx=ast.Load(),
                            ),
                        )
                    )
            out.append(stmt)
            rhs = ast.Name(id=stmt.name, ctx=ast.Load())
            rhs._gen_no_self = True
            out.append(
                ast.Assign(
                    targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
                    value=rhs,
                )
            )
            continue
        if isinstance(stmt, ast.If):
            if getattr(stmt, '_from_match', False):
                # Match-origin If: recurse into bodies but don't lift
                # the test, which contains short-circuit walrus
                # captures we must preserve.
                stmt.body = anf_transform(stmt.body, name_provider, all_locals)
                stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
                out.append(stmt)
                continue
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.For):
            # Tuple/list unpack target in a generator's for loop:
            # rewrite `for a, b in iter:` to
            #   `for _tmp in iter: a, b = _tmp; <body>`
            # so the state machine only ever sees a simple Name target.
            if not isinstance(stmt.target, ast.Name):
                tmp = name_provider()
                unpack = ast.Assign(
                    targets=[stmt.target],
                    value=ast.Name(id=tmp, ctx=ast.Load()),
                )
                stmt.body = [unpack] + stmt.body
                stmt.target = ast.Name(id=tmp, ctx=ast.Store())
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            # The .iter expression itself can contain yield — lift it.
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.While):
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.Try):
            stmt.body = anf_transform(stmt.body, name_provider, all_locals)
            for h in stmt.handlers:
                if h.name is None:
                    # Bare `except E:` — give it a synthetic name so
                    # bare `raise` inside the handler can refer to it
                    # (parse_raise on the send method has no exc stack
                    # because it's a fresh function frame).
                    h.name = name_provider()
                _rewrite_bare_raise_to_named(h.body, h.name)
                h.body = anf_transform(h.body, name_provider, all_locals)
            stmt.orelse = anf_transform(stmt.orelse, name_provider, all_locals)
            stmt.finalbody = anf_transform(stmt.finalbody, name_provider, all_locals)
            out.append(stmt)
            continue
        if isinstance(stmt, ast.With):
            # `with X as v: BODY` (possibly with multiple items) is
            # equivalent to nested withs. We lower to PEP 343 form
            # using try/except/finally so the existing CFG try
            # handling carries the with through yield boundaries.
            lowered = _lower_with_to_try(stmt, name_provider)
            out.extend(anf_transform(lowered, name_provider, all_locals))
            continue
        out.extend(lift(stmt))

    return out


def _rewrite_bare_raise_to_named(body: list, name: str):
    """In-place: every bare `raise` (no exc) in `body` becomes
    `raise <name>`. Doesn't descend into nested function/class/lambda
    or into another except handler (which would have its own name to
    bind to)."""
    class _R(ast.NodeTransformer):
        def visit_Raise(self, node):
            if node.exc is None:
                return ast.Raise(
                    exc=ast.Name(id=name, ctx=ast.Load()),
                    cause=node.cause,
                )
            return node

        def visit_FunctionDef(self, node): return node
        def visit_AsyncFunctionDef(self, node): return node
        def visit_ClassDef(self, node): return node
        def visit_Lambda(self, node): return node
        def visit_Try(self, node):
            # Body and finally still see this name as the active
            # exception. except handlers shadow with their own name
            # (bare raise inside a nested handler refers to that
            # nested except's exception, not ours).
            node.body = [self.visit(s) for s in node.body]
            node.orelse = [self.visit(s) for s in node.orelse]
            node.finalbody = [self.visit(s) for s in node.finalbody]
            # Don't descend into node.handlers — they re-enter
            # _rewrite_bare_raise_to_named with their own name in the
            # outer pass.
            return node

    r = _R()
    for i, s in enumerate(body):
        body[i] = r.visit(s)


def _lower_with_to_try(stmt: ast.With, name_provider) -> list:
    """Rewrite `with X as v: BODY` to PEP 343-equivalent try/except/
    finally so the generator CFG can carry the with across yields.

    For multiple items `with X, Y: BODY`, recurse: outer with X, inner
    with Y over BODY.
    """
    if len(stmt.items) > 1:
        first, *rest = stmt.items
        inner = ast.With(items=rest, body=stmt.body)
        return _lower_with_to_try(
            ast.With(items=[first], body=[inner]),
            name_provider,
        )
    item = stmt.items[0]
    mgr = name_provider()
    exc_flag = name_provider()
    e_name = name_provider()

    # mgr = ctx_expr
    setup = [
        ast.Assign(
            targets=[ast.Name(id=mgr, ctx=ast.Store())],
            value=item.context_expr,
        ),
    ]
    # __enter__ — bind to as-name if any.
    enter_call = ast.Call(
        func=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id='type', ctx=ast.Load()),
                args=[ast.Name(id=mgr, ctx=ast.Load())],
                keywords=[],
            ),
            attr='__enter__', ctx=ast.Load(),
        ),
        args=[ast.Name(id=mgr, ctx=ast.Load())],
        keywords=[],
    )
    if item.optional_vars is not None:
        setup.append(ast.Assign(
            targets=[item.optional_vars],
            value=enter_call,
        ))
    else:
        setup.append(ast.Expr(value=enter_call))
    # exc_flag = True
    setup.append(ast.Assign(
        targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
        value=ast.Constant(value=True),
    ))

    # except BaseException as e:
    #     exc_flag = False
    #     if not type(mgr).__exit__(mgr, type(e), e, e.__traceback__):
    #         raise
    handler_body = [
        ast.Assign(
            targets=[ast.Name(id=exc_flag, ctx=ast.Store())],
            value=ast.Constant(value=False),
        ),
        ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Name(id='type', ctx=ast.Load()),
                            args=[ast.Name(id=mgr, ctx=ast.Load())],
                            keywords=[],
                        ),
                        attr='__exit__', ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=mgr, ctx=ast.Load()),
                        ast.Call(
                            func=ast.Name(id='type', ctx=ast.Load()),
                            args=[ast.Name(id=e_name, ctx=ast.Load())],
                            keywords=[],
                        ),
                        ast.Name(id=e_name, ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Name(id=e_name, ctx=ast.Load()),
                            attr='__traceback__', ctx=ast.Load(),
                        ),
                    ],
                    keywords=[],
                ),
            ),
            body=[ast.Raise(
                exc=ast.Name(id=e_name, ctx=ast.Load()),
                cause=None,
            )],
            orelse=[],
        ),
    ]
    # finally:
    #     if exc_flag:
    #         type(mgr).__exit__(mgr, None, None, None)
    final_body = [
        ast.If(
            test=ast.Name(id=exc_flag, ctx=ast.Load()),
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id='type', ctx=ast.Load()),
                                args=[ast.Name(id=mgr, ctx=ast.Load())],
                                keywords=[],
                            ),
                            attr='__exit__', ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(id=mgr, ctx=ast.Load()),
                            ast.Constant(value=None),
                            ast.Constant(value=None),
                            ast.Constant(value=None),
                        ],
                        keywords=[],
                    )
                ),
            ],
            orelse=[],
        ),
    ]
    try_node = ast.Try(
        body=stmt.body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id='BaseException', ctx=ast.Load()),
                name=e_name,
                body=handler_body,
            )
        ],
        orelse=[],
        finalbody=final_body,
    )
    return setup + [try_node]


def _free_user_names(node, all_locals: set) -> set:
    """Return the subset of `all_locals` that's referenced (Load ctx)
    inside `node` (a nested def/class). Descends into the entire
    subtree because we want to dehydrate any name the nested scope
    might close over, including in deeply nested expressions."""
    found = set()

    class _V(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id in all_locals:
                found.add(n.id)

    _V().visit(node)
    return found


class _MiniFrame:
    """Adapter so compile_match (which expects a Frame-like object with
    get_temp_var) can be called from the generator pipeline."""

    def __init__(self, name_provider):
        self.get_temp_var = name_provider


def _mark_match_origin(nodes):
    """Tag every If statement reachable from these nodes (without
    crossing nested function/class/lambda scopes) as match-origin so
    anf_transform skips lifting their tests."""

    class _M(ast.NodeVisitor):
        def visit_If(self, node):
            node._from_match = True
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)

        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass

    m = _M()
    for n in nodes:
        m.visit(n)


# ---------------------------------------------------------------------
# Locals discovery

def collect_user_locals(body: list, args: ast.arguments) -> set:
    """Names that the user reads or writes at top level inside the
    generator body, plus the function's parameters. Used to decide
    what to box on self.

    Doesn't recurse into nested function/class/lambda — those have
    their own scope. Names declared `nonlocal` or `global` are
    excluded: those resolve up the closure / module scope and are
    rewritten to `outer_helper._b_<name>` / `globals()[<name>]` by
    the nonlocal pre-pass before SelfRewriter runs."""
    names = set()

    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            names.add(a.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)

    # Collect direct nonlocal / global declarations — those names must
    # NOT be boxed on self.
    nonlocal_or_global = set()
    class _DeclWalker(ast.NodeVisitor):
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
        def visit_Lambda(self, node): pass
        def visit_Nonlocal(self, node):
            for n in node.names:
                nonlocal_or_global.add(n)
        def visit_Global(self, node):
            for n in node.names:
                nonlocal_or_global.add(n)
    dw = _DeclWalker()
    for stmt in body:
        dw.visit(stmt)

    def add_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    class _Walker(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)
        def visit_AsyncFunctionDef(self, node):
            names.add(node.name)
        def visit_ClassDef(self, node):
            names.add(node.name)
        def visit_Lambda(self, node):
            pass

        def visit_ExceptHandler(self, node):
            if node.name is not None:
                names.add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_Assign(self, node):
            for t in node.targets:
                add_target(t)
            self.visit(node.value)

        def visit_AugAssign(self, node):
            add_target(node.target)
            self.visit(node.value)

        def visit_AnnAssign(self, node):
            add_target(node.target)
            if node.value is not None:
                self.visit(node.value)

        def visit_For(self, node):
            add_target(node.target)
            self.visit(node.iter)
            for s in node.body + node.orelse:
                self.visit(s)

        def visit_Name(self, node):
            # Reads also matter — they tell us this name is a local
            # (vs. a module-level reference).
            if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                # A pure Load could be a global; we can't tell statically
                # without a full scope analysis. To stay safe we DON'T
                # add Load-only names; only Store / Del. This means a
                # generator that only reads a closure variable doesn't
                # accidentally box that name.
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)

        def visit_NamedExpr(self, node):
            # Walrus targets ARE boxed in the generator. _SelfRewriter
            # rewrites the walrus to a setattr-based tuple expression
            # so the binding survives across yield boundaries.
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            self.visit(node.value)

        def visit_Import(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)

    walker = _Walker()
    for stmt in body:
        walker.visit(stmt)

    # Remove names declared nonlocal / global — those resolve up the
    # closure / module scope, not on self.
    names -= nonlocal_or_global

    return names


# ---------------------------------------------------------------------
# CFG construction

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


class _CFGBuilder:
    def __init__(self, name_provider):
        self.blocks: list[Block] = []
        self.name = name_provider
        # Stack of (continue_target, break_target) for nested loops.
        self.loop_stack: list[tuple[int, int]] = []
        # Stack of currently-active try regions. Each entry is a dict
        # carrying the handler dispatcher block id (where to jump when
        # an exception fires inside the try body) and other metadata
        # used by phase 2 (finally support).
        self.try_stack: list[dict] = []

    def _current_handler(self):
        """Top of try_stack's handler block id, or None if no active
        try. Used to stamp every newly-emitted block so the dispatch
        loop knows where to redirect on exception."""
        return self.try_stack[-1]['handler'] if self.try_stack else None

    def new_block(self) -> Block:
        b = Block(
            id=len(self.blocks), stmts=[], terminator=None,
            exc_handler=self._current_handler(),
        )
        self.blocks.append(b)
        return b

    def emit(self, body: list, current: Block) -> Block:
        """Append `body`'s statements to the CFG starting at `current`.
        Returns the (possibly new) current block at the end (could be
        already-terminated, in which case caller should not append
        more)."""
        for stmt in body:
            current = self._emit_one(stmt, current)
            if current.is_terminated:
                # Anything after a terminating statement is dead code.
                # We still need a block to represent "after dead code"
                # for callers, but the body iteration just stops.
                break
        return current

    def _innermost_finally(self):
        """Return the topmost try_stack entry that has a finally (so
        break/continue/return inside the try body must run that
        finally before reaching their target). None if there's no
        active try with finally."""
        for entry in reversed(self.try_stack):
            if entry.get('finally_entry') is not None:
                return entry
        return None

    def _innermost_finally_in_loop(self):
        """Like _innermost_finally, but only returns entries that were
        pushed *inside* the current innermost loop. break/continue
        only need to detour through finallies that sit between them
        and the loop exit."""
        if not self.loop_stack:
            return None
        loop_idx = len(self.loop_stack) - 1
        for entry in reversed(self.try_stack):
            if entry.get('finally_entry') is None:
                continue
            if entry.get('loop_depth_at_push', -1) < loop_idx:
                # The finally is outside the current loop — break /
                # continue exit the loop without going through it.
                return None
            return entry
        return None

    def _emit_one(self, stmt, current: Block) -> Block:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            return self._emit_yield(stmt.value.value, None, current)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.YieldFrom):
            return self._emit_yield_from(stmt.value.value, None, current)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) \
                and isinstance(stmt.value, ast.Yield):
            return self._emit_yield(
                stmt.value.value, stmt.targets[0].id, current,
            )
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) \
                and isinstance(stmt.value, ast.YieldFrom):
            return self._emit_yield_from(
                stmt.value.value, stmt.targets[0].id, current,
            )
        if isinstance(stmt, ast.Return):
            fin = self._innermost_finally()
            if fin is not None:
                # Stash the return value on self.<outcome>_value, set
                # outcome=return, jump to finally. The finally's
                # outcome router will re-emit a TReturn after running.
                outcome = fin['outcome_attr']
                value_attr = outcome + '_retval'
                ret_val = (stmt.value if stmt.value is not None
                           else ast.Constant(value=None))
                current.stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=value_attr, ctx=ast.Store(),
                            )
                        ],
                        value=ret_val,
                    )
                )
                fin.setdefault('uses_return', True)
                current.stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=outcome, ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value='return'),
                    )
                )
                current.terminator = TGoto(target=fin['finally_entry'])
                return current
            current.terminator = TReturn(
                value=stmt.value if stmt.value is not None else ast.Constant(value=None)
            )
            return current
        if isinstance(stmt, ast.If):
            return self._emit_if(stmt, current)
        if isinstance(stmt, ast.For):
            return self._emit_for(stmt, current)
        if isinstance(stmt, ast.While):
            return self._emit_while(stmt, current)
        if isinstance(stmt, ast.Break):
            if not self.loop_stack:
                raise SyntaxError("'break' outside loop")
            _cont, brk = self.loop_stack[-1]
            fin = self._innermost_finally_in_loop()
            if fin is not None:
                fin.setdefault('uses_break', set()).add(brk)
                current.stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=fin['outcome_attr'], ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=f'break:{brk}'),
                    )
                )
                current.terminator = TGoto(target=fin['finally_entry'])
                return current
            current.terminator = TGoto(target=brk)
            return current
        if isinstance(stmt, ast.Continue):
            if not self.loop_stack:
                raise SyntaxError("'continue' outside loop")
            cont, _brk = self.loop_stack[-1]
            fin = self._innermost_finally_in_loop()
            if fin is not None:
                fin.setdefault('uses_continue', set()).add(cont)
                current.stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=fin['outcome_attr'], ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=f'continue:{cont}'),
                    )
                )
                current.terminator = TGoto(target=fin['finally_entry'])
                return current
            current.terminator = TGoto(target=cont)
            return current
        if isinstance(stmt, ast.Try):
            # Try statement that doesn't cross a yield AND doesn't
            # contain a return/break/continue: emit it verbatim, the
            # surrounding parse_try in onexpr handles it at the lambda
            # level. Only when a yield lives inside one of the clauses
            # (or a control-flow stmt that needs to escape the try)
            # do we need to break the try across CFG blocks.
            if (
                not _stmt_contains_yield(stmt)
                and not _stmt_contains_break_continue_return(stmt)
            ):
                current.stmts.append(stmt)
                return current
            return self._emit_try(stmt, current)
        if isinstance(stmt, ast.With):
            # Same fast-path: with-no-yield stays at the lambda level.
            # `with` crossing yield got lowered to try/except/finally
            # in anf_transform, so by the time _emit_one sees a real
            # ast.With, we're guaranteed it doesn't cross a yield.
            current.stmts.append(stmt)
            return current
        # Anything else (Assign, AugAssign, AnnAssign, Expr-without-yield,
        # Import, Pass, etc.) — just append to current block. We trust
        # there's no embedded yield because anf_transform lifted them.
        if _stmt_contains_yield(stmt):
            raise NotImplementedError(
                f"unsupported statement containing yield: {type(stmt).__name__}"
            )
        current.stmts.append(stmt)
        return current

    def _emit_yield(self, value, capture_name, current: Block) -> Block:
        """`yield value` (or `name = yield value`). The current block's
        terminator becomes TYield with the next block. The next block,
        if capture_name, starts with `<name> = sent_value` (we read
        the sent value from a known slot)."""
        nxt = self.new_block()
        current.terminator = TYield(
            value=value if value is not None else ast.Constant(value=None),
            next=nxt.id,
        )
        if capture_name is not None:
            nxt.stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=capture_name, ctx=ast.Store())],
                    value=ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='_sent',
                        ctx=ast.Load(),
                    ),
                )
            )
        return nxt

    def _emit_yield_from(self, iterable, capture_name, current: Block) -> Block:
        """`yield from X` (or `name = yield from X`). Bind iter(X) to
        a fresh slot in the *current* block, then transition to a new
        block whose terminator drives the sub-iterator. After the
        sub-iter is exhausted, optionally capture its return value as
        <name>. Phase 1: the captured value is None (yield from on a
        plain iterator loses StopIteration.value)."""
        sub_var = self.name()
        # iter setup goes in current block (runs once on entry to the
        # yield-from sequence).
        current.stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr=sub_var,
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='iter', ctx=ast.Load()),
                    args=[iterable],
                    keywords=[],
                ),
            )
        )
        # Drive block — entered repeatedly by the state machine, each
        # entry pulls one value or transitions to nxt on exhaustion.
        drive = self.new_block()
        nxt = self.new_block()
        current.terminator = TGoto(target=drive.id)
        drive.terminator = TYieldFrom(iter_var=sub_var, next=nxt.id)
        if capture_name is not None:
            # Bind the yield-from return value (StopIteration.value
            # captured by the TYieldFrom emission) to the user's
            # capture name. Marked _gen_no_self so the rhs reads our
            # generated slot rather than self.<sub_var>_value.<value>.
            rhs = ast.Attribute(
                value=_self_name(ast.Load()),
                attr='_yfrom_value',
                ctx=ast.Load(),
            )
            nxt.stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=capture_name, ctx=ast.Store())],
                    value=rhs,
                )
            )
        return nxt

    def _emit_if(self, stmt, current) -> Block:
        merge = self.new_block()
        # True branch
        true_block = self.new_block()
        end_true = self.emit(stmt.body, true_block)
        if not end_true.is_terminated:
            end_true.terminator = TGoto(target=merge.id)
        # False branch
        false_block = self.new_block()
        end_false = self.emit(stmt.orelse, false_block)
        if not end_false.is_terminated:
            end_false.terminator = TGoto(target=merge.id)
        current.terminator = TBranch(
            test=stmt.test, true=true_block.id, false=false_block.id,
        )
        return merge

    def _emit_for(self, stmt, current) -> Block:
        if not isinstance(stmt.target, ast.Name):
            raise NotImplementedError(
                "for-target must be a simple Name in a generator function"
            )
        sub_var = self.name()
        # iter setup goes in current
        current.stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr=sub_var,
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id='iter', ctx=ast.Load()),
                    args=[stmt.iter],
                    keywords=[],
                ),
            )
        )
        head = self.new_block()
        body_b = self.new_block()
        # `after` is what we jump to on exhaustion (no break). If
        # there's an else, we splice it in between exhaustion and the
        # join. break goes to `join` directly to skip else.
        if stmt.orelse:
            else_b = self.new_block()
            join = self.new_block()
            else_end = self.emit(stmt.orelse, else_b)
            if not else_end.is_terminated:
                else_end.terminator = TGoto(target=join.id)
            after = else_b
            break_target = join
        else:
            after = self.new_block()
            join = after
            break_target = after
        current.terminator = TGoto(target=head.id)
        head.terminator = TForIter(
            iter_var=sub_var,
            target_name=stmt.target.id,
            body=body_b.id,
            after=after.id,
        )
        self.loop_stack.append((head.id, break_target.id))
        end_body = self.emit(stmt.body, body_b)
        self.loop_stack.pop()
        if not end_body.is_terminated:
            end_body.terminator = TGoto(target=head.id)
        return join

    def _emit_while(self, stmt, current) -> Block:
        head = self.new_block()
        body_b = self.new_block()
        if stmt.orelse:
            else_b = self.new_block()
            join = self.new_block()
            else_end = self.emit(stmt.orelse, else_b)
            if not else_end.is_terminated:
                else_end.terminator = TGoto(target=join.id)
            after = else_b
            break_target = join
        else:
            after = self.new_block()
            join = after
            break_target = after
        current.terminator = TGoto(target=head.id)
        head.terminator = TBranch(
            test=stmt.test, true=body_b.id, false=after.id,
        )
        self.loop_stack.append((head.id, break_target.id))
        end_body = self.emit(stmt.body, body_b)
        self.loop_stack.pop()
        if not end_body.is_terminated:
            end_body.terminator = TGoto(target=head.id)
        return join

    def _emit_try(self, stmt: ast.Try, current: Block) -> Block:
        """Try/except (and try/finally) crossing yield. Strategy:

            [current] --(no exc)--> [body_entry] ... [body_end] --> join
                                                           or [else_entry]
                                                                 ...
                                                                 [else_end] --> join
            on exc inside body/else: --> [dispatcher]
                                              if isinstance(exc, T1): goto h1
                                              elif ...: goto h2
                                              else: reraise via [reraise]
            [h1] body --> join
            [h2] body --> join
            [join] = caller's continuation

        Each block created with the try region active gets exc_handler
        = dispatcher.id stamped on it (via _CFGBuilder.new_block).
        """
        if stmt.finalbody:
            return self._emit_try_finally(stmt, current)

        # Phase 1: try/except[/else], no finally.
        join = self.new_block()
        # Allocate dispatcher BEFORE pushing the try region so that
        # its own blocks (the handler bodies) are NOT routed back to
        # itself on a fresh exception — they go to the enclosing
        # handler if any.
        dispatcher = self.new_block()

        self.try_stack.append({'handler': dispatcher.id})
        body_entry = self.new_block()
        current.terminator = TGoto(target=body_entry.id)
        body_end = self.emit(stmt.body, body_entry)
        self.try_stack.pop()

        if stmt.orelse:
            else_entry = self.new_block()
            if not body_end.is_terminated:
                body_end.terminator = TGoto(target=else_entry.id)
            else_end = self.emit(stmt.orelse, else_entry)
            if not else_end.is_terminated:
                else_end.terminator = TGoto(target=join.id)
        else:
            if not body_end.is_terminated:
                body_end.terminator = TGoto(target=join.id)

        # Build dispatcher: chained isinstance checks, fall-through
        # = re-raise via TReraise.
        reraise_blk = self.new_block()
        reraise_blk.terminator = TReraise()
        cur_orelse = [
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='state', ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=reraise_blk.id),
            ),
            ast.Continue(),
        ]
        for h in reversed(stmt.handlers):
            h_entry = self.new_block()
            if h.name is not None:
                h_entry.stmts.append(
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=h.name, ctx=ast.Store(),
                            )
                        ],
                        value=ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='_exc', ctx=ast.Load(),
                        ),
                    )
                )
            h_end = self.emit(h.body, h_entry)
            if not h_end.is_terminated:
                h_end.terminator = TGoto(target=join.id)
            if h.type is None:
                cond = ast.Constant(value=True)
            else:
                cond = ast.Call(
                    func=ast.Name(id='isinstance', ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='_exc', ctx=ast.Load(),
                        ),
                        h.type,
                    ],
                    keywords=[],
                )
            cur_orelse = [
                ast.If(
                    test=cond,
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=_self_name(ast.Load()),
                                    attr='state', ctx=ast.Store(),
                                )
                            ],
                            value=ast.Constant(value=h_entry.id),
                        ),
                        ast.Continue(),
                    ],
                    orelse=cur_orelse,
                )
            ]
        dispatcher.stmts = cur_orelse
        dispatcher.terminator = TUnreachable()
        return join

    def _emit_try_finally(self, stmt: ast.Try, current: Block) -> Block:
        """Try with a finally clause crossing yield.

        Structure (try / except / else / finally — each part optional
        except `try` and `finally`):

            [current] -> [body_entry] ... [body_end]
                                              \
                                               > set fin_outcome=normal -> [finally_entry]
                                              /
            on exc inside body/else: -> [dispatcher]
                                              if matches handler: set fin_outcome=normal -> [hN]
                                              else:               set fin_outcome=exc    -> [finally_entry]
            [hN_body] -> set fin_outcome=normal -> [finally_entry]
            on exc inside hN: -> set fin_outcome=exc -> [finally_entry]
            [else_entry] -> [else_end] -> set fin_outcome=normal -> [finally_entry]

            [finally_entry] -> ... user finalbody ... -> [finally_end] -> [outcome_router]
                                                                    fin_outcome
                                                                       ↳ normal -> join
                                                                       ↳ exc    -> reraise self._exc

        We track outcome on a self attribute named after the try id.
        Phase 2 doesn't yet route Return / Break / Continue inside
        the try body through finally — those still terminate the
        respective block early and don't run the finally clause. A
        later improvement can intercept TReturn / break / continue
        emitted inside the try region and route them via the
        finally first.
        """
        fin_id = self.name()
        outcome_attr = '_fin_' + fin_id
        join = self.new_block()
        finally_entry = self.new_block()

        def set_outcome(label):
            return ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr=outcome_attr, ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=label),
            )

        # Build dispatcher BEFORE the try region so blocks created
        # inside the body get exc_handler = dispatcher.id.
        dispatcher = self.new_block() if stmt.handlers else None
        if dispatcher is None:
            # Pure try/finally: any exception in body sets outcome=exc
            # and goes straight to finally_entry. We model this with
            # a dispatcher block that sets outcome and gotos finally.
            dispatcher = self.new_block()
            dispatcher.stmts = [
                set_outcome('exc'),
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=finally_entry.id),
                ),
                ast.Continue(),
            ]
            dispatcher.terminator = TUnreachable()

        try_entry = {
            'handler': dispatcher.id,
            'finally_entry': finally_entry.id,
            'outcome_attr': outcome_attr,
            'loop_depth_at_push': len(self.loop_stack) - 1,
        }
        self.try_stack.append(try_entry)
        body_entry = self.new_block()
        current.terminator = TGoto(target=body_entry.id)
        body_end = self.emit(stmt.body, body_entry)
        self.try_stack.pop()

        # else (only if no exception)
        if stmt.orelse:
            self.try_stack.append(try_entry)
            else_entry = self.new_block()
            if not body_end.is_terminated:
                body_end.stmts.append(set_outcome('normal'))
                body_end.terminator = TGoto(target=else_entry.id)
            else_end = self.emit(stmt.orelse, else_entry)
            self.try_stack.pop()
            if not else_end.is_terminated:
                else_end.stmts.append(set_outcome('normal'))
                else_end.terminator = TGoto(target=finally_entry.id)
        else:
            if not body_end.is_terminated:
                body_end.stmts.append(set_outcome('normal'))
                body_end.terminator = TGoto(target=finally_entry.id)

        # If we have explicit handlers, build the dispatcher chain.
        if stmt.handlers:
            cur_orelse = [
                set_outcome('exc'),
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=finally_entry.id),
                ),
                ast.Continue(),
            ]
            for h in reversed(stmt.handlers):
                # Build the per-handler exception dispatcher BEFORE
                # the handler entry so h_entry inherits exc_handler
                # = h_exc_disp.id.
                h_exc_disp = self.new_block()
                h_exc_disp.stmts = [
                    set_outcome('exc'),
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr='state', ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=finally_entry.id),
                    ),
                    ast.Continue(),
                ]
                h_exc_disp.terminator = TUnreachable()
                # Push the try region so any block created from now on
                # is stamped with exc_handler = h_exc_disp.id.
                self.try_stack.append({
                    'handler': h_exc_disp.id,
                    'finally_entry': finally_entry.id,
                    'outcome_attr': outcome_attr,
                    'loop_depth_at_push': len(self.loop_stack) - 1,
                })
                h_entry = self.new_block()
                if h.name is not None:
                    h_entry.stmts.append(
                        ast.Assign(
                            targets=[
                                ast.Attribute(
                                    value=_self_name(ast.Load()),
                                    attr=h.name, ctx=ast.Store(),
                                )
                            ],
                            value=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr='_exc', ctx=ast.Load(),
                            ),
                        )
                    )
                h_end = self.emit(h.body, h_entry)
                self.try_stack.pop()
                if not h_end.is_terminated:
                    h_end.stmts.append(set_outcome('normal'))
                    h_end.terminator = TGoto(target=finally_entry.id)
                if h.type is None:
                    cond = ast.Constant(value=True)
                else:
                    cond = ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr='_exc', ctx=ast.Load(),
                            ),
                            h.type,
                        ],
                        keywords=[],
                    )
                cur_orelse = [
                    ast.If(
                        test=cond,
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='state', ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Constant(value=h_entry.id),
                            ),
                            ast.Continue(),
                        ],
                        orelse=cur_orelse,
                    )
                ]
            dispatcher.stmts = cur_orelse
            dispatcher.terminator = TUnreachable()

        # finally body — runs unconditionally. If anything in the
        # finally body itself raises, that propagates to the enclosing
        # try (or out of send if none).
        finally_end = self.emit(stmt.finalbody, finally_entry)
        # outcome router after finally: read outcome and act. We
        # build a chain of `if outcome == X: ...` branches in reverse.
        if not finally_end.is_terminated:
            router_chain = [
                # Default tail — outcome=='normal': fall through to join.
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=join.id),
                ),
                ast.Continue(),
            ]
            # break:N — set state=N, continue.
            for brk_target in try_entry.get('uses_break', set()):
                router_chain = [
                    ast.If(
                        test=ast.Compare(
                            left=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=outcome_attr, ctx=ast.Load(),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(
                                value=f'break:{brk_target}'
                            )],
                        ),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='state', ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Constant(value=brk_target),
                            ),
                            ast.Continue(),
                        ],
                        orelse=router_chain,
                    )
                ]
            # continue:N — set state=N, continue.
            for cont_target in try_entry.get('uses_continue', set()):
                router_chain = [
                    ast.If(
                        test=ast.Compare(
                            left=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=outcome_attr, ctx=ast.Load(),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(
                                value=f'continue:{cont_target}'
                            )],
                        ),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='state', ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Constant(value=cont_target),
                            ),
                            ast.Continue(),
                        ],
                        orelse=router_chain,
                    )
                ]
            # return — raise StopIteration(self.<outcome>_retval).
            if try_entry.get('uses_return'):
                router_chain = [
                    ast.If(
                        test=ast.Compare(
                            left=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr=outcome_attr, ctx=ast.Load(),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value='return')],
                        ),
                        body=[
                            # PEP 479 marker: this StopIteration is from
                            # `return` (routed through finally), not a
                            # user-level raise.
                            ast.Assign(
                                targets=[ast.Attribute(
                                    value=_self_name(ast.Load()),
                                    attr='_stopping_via_return', ctx=ast.Store(),
                                )],
                                value=ast.Constant(value=True),
                            ),
                            ast.Raise(
                                exc=ast.Call(
                                    func=ast.Name(id='StopIteration', ctx=ast.Load()),
                                    args=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr=outcome_attr + '_retval',
                                            ctx=ast.Load(),
                                        )
                                    ],
                                    keywords=[],
                                ),
                                cause=None,
                            ),
                        ],
                        orelse=router_chain,
                    )
                ]
            # exc — reraise.
            router_chain = [
                ast.If(
                    test=ast.Compare(
                        left=ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr=outcome_attr, ctx=ast.Load(),
                        ),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(value='exc')],
                    ),
                    body=[
                        ast.Raise(
                            exc=ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr='_exc', ctx=ast.Load(),
                            ),
                            cause=None,
                        ),
                    ],
                    orelse=router_chain,
                )
            ]
            finally_end.stmts.extend(router_chain)
            finally_end.terminator = TUnreachable()

        return join


def _stmt_contains_yield(stmt) -> bool:
    """Does this statement contain a Yield/YieldFrom anywhere (not
    descending into nested function/class/lambda)?"""
    class _V(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Yield(self, node): self.found = True
        def visit_YieldFrom(self, node): self.found = True
        def visit_Lambda(self, node): pass
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
    v = _V()
    v.visit(stmt)
    return v.found


def _stmt_contains_break_continue_return(stmt) -> bool:
    """Does this statement contain Break/Continue/Return at the
    generator-body level (not inside a nested loop/function)?

    For the generator state machine: a try whose body contains any of
    these must go through the CFG so the surrounding finally/loop can
    intercept them via outcome routing. Otherwise the fast path keeps
    the try at the lambda level where Return short-circuits the send
    method, causing the value to be yielded instead of stop-iterated.
    """
    class _V(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Return(self, node): self.found = True
        def visit_Break(self, node): self.found = True
        def visit_Continue(self, node): self.found = True
        def visit_Lambda(self, node): pass
        def visit_FunctionDef(self, node): pass
        def visit_AsyncFunctionDef(self, node): pass
        def visit_ClassDef(self, node): pass
    v = _V()
    v.visit(stmt)
    return v.found


def build_cfg(body: list, name_provider) -> list:
    """Returns a list of Blocks. Block 0 is the entry. The function
    must have a final terminator on every path; we add an implicit
    TEnd at the tail if the user's body falls off the end."""
    builder = _CFGBuilder(name_provider)
    entry = builder.new_block()
    end = builder.emit(body, entry)
    if not end.is_terminated:
        end.terminator = TEnd()
    return builder.blocks


# ---------------------------------------------------------------------
# Self-rewriting: replace user names with self attributes

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


# ---------------------------------------------------------------------
# Emission: build the state-machine class

_GEN_DONE = '_GEN_DONE_SENTINEL'


def emit_state_machine(
    name: str,
    args: ast.arguments,
    blocks: list,
    boxed: set,
    decorator_list: list,
    is_async: bool = False,
    async_kind: str = None,
    gen_self_alias: str = None,
    returns: ast.expr = None,
) -> list:
    """Build a sequence of statements that defines the generator class
    and binds it to `name`. Returned statements are intended to be
    spliced into the enclosing scope and processed by parse_stmts /
    parse_class_def normally."""

    # __init__: setattr(_inst, '<arg>', <arg>) for every parameter,
    # then setattr(_inst, 'state', 0) and any housekeeping flags.
    # We use `_inst` instead of `self` for the init's first parameter
    # because the user's generator may itself be a method whose first
    # parameter is named `self` — that would clash with the implicit
    # `self` we'd otherwise inject. `_inst` avoids the collision and
    # is never a user-visible name.
    init_self = '_inst'
    init_body: list = []
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            init_body.append(
                ast.Assign(
                    targets=[ast.Attribute(
                        value=ast.Name(id=init_self, ctx=ast.Load()),
                        attr=a.arg, ctx=ast.Store(),
                    )],
                    value=ast.Name(id=a.arg, ctx=ast.Load()),
                )
            )
    if args.vararg is not None:
        init_body.append(
            ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id=init_self, ctx=ast.Load()),
                    attr=args.vararg.arg, ctx=ast.Store(),
                )],
                value=ast.Name(id=args.vararg.arg, ctx=ast.Load()),
            )
        )
    if args.kwarg is not None:
        init_body.append(
            ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id=init_self, ctx=ast.Load()),
                    attr=args.kwarg.arg, ctx=ast.Store(),
                )],
                value=ast.Name(id=args.kwarg.arg, ctx=ast.Load()),
            )
        )
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id=init_self, ctx=ast.Load()),
                attr='state', ctx=ast.Store(),
            )],
            value=ast.Constant(value=0),
        )
    )
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id=init_self, ctx=ast.Load()),
                attr='_sent', ctx=ast.Store(),
            )],
            value=ast.Constant(value=None),
        )
    )
    # _yfrom: the inner iterator that's currently being driven by a
    # `yield from`, or None when no yield-from is active. send/throw/
    # close consult this so they can forward to the inner iterator
    # per PEP 380.
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id=init_self, ctx=ast.Load()),
                attr='_yfrom', ctx=ast.Store(),
            )],
            value=ast.Constant(value=None),
        )
    )
    # _stopping_via_return: True iff the StopIteration that's about
    # to escape send() came from `return` or fall-off-end (PEP 479).
    # User-level `raise StopIteration(...)` leaves this False so the
    # send wrapper can convert it to RuntimeError.
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id=init_self, ctx=ast.Load()),
                attr='_stopping_via_return', ctx=ast.Store(),
            )],
            value=ast.Constant(value=False),
        )
    )

    init_args = ast.arguments(
        posonlyargs=list(args.posonlyargs),
        args=[ast.arg(arg=init_self, annotation=None)] + list(args.args),
        vararg=args.vararg,
        kwonlyargs=list(args.kwonlyargs),
        kw_defaults=list(args.kw_defaults),
        defaults=list(args.defaults),
        kwarg=args.kwarg,
    )

    init_def = ast.FunctionDef(
        name='__init__',
        args=init_args,
        body=init_body,
        decorator_list=[],
        returns=None,
        type_params=[],
    )

    # __iter__: return self
    iter_def = ast.FunctionDef(
        name='__iter__',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self', annotation=None)],
            vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=[ast.Return(value=_self_name(ast.Load()))],
        decorator_list=[],
        returns=None,
        type_params=[],
    )

    # __next__: drive one step
    next_def = ast.FunctionDef(
        name='__next__',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self', annotation=None)],
            vararg=None,
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=[
            # self._sent = None; return self.send(None)
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_sent', ctx=ast.Store(),
                )],
                value=ast.Constant(value=None),
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='send', ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=None)],
                    keywords=[],
                )
            ),
        ],
        decorator_list=[],
        returns=None,
        type_params=[],
    )

    # send: the actual state machine. while True: dispatch by self.state.
    # Each block becomes a branch; terminators write self.state and
    # either continue (loop again) or return (yield) or raise.
    send_def = _emit_send(blocks, gen_self_alias=gen_self_alias)
    throw_def = _emit_throw(blocks)
    close_def = _emit_close()

    cls = ast.ClassDef(
        name='_Gen_' + name,
        bases=[],
        keywords=[],
        body=[init_def, iter_def, next_def, send_def, throw_def, close_def],
        decorator_list=[],
        type_params=[],
    )

    # The user's `def name(...)` becomes:
    #   _Gen_name = <class>
    #   name = lambda *a, **kw: _Gen_name(*a, **kw)  (with decorators)
    # Decorators are applied to the lambda, not the class.
    #
    # When the original user function was `async def`, the forwarder
    # is a generator-yielding-from-the-state-machine wrapped in
    # types.coroutine, which makes it awaitable. types.coroutine
    # requires a generator function — a lambda containing `yield from`
    # qualifies.
    forwarder_args = _clone_args_for_forwarder(args)
    forwarder_call = ast.Call(
        func=ast.Name(id=cls.name, ctx=ast.Load()),
        args=_args_as_call_args(args),
        keywords=_args_as_call_kwargs(args),
    )
    if async_kind is None and is_async:
        async_kind = 'coro'
    if async_kind == 'coro':
        gen_lambda = ast.Lambda(
            args=forwarder_args,
            body=ast.YieldFrom(value=forwarder_call),
        )
        # types.coroutine(gen_lambda)
        forwarder = ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id='__import__', ctx=ast.Load()),
                    args=[ast.Constant(value='types')],
                    keywords=[],
                ),
                attr='coroutine', ctx=ast.Load(),
            ),
            args=[gen_lambda],
            keywords=[],
        )
    elif async_kind == 'gen':
        # Wrap _Gen_name(*a, **kw) in _AsyncGenWrapper so the user's
        # call to name(*a, **kw) returns an async iterable.
        forwarder = ast.Lambda(
            args=forwarder_args,
            body=ast.Call(
                func=ast.Name(id='_AsyncGenWrapper', ctx=ast.Load()),
                args=[forwarder_call],
                keywords=[],
            ),
        )
    else:
        # Plain generator forwarder: `yield from` makes the lambda
        # itself a generator function, so inspect.isgeneratorfunction()
        # returns True and the user can `yield from gen()` straight
        # from another generator without an extra iter() call.
        forwarder = ast.Lambda(
            args=forwarder_args,
            body=ast.YieldFrom(value=forwarder_call),
        )

    # We build the forwarder, then set __name__/__qualname__/
    # __annotations__/_is_coroutine_marker on it BEFORE applying user
    # decorators. Decorators (e.g. FastAPI's @app.get) read these
    # attributes at decoration time, so getting them on the inner
    # function is what makes introspection-driven frameworks work.
    # Bind through a temp name to keep the AST a sequence of stmts.
    raw_name = '__onexpr_raw_' + name

    bind_raw = ast.Assign(
        targets=[ast.Name(id=raw_name, ctx=ast.Store())],
        value=forwarder,
    )

    set_name_stmts = []
    for attr in ('__name__', '__qualname__'):
        set_name_stmts.append(
            ast.If(
                test=ast.Call(
                    func=ast.Name(id='hasattr', ctx=ast.Load()),
                    args=[
                        ast.Name(id=raw_name, ctx=ast.Load()),
                        ast.Constant(value=attr),
                    ],
                    keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id=raw_name, ctx=ast.Load()),
                                attr=attr, ctx=ast.Store(),
                            ),
                        ],
                        value=ast.Constant(value=name),
                    ),
                ],
                orelse=[],
            )
        )

    # For coroutines, expose the inspect.iscoroutinefunction marker so
    # `inspect.iscoroutinefunction(name)` returns True. The lambda's
    # __code__ doesn't carry CO_COROUTINE (we synthesize it from a
    # `yield from` lambda + types.coroutine), so we use the
    # documented escape hatch: set _is_coroutine_marker to
    # inspect._is_coroutine_mark.
    if async_kind == 'gen':
        # Async generator function. inspect.isasyncgenfunction queries
        # CO_ASYNC_GENERATOR (0x200), which a lambda can't acquire.
        # The runtime monkey-patch on inspect._has_code_flag (injected
        # via inspect_patch.py when the tree contains async generators)
        # consults _onexpr_code_flags first, so advertising 0x200 here
        # makes inspect.isasyncgenfunction(forwarder) return True.
        set_name_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=raw_name, ctx=ast.Load()),
                        attr='_onexpr_code_flags',
                        ctx=ast.Store(),
                    ),
                ],
                # 0x200 == inspect.CO_ASYNC_GENERATOR
                value=ast.Constant(value=0x200),
            )
        )
    elif async_kind == 'coro':
        set_name_stmts.append(
            ast.If(
                test=ast.Call(
                    func=ast.Name(id='hasattr', ctx=ast.Load()),
                    args=[
                        ast.Name(id=raw_name, ctx=ast.Load()),
                        ast.Constant(value='_is_coroutine_marker'),
                    ],
                    keywords=[],
                ),
                body=[ast.Pass()],
                orelse=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id=raw_name, ctx=ast.Load()),
                                attr='_is_coroutine_marker',
                                ctx=ast.Store(),
                            ),
                        ],
                        value=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id='__import__', ctx=ast.Load()),
                                args=[ast.Constant(value='inspect')],
                                keywords=[],
                            ),
                            attr='_is_coroutine_mark',
                            ctx=ast.Load(),
                        ),
                    ),
                ],
            )
        )

    # Reconstruct forwarder.__annotations__ from the user's parameter
    # annotations + return annotation so introspection (typing.get_type_hints,
    # FastAPI's signature parsing, dataclasses, pydantic) sees them.
    # Crucial: the assignment must run BEFORE decorators apply, because
    # frameworks like FastAPI inspect annotations at decoration time.
    ann_keys = []
    ann_values = []
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            if a.annotation is not None:
                ann_keys.append(ast.Constant(value=a.arg))
                ann_values.append(a.annotation)
    if args.vararg is not None and args.vararg.annotation is not None:
        ann_keys.append(ast.Constant(value=args.vararg.arg))
        ann_values.append(args.vararg.annotation)
    if args.kwarg is not None and args.kwarg.annotation is not None:
        ann_keys.append(ast.Constant(value=args.kwarg.arg))
        ann_values.append(args.kwarg.annotation)
    if returns is not None:
        ann_keys.append(ast.Constant(value='return'))
        ann_values.append(returns)
    if ann_keys:
        set_name_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=raw_name, ctx=ast.Load()),
                        attr='__annotations__', ctx=ast.Store(),
                    ),
                ],
                value=ast.Dict(keys=ann_keys, values=ann_values),
            )
        )

    # Apply decorators (outside-in, same order as Python's `@deco`)
    # AFTER all introspection-visible attributes are set.
    decorated = ast.Name(id=raw_name, ctx=ast.Load())
    for d in reversed(decorator_list):
        decorated = ast.Call(func=d, args=[decorated], keywords=[])

    bind = ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=decorated,
    )

    return [cls, bind_raw] + set_name_stmts + [bind]


def _clone_args_for_forwarder(args: ast.arguments) -> ast.arguments:
    """The forwarder lambda needs the same parameters as the user
    function (so calls flow through). Annotations are stripped
    because lambda doesn't accept them."""
    return ast.arguments(
        posonlyargs=[ast.arg(arg=a.arg, annotation=None) for a in args.posonlyargs],
        args=[ast.arg(arg=a.arg, annotation=None) for a in args.args],
        vararg=ast.arg(arg=args.vararg.arg, annotation=None) if args.vararg else None,
        kwonlyargs=[ast.arg(arg=a.arg, annotation=None) for a in args.kwonlyargs],
        kw_defaults=list(args.kw_defaults),
        kwarg=ast.arg(arg=args.kwarg.arg, annotation=None) if args.kwarg else None,
        defaults=list(args.defaults),
    )


def _args_as_call_args(args: ast.arguments) -> list:
    """Forward the lambda's positional parameters to the class call."""
    out = []
    for a in args.posonlyargs:
        out.append(ast.Name(id=a.arg, ctx=ast.Load()))
    for a in args.args:
        out.append(ast.Name(id=a.arg, ctx=ast.Load()))
    if args.vararg is not None:
        out.append(ast.Starred(
            value=ast.Name(id=args.vararg.arg, ctx=ast.Load()),
            ctx=ast.Load(),
        ))
    return out


def _args_as_call_kwargs(args: ast.arguments) -> list:
    """Forward keyword-only and **kwargs to the class call."""
    out = []
    for a in args.kwonlyargs:
        out.append(ast.keyword(
            arg=a.arg,
            value=ast.Name(id=a.arg, ctx=ast.Load()),
        ))
    if args.kwarg is not None:
        out.append(ast.keyword(
            arg=None,
            value=ast.Name(id=args.kwarg.arg, ctx=ast.Load()),
        ))
    return out


def _emit_close() -> ast.FunctionDef:
    """`close()` is throw(GeneratorExit). Per PEP 342:
       - If body re-raises GeneratorExit / StopIteration: silently OK.
       - If body yields a value: raise RuntimeError.
       - If body raises something else: that propagates out.

    PEP 380: if we're suspended on `yield from inner`, close the inner
    iterator first (via its own close() if it has one), then raise
    GeneratorExit on ourselves so any surrounding finally clauses in
    *our* body run too.
    """
    body = [
        # If suspended on yield from, close the inner iterator first.
        ast.If(
            test=ast.Compare(
                left=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_yfrom', ctx=ast.Load(),
                ),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                ast.Try(
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id='_m', ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id='getattr', ctx=ast.Load()),
                                args=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='_yfrom', ctx=ast.Load(),
                                    ),
                                    ast.Constant(value='close'),
                                    ast.Constant(value=None),
                                ],
                                keywords=[],
                            ),
                        ),
                        ast.If(
                            test=ast.Compare(
                                left=ast.Name(id='_m', ctx=ast.Load()),
                                ops=[ast.IsNot()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            body=[
                                ast.Expr(value=ast.Call(
                                    func=ast.Name(id='_m', ctx=ast.Load()),
                                    args=[], keywords=[],
                                )),
                            ],
                            orelse=[],
                        ),
                    ],
                    handlers=[
                        ast.ExceptHandler(
                            type=ast.Name(id='Exception', ctx=ast.Load()),
                            name=None,
                            body=[ast.Pass()],
                        ),
                    ],
                    orelse=[],
                    finalbody=[],
                ),
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='_yfrom', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=None),
                ),
                # Move state to the post-yield-from continuation so
                # the GeneratorExit we throw next isn't re-routed back
                # into the (now closed) inner.
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='_yfrom_next', ctx=ast.Load(),
                    ),
                ),
            ],
            orelse=[],
        ),
        ast.Try(
            body=[
                ast.Assign(
                    targets=[ast.Name(id='_v', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='throw', ctx=ast.Load(),
                        ),
                        args=[ast.Name(id='GeneratorExit', ctx=ast.Load())],
                        keywords=[],
                    ),
                ),
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                        args=[ast.Constant(value='generator ignored GeneratorExit')],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Tuple(
                        elts=[
                            ast.Name(id='GeneratorExit', ctx=ast.Load()),
                            ast.Name(id='StopIteration', ctx=ast.Load()),
                        ],
                        ctx=ast.Load(),
                    ),
                    name=None,
                    body=[ast.Pass()],
                ),
            ],
            orelse=[],
            finalbody=[],
        ),
    ]
    return ast.FunctionDef(
        name='close',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self', annotation=None)],
            vararg=None, kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_params=[],
    )


def _emit_throw(blocks: list) -> ast.FunctionDef:
    """`throw(exc)` injects an exception at the current state — same as
    if the user's body raised it from inside the active block. We
    instantiate exc if it was passed as a class, then drive the same
    handler-lookup logic as the dispatch loop's except wrapper. If no
    enclosing try matches, the exception escapes out of throw(), which
    matches the standard generator semantics."""
    eh_keys = []
    eh_values = []
    for blk in blocks:
        if blk.exc_handler is not None:
            eh_keys.append(ast.Constant(value=blk.id))
            eh_values.append(ast.Constant(value=blk.exc_handler))

    handler_lookup = ast.Call(
        func=ast.Attribute(
            value=ast.Dict(keys=eh_keys, values=eh_values),
            attr='get', ctx=ast.Load(),
        ),
        args=[
            ast.Attribute(
                value=_self_name(ast.Load()),
                attr='state', ctx=ast.Load(),
            )
        ],
        keywords=[],
    )

    body = [
        # Resolve (typ, val) into a single exception instance, matching
        # the standard generator throw protocol:
        #   throw(ExcType)              → ExcType()
        #   throw(ExcType, val)         → ExcType(val) if val isn't already
        #                                 an instance of ExcType
        #   throw(instance)             → instance
        # tb is honoured if provided; otherwise we leave the traceback
        # alone and let Python compose one at the raise site.
        ast.If(
            test=ast.Call(
                func=ast.Name(id='isinstance', ctx=ast.Load()),
                args=[
                    ast.Name(id='typ', ctx=ast.Load()),
                    ast.Name(id='type', ctx=ast.Load()),
                ],
                keywords=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id='exc', ctx=ast.Store())],
                    value=ast.IfExp(
                        test=ast.Compare(
                            left=ast.Name(id='val', ctx=ast.Load()),
                            ops=[ast.Is()],
                            comparators=[ast.Constant(value=None)],
                        ),
                        body=ast.Call(
                            func=ast.Name(id='typ', ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        orelse=ast.IfExp(
                            test=ast.Call(
                                func=ast.Name(id='isinstance', ctx=ast.Load()),
                                args=[
                                    ast.Name(id='val', ctx=ast.Load()),
                                    ast.Name(id='typ', ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                            body=ast.Name(id='val', ctx=ast.Load()),
                            orelse=ast.Call(
                                func=ast.Name(id='typ', ctx=ast.Load()),
                                args=[ast.Name(id='val', ctx=ast.Load())],
                                keywords=[],
                            ),
                        ),
                    ),
                ),
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Name(id='exc', ctx=ast.Store())],
                    value=ast.Name(id='typ', ctx=ast.Load()),
                ),
            ],
        ),
        # if tb is not None: exc = exc.with_traceback(tb)
        ast.If(
            test=ast.Compare(
                left=ast.Name(id='tb', ctx=ast.Load()),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id='exc', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='exc', ctx=ast.Load()),
                            attr='with_traceback',
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id='tb', ctx=ast.Load())],
                        keywords=[],
                    ),
                ),
            ],
            orelse=[],
        ),
        # if self._yfrom is not None: forward the exception to the
        # inner iterator (PEP 380). If inner has a .throw, call it;
        # otherwise close inner and re-raise here. inner.throw can
        # return a yielded value (we propagate that), raise
        # StopIteration (we capture .value into <iter>_value — but we
        # don't track which iter_var is active across throw, so the
        # caller's send() picks up the next state on the subsequent
        # call), or raise something else (escapes throw).
        ast.If(
            test=ast.Compare(
                left=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_yfrom', ctx=ast.Load(),
                ),
                ops=[ast.IsNot()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                ast.Try(
                    body=[
                        # _m = getattr(self._yfrom, 'throw', None)
                        ast.Assign(
                            targets=[ast.Name(id='_m', ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id='getattr', ctx=ast.Load()),
                                args=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='_yfrom', ctx=ast.Load(),
                                    ),
                                    ast.Constant(value='throw'),
                                    ast.Constant(value=None),
                                ],
                                keywords=[],
                            ),
                        ),
                        # if _m is None: raise exc (after closing inner)
                        ast.If(
                            test=ast.Compare(
                                left=ast.Name(id='_m', ctx=ast.Load()),
                                ops=[ast.Is()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            body=[
                                ast.Try(
                                    body=[
                                        ast.Expr(value=ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Attribute(
                                                    value=_self_name(ast.Load()),
                                                    attr='_yfrom', ctx=ast.Load(),
                                                ),
                                                attr='close', ctx=ast.Load(),
                                            ),
                                            args=[], keywords=[],
                                        )),
                                    ],
                                    handlers=[
                                        ast.ExceptHandler(
                                            type=ast.Name(id='AttributeError', ctx=ast.Load()),
                                            name=None,
                                            body=[ast.Pass()],
                                        ),
                                    ],
                                    orelse=[],
                                    finalbody=[],
                                ),
                                ast.Assign(
                                    targets=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr='_yfrom', ctx=ast.Store(),
                                        )
                                    ],
                                    value=ast.Constant(value=None),
                                ),
                                ast.Raise(
                                    exc=ast.Name(id='exc', ctx=ast.Load()),
                                    cause=None,
                                ),
                            ],
                            orelse=[
                                # Resolve typ/val/tb into a single
                                # exception instance, then call
                                # _m(exc) (single-arg form — the
                                # 3-arg signature is deprecated in
                                # 3.12 and removed in future).
                                ast.Assign(
                                    targets=[ast.Name(id='_v', ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='_m', ctx=ast.Load()),
                                        args=[ast.Name(id='exc', ctx=ast.Load())],
                                        keywords=[],
                                    ),
                                ),
                                # return _v — yielded by inner, surface to caller
                                ast.Return(value=ast.Name(id='_v', ctx=ast.Load())),
                            ],
                        ),
                    ],
                    handlers=[
                        ast.ExceptHandler(
                            type=ast.Name(id='StopIteration', ctx=ast.Load()),
                            name='_si',
                            body=[
                                # Inner finished — capture its return
                                # value, clear _yfrom, advance state to
                                # the post-yield-from continuation, and
                                # drive the state machine forward.
                                ast.Assign(
                                    targets=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr='_yfrom_value',
                                            ctx=ast.Store(),
                                        )
                                    ],
                                    value=ast.Attribute(
                                        value=ast.Name(id='_si', ctx=ast.Load()),
                                        attr='value', ctx=ast.Load(),
                                    ),
                                ),
                                ast.Assign(
                                    targets=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr='_yfrom', ctx=ast.Store(),
                                        )
                                    ],
                                    value=ast.Constant(value=None),
                                ),
                                ast.Assign(
                                    targets=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr='state', ctx=ast.Store(),
                                        )
                                    ],
                                    value=ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='_yfrom_next', ctx=ast.Load(),
                                    ),
                                ),
                                ast.Return(value=ast.Call(
                                    func=ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='send', ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(value=None)],
                                    keywords=[],
                                )),
                            ],
                        ),
                        ast.ExceptHandler(
                            # Anything else inner.throw raised: clear
                            # _yfrom, then re-inject the new exception
                            # at the current yield-from state via
                            # self.throw, so the user's surrounding
                            # try/except can catch it. (CPython's
                            # `yield from` re-raises whatever
                            # inner.throw raised at the yield-from site.)
                            type=ast.Name(id='BaseException', ctx=ast.Load()),
                            name='_eb',
                            body=[
                                ast.Assign(
                                    targets=[
                                        ast.Attribute(
                                            value=_self_name(ast.Load()),
                                            attr='_yfrom', ctx=ast.Store(),
                                        )
                                    ],
                                    value=ast.Constant(value=None),
                                ),
                                ast.Return(value=ast.Call(
                                    func=ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='throw', ctx=ast.Load(),
                                    ),
                                    args=[ast.Name(id='_eb', ctx=ast.Load())],
                                    keywords=[],
                                )),
                            ],
                        ),
                    ],
                    orelse=[],
                    finalbody=[],
                ),
            ],
            orelse=[],
        ),
        # self._exc = exc
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_exc', ctx=ast.Store(),
                )
            ],
            value=ast.Name(id='exc', ctx=ast.Load()),
        ),
        # _h = _EH.get(self.state)
        ast.Assign(
            targets=[ast.Name(id='_h', ctx=ast.Store())],
            value=handler_lookup,
        ),
        # if _h is None: raise exc
        ast.If(
            test=ast.Compare(
                left=ast.Name(id='_h', ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                ast.Raise(
                    exc=ast.Name(id='exc', ctx=ast.Load()),
                    cause=None,
                ),
            ],
            orelse=[],
        ),
        # self.state = _h
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='state', ctx=ast.Store(),
                )
            ],
            value=ast.Name(id='_h', ctx=ast.Load()),
        ),
        # return self.send(None)
        ast.Return(
            value=ast.Call(
                func=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='send', ctx=ast.Load(),
                ),
                args=[ast.Constant(value=None)],
                keywords=[],
            ),
        ),
    ]
    return ast.FunctionDef(
        name='throw',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg='self', annotation=None),
                ast.arg(arg='typ', annotation=None),
                ast.arg(arg='val', annotation=None),
                ast.arg(arg='tb', annotation=None),
            ],
            vararg=None, kwonlyargs=[],
            kw_defaults=[],
            defaults=[ast.Constant(value=None), ast.Constant(value=None)],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_params=[],
    )


def _emit_send(blocks: list, gen_self_alias: str = None) -> ast.FunctionDef:
    """The state-machine dispatch. Big while True: if/elif chain by
    self.state, wrapped in a try/except that consults a per-state
    handler map for routing exceptions to user except clauses."""
    # Build the if/elif chain.
    chain_body: list = []
    cur_orelse: list = []
    for blk in reversed(blocks):
        block_body = list(blk.stmts) + _emit_terminator(blk.terminator, blocks)
        cur = ast.If(
            test=ast.Compare(
                left=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='state', ctx=ast.Load(),
                ),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=blk.id)],
            ),
            body=block_body,
            orelse=cur_orelse,
        )
        cur_orelse = [cur]
    chain_body = cur_orelse

    while_inner = chain_body + [
        # If we somehow drop out of all branches, that's a bug. Raise.
        ast.Raise(
            exc=ast.Call(
                func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                args=[ast.Constant(value='gen state machine fell through')],
                keywords=[],
            ),
            cause=None,
        ),
    ]

    # Build _EH: dict mapping each block-id with an active handler to
    # the dispatcher block-id. Embedded as a literal so the runtime
    # can do a single dict lookup per caught exception.
    eh_keys = []
    eh_values = []
    for blk in blocks:
        if blk.exc_handler is not None:
            eh_keys.append(ast.Constant(value=blk.id))
            eh_values.append(ast.Constant(value=blk.exc_handler))

    # On caught exception: stash on self._exc, look up the dispatcher
    # for the current state, set state, continue. If no handler — re-
    # raise out of send (propagates to the user's next() / send()).
    handler_lookup = ast.Call(
        func=ast.Attribute(
            value=ast.Dict(keys=eh_keys, values=eh_values),
            attr='get', ctx=ast.Load(),
        ),
        args=[
            ast.Attribute(
                value=_self_name(ast.Load()),
                attr='state', ctx=ast.Load(),
            )
        ],
        keywords=[],
    )
    except_body = [
        # PEP 479: if a StopIteration escapes from inside the user
        # body (not from one of our injected return / fall-off-end
        # paths), convert it to RuntimeError. CPython does this in
        # the generator frame; we approximate with a flag set
        # immediately before the synthetic StopIteration we emit
        # for `return`.
        ast.If(
            test=ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Call(
                        func=ast.Name(id='isinstance', ctx=ast.Load()),
                        args=[
                            ast.Name(id='_e', ctx=ast.Load()),
                            ast.Name(id='StopIteration', ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                    ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='_stopping_via_return', ctx=ast.Load(),
                        ),
                    ),
                ],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                        args=[
                            ast.Constant(
                                value='generator raised StopIteration'
                            ),
                        ],
                        keywords=[],
                    ),
                    cause=ast.Name(id='_e', ctx=ast.Load()),
                ),
            ],
            orelse=[],
        ),
        # self._exc = e
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_exc', ctx=ast.Store(),
                )
            ],
            value=ast.Name(id='_e', ctx=ast.Load()),
        ),
        # _h = handler_lookup
        ast.Assign(
            targets=[ast.Name(id='_h', ctx=ast.Store())],
            value=handler_lookup,
        ),
        # if _h is None: raise
        ast.If(
            test=ast.Compare(
                left=ast.Name(id='_h', ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[ast.Raise(exc=None, cause=None)],
            orelse=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )
                    ],
                    value=ast.Name(id='_h', ctx=ast.Load()),
                ),
            ],
        ),
    ]

    if eh_keys:
        while_body = [
            ast.Try(
                body=while_inner,
                handlers=[
                    ast.ExceptHandler(
                        # BaseException, not Exception, so we also catch
                        # GeneratorExit (raised by close()) and any other
                        # BaseException-only subclass. The handler-lookup
                        # logic still re-raises out of send if no user
                        # try matches, so those keep their original
                        # propagation semantics.
                        type=ast.Name(id='BaseException', ctx=ast.Load()),
                        name='_e',
                        body=except_body,
                    ),
                ],
                orelse=[],
                finalbody=[],
            )
        ]
    else:
        # No user try in body — but we still need to convert a
        # user-level `raise StopIteration` into RuntimeError per
        # PEP 479. Wrap with a narrow try that only catches
        # StopIteration; everything else (including GeneratorExit
        # from close()) propagates out of send naturally.
        pep479_handler = [
            ast.If(
                test=ast.UnaryOp(
                    op=ast.Not(),
                    operand=ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='_stopping_via_return', ctx=ast.Load(),
                    ),
                ),
                body=[
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                            args=[ast.Constant(value='generator raised StopIteration')],
                            keywords=[],
                        ),
                        cause=ast.Name(id='_e', ctx=ast.Load()),
                    ),
                ],
                orelse=[ast.Raise(exc=None, cause=None)],
            ),
        ]
        while_body = [
            ast.Try(
                body=while_inner,
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='StopIteration', ctx=ast.Load()),
                        name='_e',
                        body=pep479_handler,
                    ),
                ],
                orelse=[],
                finalbody=[],
            )
        ]

    body = []
    if gen_self_alias is not None:
        # Expose `self` under a stable alias name so any nested
        # generator/coroutine class defined inside this send body
        # can capture us via closure and resolve `nonlocal x` to
        # <alias>.x.
        body.append(
            ast.Assign(
                targets=[ast.Name(id=gen_self_alias, ctx=ast.Store())],
                value=_self_name(ast.Load()),
            )
        )
    body += [
        # self._sent = sent
        ast.Assign(
            targets=[ast.Attribute(
                value=_self_name(ast.Load()),
                attr='_sent', ctx=ast.Store(),
            )],
            value=ast.Name(id='sent', ctx=ast.Load()),
        ),
        ast.While(
            test=ast.Constant(value=True),
            body=while_body,
            orelse=[],
        ),
    ]

    return ast.FunctionDef(
        name='send',
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg='self', annotation=None),
                ast.arg(arg='sent', annotation=None),
            ],
            vararg=None, kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_params=[],
    )


def _emit_terminator(t, blocks) -> list:
    """Emit the statements that close out a block based on its
    terminator. These get spliced into the if/elif body."""
    if isinstance(t, TGoto):
        return [
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='state', ctx=ast.Store(),
                )],
                value=ast.Constant(value=t.target),
            ),
            ast.Continue(),
        ]
    if isinstance(t, TBranch):
        return [
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='state', ctx=ast.Store(),
                )],
                value=ast.IfExp(
                    test=t.test,
                    body=ast.Constant(value=t.true),
                    orelse=ast.Constant(value=t.false),
                ),
            ),
            ast.Continue(),
        ]
    if isinstance(t, TYield):
        return [
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='state', ctx=ast.Store(),
                )],
                value=ast.Constant(value=t.next),
            ),
            ast.Return(value=t.value),
        ]
    if isinstance(t, TYieldFrom):
        # Drive self.<iter_var> with next() (or send(self._sent) if we
        # have a value to inject — PEP 380 says `yield from` forwards
        # the most recent send into the inner). We capture
        # StopIteration to preserve the sub-generator's return value
        # for the post-yield-from binding. self._yfrom is set so
        # throw()/close() can forward into the inner iterator if a
        # caller injects an exception while we're suspended here.
        # self._yfrom_next records t.next so throw() can route to the
        # post-yield-from state directly when it consumes inner.
        sent = ast.Attribute(
            value=_self_name(ast.Load()),
            attr='_sent', ctx=ast.Load(),
        )
        return [
            # self._yfrom = self.<iter_var>  — mark active for throw()/close()
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='_yfrom', ctx=ast.Store(),
                    )
                ],
                value=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr=t.iter_var, ctx=ast.Load(),
                ),
            ),
            # self._yfrom_next = t.next
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=_self_name(ast.Load()),
                        attr='_yfrom_next', ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=t.next),
            ),
            ast.Try(
                body=[
                    # _v = (next(_i) if _sent is None else _i.send(_sent))
                    ast.Assign(
                        targets=[ast.Name(id='_v', ctx=ast.Store())],
                        value=ast.IfExp(
                            test=ast.Compare(
                                left=sent,
                                ops=[ast.Is()],
                                comparators=[ast.Constant(value=None)],
                            ),
                            body=ast.Call(
                                func=ast.Name(id='next', ctx=ast.Load()),
                                args=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr=t.iter_var, ctx=ast.Load(),
                                    )
                                ],
                                keywords=[],
                            ),
                            orelse=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr=t.iter_var, ctx=ast.Load(),
                                    ),
                                    attr='send', ctx=ast.Load(),
                                ),
                                args=[sent],
                                keywords=[],
                            ),
                        ),
                    ),
                    # self._sent = None — we've consumed the sent value
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=_self_name(ast.Load()),
                                attr='_sent', ctx=ast.Store(),
                            )
                        ],
                        value=ast.Constant(value=None),
                    ),
                    ast.Return(value=ast.Name(id='_v', ctx=ast.Load())),
                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='StopIteration', ctx=ast.Load()),
                        name='_si',
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='_yfrom_value',
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Attribute(
                                    value=ast.Name(id='_si', ctx=ast.Load()),
                                    attr='value', ctx=ast.Load(),
                                ),
                            ),
                            # self._yfrom = None — yield from done
                            ast.Assign(
                                targets=[
                                    ast.Attribute(
                                        value=_self_name(ast.Load()),
                                        attr='_yfrom', ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Constant(value=None),
                            ),
                            ast.Assign(
                                targets=[ast.Attribute(
                                    value=_self_name(ast.Load()),
                                    attr='state', ctx=ast.Store(),
                                )],
                                value=ast.Constant(value=t.next),
                            ),
                            ast.Continue(),
                        ],
                    )
                ],
                orelse=[],
                finalbody=[],
            ),
        ]
    if isinstance(t, TForIter):
        return [
            ast.Assign(
                targets=[ast.Name(id='_v', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='next', ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr=t.iter_var, ctx=ast.Load(),
                        ),
                        ast.Name(id=_GEN_DONE, ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id='_v', ctx=ast.Load()),
                    ops=[ast.Is()],
                    comparators=[ast.Name(id=_GEN_DONE, ctx=ast.Load())],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )],
                        value=ast.Constant(value=t.after),
                    ),
                    ast.Continue(),
                ],
                orelse=[
                    ast.Assign(
                        targets=[ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr=t.target_name, ctx=ast.Store(),
                        )],
                        value=ast.Name(id='_v', ctx=ast.Load()),
                    ),
                    ast.Assign(
                        targets=[ast.Attribute(
                            value=_self_name(ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )],
                        value=ast.Constant(value=t.body),
                    ),
                    ast.Continue(),
                ],
            ),
        ]
    if isinstance(t, TReturn):
        return [
            # Mark this StopIteration as "from return / fall-off-end"
            # so PEP 479 doesn't turn it into RuntimeError.
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_stopping_via_return', ctx=ast.Store(),
                )],
                value=ast.Constant(value=True),
            ),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id='StopIteration', ctx=ast.Load()),
                    args=[t.value],
                    keywords=[],
                ),
                cause=None,
            ),
        ]
    if isinstance(t, TEnd):
        return [
            ast.Assign(
                targets=[ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_stopping_via_return', ctx=ast.Store(),
                )],
                value=ast.Constant(value=True),
            ),
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id='StopIteration', ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                cause=None,
            ),
        ]
    if isinstance(t, TReraise):
        # Bare `raise` reraises self._exc, which the dispatcher set
        # before transferring to this block.
        return [
            ast.Raise(
                exc=ast.Attribute(
                    value=_self_name(ast.Load()),
                    attr='_exc', ctx=ast.Load(),
                ),
                cause=None,
            ),
        ]
    if isinstance(t, TUnreachable):
        # Defensive — every legitimate path through this block has
        # already exited via an explicit `continue` in the stmts.
        return [
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                    args=[ast.Constant(value='gen unreachable terminator')],
                    keywords=[],
                ),
                cause=None,
            ),
        ]
    raise NotImplementedError(f"terminator: {type(t).__name__}")


# ---------------------------------------------------------------------
# Top-level entrypoint, called from parse_function_def

def compile_generator(stmt, frame, is_async=False, async_kind=None) -> list:
    """Replace `def gen(args): BODY` (with yields in BODY) with the
    state-machine class + a forwarder lambda binding the user's name.

    Returned statements are spliced into the enclosing scope and
    processed by parse_stmts. parse_class_def will compile the class
    via the regular onexpr path; the methods inside are plain
    synchronous Python and don't trip on yields.

    `async_kind` toggles forwarder shape:
    - None (default) / 'sync': plain generator forwarder (no wrapping)
    - 'coro': `async def` without yield. Forwarder is wrapped in
      types.coroutine so the result is awaitable.
    - 'gen': `async def` with yield (PEP 525 async generator).
      Forwarder returns _AsyncGenWrapper(_Gen_name(...)) so `async
      for` works.

    `is_async` is the legacy boolean for backward compatibility; when
    set without async_kind, it means 'coro'."""
    if async_kind is None and is_async:
        async_kind = 'coro'

    name_provider = frame.get_temp_var

    # 1. Pre-collect user locals on the original body so ANF knows what
    #    to dehydrate before nested def/class.
    prelim_locals = collect_user_locals(stmt.body, stmt.args)

    # 2. ANF-lift any yields embedded in sub-expressions, plus flatten
    #    Match into if-chain, plus stage closure dehydration before
    #    nested def/class.
    body = anf_transform(stmt.body, name_provider, all_locals=prelim_locals)

    # 3. Re-discover locals on the rewritten body (ANF may have
    #    introduced new temp names that also need boxing).
    boxed = collect_user_locals(body, stmt.args)

    # 3. Build the CFG.
    blocks = build_cfg(body, name_provider)

    # 4. Rewrite Name references to self.<name> for boxed names.
    rewrite_block_to_self(blocks, boxed)

    # 5. Emit the class + forwarder.
    return emit_state_machine(
        name=stmt.name,
        args=stmt.args,
        blocks=blocks,
        boxed=boxed,
        decorator_list=stmt.decorator_list,
        async_kind=async_kind,
        gen_self_alias=getattr(stmt, '_gen_self_alias', None),
        returns=stmt.returns,
    )
