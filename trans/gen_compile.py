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
class Block:
    id: int
    stmts: list           # body stmts (no control flow, no yield)
    terminator: object    # one of T*

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


def anf_transform(body: list, name_provider) -> list:
    """Walk a generator body, returning a new list of statements where
    every Yield / YieldFrom appears either as a top-level expression
    statement or as the entire RHS of a simple-name assignment.

    Recurses into If / For / While bodies but stops at nested function
    / class / lambda — those have their own scopes."""
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
            # Flatten match → subj-Assign + if/elif chain so the state
            # machine sees only constructs it knows. compile_match's
            # output uses walrus operators inside the test expressions
            # for capture; those have short-circuit semantics that
            # would break if we lifted them, so we mark the produced
            # Ifs as match-origin and skip the lift step on them.
            from .match_patterns import compile_match
            fake_frame = _MiniFrame(name_provider)
            flat = compile_match(stmt, fake_frame)
            _mark_match_origin(flat)
            out.extend(anf_transform(flat, name_provider))
            continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested def/class: treat the bound name like any other
            # user local. The def's own body has its own scope and
            # goes through the regular onexpr path inside the state-
            # machine class, so we don't descend.
            #
            # Emit `<name> = <name>` with the RHS Name marked so the
            # state-machine self-rewriter doesn't rewrite it. After
            # rewrite the LHS becomes `self.<name>` while the RHS
            # stays as the send-local `<name>` produced by the def
            # itself. The binding survives across state transitions.
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
                stmt.body = anf_transform(stmt.body, name_provider)
                stmt.orelse = anf_transform(stmt.orelse, name_provider)
                out.append(stmt)
                continue
            stmt.body = anf_transform(stmt.body, name_provider)
            stmt.orelse = anf_transform(stmt.orelse, name_provider)
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.For):
            stmt.body = anf_transform(stmt.body, name_provider)
            stmt.orelse = anf_transform(stmt.orelse, name_provider)
            # The .iter expression itself can contain yield — lift it.
            out.extend(lift(stmt))
            continue
        if isinstance(stmt, ast.While):
            stmt.body = anf_transform(stmt.body, name_provider)
            stmt.orelse = anf_transform(stmt.orelse, name_provider)
            out.extend(lift(stmt))
            continue
        out.extend(lift(stmt))

    return out


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
    their own scope."""
    names = set()

    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            names.add(a.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)

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
            # Walrus targets are NOT boxed: the walrus stays as a
            # send-local, because Python doesn't allow walrus on
            # attribute targets (`(self.x := v)` is a SyntaxError).
            # Only descend into the value side so any nested
            # assignments / for targets in there still get collected.
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

    return names


# ---------------------------------------------------------------------
# CFG construction

class _CFGBuilder:
    def __init__(self, name_provider):
        self.blocks: list[Block] = []
        self.name = name_provider
        # Stack of (continue_target, break_target) for nested loops.
        self.loop_stack: list[tuple[int, int]] = []

    def new_block(self) -> Block:
        b = Block(id=len(self.blocks), stmts=[], terminator=None)
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
            current.terminator = TGoto(target=brk)
            return current
        if isinstance(stmt, ast.Continue):
            if not self.loop_stack:
                raise SyntaxError("'continue' outside loop")
            cont, _brk = self.loop_stack[-1]
            current.terminator = TGoto(target=cont)
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
                        value=ast.Name(id='self', ctx=ast.Load()),
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
                        value=ast.Name(id='self', ctx=ast.Load()),
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
            # Phase 1 simplification: bind to None.
            nxt.stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=capture_name, ctx=ast.Store())],
                    value=ast.Constant(value=None),
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
        if stmt.orelse:
            raise NotImplementedError("for-else not supported in generator")
        sub_var = self.name()
        # iter setup goes in current
        current.stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
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
        after = self.new_block()
        current.terminator = TGoto(target=head.id)
        head.terminator = TForIter(
            iter_var=sub_var,
            target_name=stmt.target.id,
            body=body_b.id,
            after=after.id,
        )
        self.loop_stack.append((head.id, after.id))
        end_body = self.emit(stmt.body, body_b)
        self.loop_stack.pop()
        if not end_body.is_terminated:
            end_body.terminator = TGoto(target=head.id)
        return after

    def _emit_while(self, stmt, current) -> Block:
        if stmt.orelse:
            raise NotImplementedError("while-else not supported in generator")
        head = self.new_block()
        body_b = self.new_block()
        after = self.new_block()
        current.terminator = TGoto(target=head.id)
        head.terminator = TBranch(
            test=stmt.test, true=body_b.id, false=after.id,
        )
        self.loop_stack.append((head.id, after.id))
        end_body = self.emit(stmt.body, body_b)
        self.loop_stack.pop()
        if not end_body.is_terminated:
            end_body.terminator = TGoto(target=head.id)
        return after


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
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=node.id,
                ctx=node.ctx,
            )
        return node

    def visit_NamedExpr(self, node):
        # Walrus: leave the target Name as a send-local (Python rejects
        # walrus into an attribute), only rewrite Names inside .value.
        node.value = self.visit(node.value)
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
) -> list:
    """Build a sequence of statements that defines the generator class
    and binds it to `name`. Returned statements are intended to be
    spliced into the enclosing scope and processed by parse_stmts /
    parse_class_def normally."""

    # __init__: setattr(self, '<arg>', <arg>) for every parameter,
    # then setattr(self, 'state', 0) and any housekeeping flags.
    init_body: list = []
    for group in (args.posonlyargs, args.args, args.kwonlyargs):
        for a in group:
            init_body.append(
                ast.Assign(
                    targets=[ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr=a.arg, ctx=ast.Store(),
                    )],
                    value=ast.Name(id=a.arg, ctx=ast.Load()),
                )
            )
    if args.vararg is not None:
        init_body.append(
            ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=args.vararg.arg, ctx=ast.Store(),
                )],
                value=ast.Name(id=args.vararg.arg, ctx=ast.Load()),
            )
        )
    if args.kwarg is not None:
        init_body.append(
            ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=args.kwarg.arg, ctx=ast.Store(),
                )],
                value=ast.Name(id=args.kwarg.arg, ctx=ast.Load()),
            )
        )
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr='state', ctx=ast.Store(),
            )],
            value=ast.Constant(value=0),
        )
    )
    init_body.append(
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr='_sent', ctx=ast.Store(),
            )],
            value=ast.Constant(value=None),
        )
    )

    init_args = ast.arguments(
        posonlyargs=list(args.posonlyargs),
        args=[ast.arg(arg='self', annotation=None)] + list(args.args),
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
        body=[ast.Return(value=ast.Name(id='self', ctx=ast.Load()))],
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
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr='_sent', ctx=ast.Store(),
                )],
                value=ast.Constant(value=None),
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
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
    send_def = _emit_send(blocks)

    cls = ast.ClassDef(
        name='_Gen_' + name,
        bases=[],
        keywords=[],
        body=[init_def, iter_def, next_def, send_def],
        decorator_list=[],
        type_params=[],
    )

    # The user's `def name(...)` becomes:
    #   _Gen_name = <class>
    #   name = lambda *a, **kw: _Gen_name(*a, **kw)  (with decorators)
    # Decorators are applied to the lambda, not the class.
    forwarder_args = _clone_args_for_forwarder(args)
    forwarder_call = ast.Call(
        func=ast.Name(id=cls.name, ctx=ast.Load()),
        args=_args_as_call_args(args),
        keywords=_args_as_call_kwargs(args),
    )
    forwarder = ast.Lambda(args=forwarder_args, body=forwarder_call)

    # Apply decorators outside-in (same order as Python's `@deco`).
    decorated = forwarder
    for d in reversed(decorator_list):
        decorated = ast.Call(func=d, args=[decorated], keywords=[])

    bind = ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=decorated,
    )

    return [cls, bind]


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


def _emit_send(blocks: list) -> ast.FunctionDef:
    """The state-machine dispatch. Big while True: if/elif chain by
    self.state."""
    # Build the if/elif chain.
    chain_body: list = []
    cur_orelse: list = []
    for blk in reversed(blocks):
        block_body = list(blk.stmts) + _emit_terminator(blk.terminator, blocks)
        cur = ast.If(
            test=ast.Compare(
                left=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
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

    while_body = chain_body + [
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

    body = [
        # self._sent = sent
        ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
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
                    value=ast.Name(id='self', ctx=ast.Load()),
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
                    value=ast.Name(id='self', ctx=ast.Load()),
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
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr='state', ctx=ast.Store(),
                )],
                value=ast.Constant(value=t.next),
            ),
            ast.Return(value=t.value),
        ]
    if isinstance(t, TYieldFrom):
        # Drive self.<iter_var> with next(); return its yield value.
        # When it's exhausted, advance to next state and continue the
        # loop. Phase 1 doesn't propagate StopIteration.value.
        return [
            ast.Assign(
                targets=[ast.Name(id='_v', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='next', ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
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
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )],
                        value=ast.Constant(value=t.next),
                    ),
                    ast.Continue(),
                ],
                orelse=[ast.Return(value=ast.Name(id='_v', ctx=ast.Load()))],
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
                            value=ast.Name(id='self', ctx=ast.Load()),
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
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='state', ctx=ast.Store(),
                        )],
                        value=ast.Constant(value=t.after),
                    ),
                    ast.Continue(),
                ],
                orelse=[
                    ast.Assign(
                        targets=[ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr=t.target_name, ctx=ast.Store(),
                        )],
                        value=ast.Name(id='_v', ctx=ast.Load()),
                    ),
                    ast.Assign(
                        targets=[ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
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
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id='StopIteration', ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                cause=None,
            ),
        ]
    raise NotImplementedError(f"terminator: {type(t).__name__}")


# ---------------------------------------------------------------------
# Top-level entrypoint, called from parse_function_def

def compile_generator(stmt, frame) -> list:
    """Replace `def gen(args): BODY` (with yields in BODY) with the
    state-machine class + a forwarder lambda binding the user's name.

    Returned statements are spliced into the enclosing scope and
    processed by parse_stmts. parse_class_def will compile the class
    via the regular onexpr path; the methods inside are plain
    synchronous Python and don't trip on yields."""

    name_provider = frame.get_temp_var

    # 1. ANF-lift any yields embedded in sub-expressions.
    body = anf_transform(stmt.body, name_provider)

    # 2. Discover locals that need boxing on self.
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
    )
