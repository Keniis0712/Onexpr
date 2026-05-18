import ast

from .ir import (Block, TGoto, TBranch, TYield, TYieldFrom, TForIter,
                 TReturn, TEnd, TReraise, TUnreachable)
from .self_rewrite import _self_name


# ---------------------------------------------------------------------
# CFG builder

class _CFGBuilder:
    def __init__(self, name_provider, frame=None):
        self.blocks: list[Block] = []
        self.name = name_provider
        self.frame = frame
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
                        attr=self.frame.get_helper_member('_sent'),
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
                        attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                            attr=self.frame.get_helper_member('_exc'), ctx=ast.Load(),
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
                            attr=self.frame.get_helper_member('_exc'), ctx=ast.Load(),
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
                                    attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                            attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                            attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                                attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                                attr=self.frame.get_helper_member('_exc'), ctx=ast.Load(),
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
                                attr=self.frame.get_helper_member('_exc'), ctx=ast.Load(),
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
                                        attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                            attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                                        attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                                        attr=self.frame.get_helper_member('state'), ctx=ast.Store(),
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
                                    attr=self.frame.get_helper_member('_stopping_via_return'), ctx=ast.Store(),
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
                                attr=self.frame.get_helper_member('_exc'), ctx=ast.Load(),
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


def build_cfg(body: list, name_provider, frame=None) -> list:
    """Returns a list of Blocks. Block 0 is the entry. The function
    must have a final terminator on every path; we add an implicit
    TEnd at the tail if the user's body falls off the end."""
    builder = _CFGBuilder(name_provider, frame=frame)
    entry = builder.new_block()
    end = builder.emit(body, entry)
    if not end.is_terminated:
        end.terminator = TEnd()
    return builder.blocks

