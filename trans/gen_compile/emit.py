import ast

from .ir import (Block, TGoto, TBranch, TYield, TYieldFrom, TForIter,
                 TReturn, TEnd, TReraise, TUnreachable)
from .self_rewrite import _self_name, rewrite_block_to_self


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

    # Initialize all boxed user locals (non-arg names) to None so that
    # ANF dehydration reads (x = self.x before a nested def/class) never
    # fail with AttributeError before the first assignment to x.
    arg_names = {a.arg for group in (args.posonlyargs, args.args, args.kwonlyargs)
                 for a in group}
    if args.vararg:
        arg_names.add(args.vararg.arg)
    if args.kwarg:
        arg_names.add(args.kwarg.arg)
    for lname in sorted(boxed - arg_names):
        init_body.append(
            ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id=init_self, ctx=ast.Load()),
                    attr=lname, ctx=ast.Store(),
                )],
                value=ast.Constant(value=None),
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
        # Advertise CO_COROUTINE (0x80) instead of the actual lambda
        # flags (which include CO_GENERATOR because we use
        # `yield from` to drive the state machine). Without this
        # frameworks that probe inspect.isgeneratorfunction first
        # (FastAPI, dependency injection libraries, …) misclassify
        # the coroutine as a sync generator and try to stream it.
        set_name_stmts.append(
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=raw_name, ctx=ast.Load()),
                        attr='_onexpr_code_flags',
                        ctx=ast.Store(),
                    ),
                ],
                # 0x80 == inspect.CO_COROUTINE
                value=ast.Constant(value=0x80),
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

