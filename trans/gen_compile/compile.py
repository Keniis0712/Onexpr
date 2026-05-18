from .anf import anf_transform
from .cfg import build_cfg
from .emit import emit_state_machine
from .locals import collect_user_locals
from .self_rewrite import rewrite_block_to_self


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
        frame=frame,
    )
