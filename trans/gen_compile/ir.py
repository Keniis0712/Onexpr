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
