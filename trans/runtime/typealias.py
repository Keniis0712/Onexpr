"""PEP 695 runtime: lazy type-alias duck class + isinstance hook.

CPython's `typing.TypeAliasType` is a final C type — not subclassable,
its instances' `__class__` is not assignable, and its metaclass is `type`
(no `__instancecheck__` override). To match the full PEP 695 semantics
without a dedicated bytecode hook, we:

1. Define `_LazyAlias`, a duck class whose `__value__` is a property
   backed by a thunk so RHS evaluation stays lazy. `__type_params__`,
   `__getitem__` (for `Box[int]`), `__name__`, `__repr__` all match
   what the real `TypeAliasType` exposes.

2. Replace `typing.TypeAliasType` with `_TAProxy`, an `abc.ABC` subclass
   that has both the original C `TypeAliasType` and our `_LazyAlias`
   registered as virtual subclasses. After this swap,
   `isinstance(x, typing.TypeAliasType)` returns True for both real
   instances (constructed via the proxy's `__new__`, which forwards to
   the real C constructor) and our duck instances. `__new__` forwards
   so user code that explicitly calls `typing.TypeAliasType('X', v)`
   keeps working.

This mutation is process-wide. If onexpr-transformed code is imported
as a library by another program, that program's `typing.TypeAliasType`
also gets swapped. Practical impact:
  - `isinstance(x, typing.TypeAliasType)` keeps working (proxy registers
    the real C type, so its instances still match).
  - `type(x) is typing.TypeAliasType` becomes False for real instances
    (`type(x)` returns the C class; `typing.TypeAliasType` is now the
    proxy). This is the only observable regression.
  - `typing.TypeAliasType('X', v)` keeps working (proxy `__new__`
    forwards).
"""

import abc as _abc
import typing as _typing
import types as _types


_REAL_TYPE_ALIAS_TYPE = _typing.TypeAliasType


class _LazyAlias:
    """Duck-typed equivalent of typing.TypeAliasType with lazy RHS."""

    def __init__(self, name, thunk, type_params=()):
        self.__name__ = name
        self._thunk = thunk
        self.__type_params__ = tuple(type_params)
        self._evaluated = False
        self._cached = None

    @property
    def __value__(self):
        if not self._evaluated:
            self._cached = self._thunk()
            self._evaluated = True
        return self._cached

    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        return _types.GenericAlias(self, params)

    def __or__(self, other):
        # Real TypeAliasType supports `Vec | None` via __or__ on the
        # C side. Mirror that: build a Union via typing.Union to
        # match get_args / typing.get_origin output.
        return _typing.Union[self, other]

    def __ror__(self, other):
        return _typing.Union[other, self]

    def __repr__(self):
        return self.__name__


class _TAProxy(_abc.ABC):
    """Stand-in for typing.TypeAliasType. Forwards construction to the
    real C type; has the real type and _LazyAlias registered so
    isinstance succeeds for both."""

    def __new__(cls, *args, **kwargs):
        return _REAL_TYPE_ALIAS_TYPE(*args, **kwargs)


_TAProxy.register(_REAL_TYPE_ALIAS_TYPE)
_TAProxy.register(_LazyAlias)
_typing.TypeAliasType = _TAProxy
