"""Monkey-patch inspect._has_code_flag so onexpr-compiled forwarders
look like the function shape they replace.

`inspect.isgeneratorfunction(f)` and friends ultimately reduce to
`bool(f.__code__.co_flags & CO_*_FLAG)`. Our forwarders are lambdas:
some of them (sync generator) carry the real CO_GENERATOR flag because
they contain `yield from`; coroutine forwarders cheat with
`_is_coroutine_marker`; but async-generator forwarders can't acquire
CO_ASYNC_GENERATOR through any expression-level construct.

Rather than guess in each isXxxfunction helper individually, we patch
the underlying `inspect._has_code_flag` to first consult an
`_onexpr_code_flags` attribute on the object. Any of our forwarders
that wants to advertise extra flags sets that attribute. Other callers
(real Python functions, third-party callables) aren't affected because
they don't carry the attribute.

The patch is process-wide. If onexpr output is imported as a library,
the host program's `inspect` is also patched. The fallback path runs
the original implementation byte-for-byte, so non-onexpr callables
behave exactly as before.
"""

import inspect as _inspect


_ONEXPR_INSPECT_ORIG_HAS_CODE_FLAG = _inspect._has_code_flag


def _onexpr_patched_has_code_flag(f, flag):
    flags = getattr(f, '_onexpr_code_flags', None)
    if flags is not None:
        return bool(flags & flag)
    return _ONEXPR_INSPECT_ORIG_HAS_CODE_FLAG(f, flag)


if not getattr(_inspect, '_onexpr_inspect_patched', False):
    _inspect._has_code_flag = _onexpr_patched_has_code_flag
    _inspect._onexpr_inspect_patched = True
