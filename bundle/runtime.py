"""Runtime preamble injected at the top of every bundle.

Provides the module loader (`_bnd_load`) and the `from x import *`
helper (`_bnd_star`). Pure Python, no exec/eval/compile.

Defined as a plain string so the emit step can drop it into the output
verbatim.
"""

RUNTIME = '''\
import sys as _bnd_sys, types as _bnd_types, linecache as _bnd_linecache
sys = _bnd_sys

_BND_MODULES = {}        # name -> (kind, init_func)
_BND_SOURCES = {}        # name -> source text (for inspect/traceback)

def _bnd_register(name, kind, init, src):
    _BND_MODULES[name] = (kind, init)
    _BND_SOURCES[name] = src
    fname = f"<bundle:{name}>"
    _bnd_linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)

def _bnd_rebind_globals(mod):
    """After a module's init runs, every function it defined has
    ``__globals__`` pointing at the bundle file's globals (because
    each function is created inside the init function's frame, whose
    f_globals is the bundle's). That breaks `globals()` calls and
    `__name__` lookups in user code. Rebind every FunctionType in
    mod.__dict__ to point at mod.__dict__ instead.

    To keep references to bundle-level runtime helpers (`_FuncHelper`,
    `_LazyAlias`, etc. — anything injected before the user code) still
    resolvable from the rebound functions, we first copy any
    bundle-global names that aren't already on the module. This way
    `globals()` from inside the module sees the module's own state
    AND can still reach the helpers as a fallback.

    Functions with non-empty ``__closure__`` are left untouched —
    rebinding ``__globals__`` would unbind their captured cells.
    """
    # Copy bundle-level names that look like internal helpers (begin
    # with `_` but aren't dunders) so onexpr-injected runtime classes
    # like `_FuncHelper` stay visible from rebound functions.
    # Public bundle-level names (`sys`, `_BND_MODULES`, etc.) would
    # leak into user globals() output and are not needed by user
    # code — leave them out.
    bundle_globals = globals()
    for k, v in bundle_globals.items():
        if not k.startswith('_') or k.startswith('__'):
            continue
        if k.startswith('_BND_') or k.startswith('_bnd_'):
            continue
        if k in mod.__dict__:
            continue
        mod.__dict__[k] = v
    for k, v in list(mod.__dict__.items()):
        if not isinstance(v, _bnd_types.FunctionType):
            continue
        if v.__globals__ is mod.__dict__:
            continue
        if v.__closure__:
            # Has free variables — leave it alone.
            continue
        new = _bnd_types.FunctionType(
            v.__code__, mod.__dict__, v.__name__,
            v.__defaults__, v.__closure__,
        )
        new.__kwdefaults__ = v.__kwdefaults__
        new.__doc__ = v.__doc__
        new.__dict__.update(v.__dict__)
        new.__module__ = mod.__name__
        try:
            new.__qualname__ = v.__qualname__
        except AttributeError:
            pass
        mod.__dict__[k] = new

def _bnd_load(name):
    m = _bnd_sys.modules.get(name)
    if m is not None:
        return m
    if name not in _BND_MODULES:
        # external — defer to normal import system
        return __import__(name, fromlist=["*"])
    # ensure parent is loaded first
    parent, _, _ = name.rpartition(".")
    if parent:
        _bnd_load(parent)
    kind, init = _BND_MODULES[name]
    mod = _bnd_types.ModuleType(name)
    mod.__file__ = f"<bundle:{name}>"
    if kind == "package":
        mod.__package__ = name
        mod.__path__ = []   # marks as package, empty -> no filesystem search
    else:
        mod.__package__ = name.rpartition(".")[0]
    _bnd_sys.modules[name] = mod
    if parent:
        setattr(_bnd_sys.modules[parent], name.rpartition(".")[2], mod)
    try:
        init(mod)
    except BaseException:
        _bnd_sys.modules.pop(name, None)
        raise
    _bnd_rebind_globals(mod)
    return mod

def _bnd_star(dst, src):
    names = getattr(src, "__all__", None)
    if names is None:
        names = [k for k in vars(src) if not k.startswith("_")]
    for n in names:
        setattr(dst, n, getattr(src, n))
'''
