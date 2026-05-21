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
    f_globals is the bundle's). That breaks `globals()` and
    `__name__` lookups in user code. Rebind each FunctionType to a
    per-module globals dict.

    Subtlety: bundle-level helpers (`_FuncHelper`, `temp_N`,
    ``_TryHelper``...) are referenced by the function's bytecode,
    so they must be visible from the rebound function's
    ``__globals__``. But naively copying them into ``mod.__dict__``
    pollutes ``globals().keys()`` for user code — `temp_N` would
    leak into anything that does ``[k for k in globals() if not
    k.startswith('_')]``.

    Trick: ``__globals__`` can be a ``dict`` *subclass*, and
    ``LOAD_GLOBAL`` falls back to ``__missing__`` for keys not
    present. So we use a subclass that:

      - shares storage with ``mod.__dict__`` (every write goes to
        ``mod.__dict__`` so ``import x.y; x.y.X = ...`` is visible
        on the module); and
      - resolves missing keys against the bundle's own globals,
        where the helpers live.

    Functions with a non-empty ``__closure__`` are left alone —
    rebinding their ``__globals__`` would unbind captured cells.
    """
    bundle_globals = globals()
    md = mod.__dict__

    class _BundleScope(dict):
        # We initialise with a single entry so the dict is non-empty;
        # everything actually lives in ``md`` (mod.__dict__). Reads
        # delegate; writes go to ``md`` so module-level state stays
        # consistent for downstream importers.
        def __getitem__(self, k):
            try:
                return md[k]
            except KeyError:
                pass
            return bundle_globals[k]

        def __setitem__(self, k, v):
            md[k] = v

        def __delitem__(self, k):
            del md[k]

        def __contains__(self, k):
            return k in md or k in bundle_globals

        def __iter__(self):
            return iter(md)

        def keys(self):
            return md.keys()

        def values(self):
            return md.values()

        def items(self):
            return md.items()

        def __len__(self):
            return len(md)

        def __missing__(self, k):
            # Fallback path used by LOAD_GLOBAL when the bytecode
            # ALREADY tried our subclass's regular lookup and missed.
            try:
                return bundle_globals[k]
            except KeyError:
                raise KeyError(k)

        def get(self, k, default=None):
            if k in md:
                return md[k]
            return bundle_globals.get(k, default)

    scope = _BundleScope()

    for k, v in list(md.items()):
        if not isinstance(v, _bnd_types.FunctionType):
            continue
        if v.__globals__ is scope:
            continue
        if v.__closure__:
            continue
        new = _bnd_types.FunctionType(
            v.__code__, scope, v.__name__,
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
        md[k] = new

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
