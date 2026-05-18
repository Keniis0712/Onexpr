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
    return mod

def _bnd_star(dst, src):
    names = getattr(src, "__all__", None)
    if names is None:
        names = [k for k in vars(src) if not k.startswith("_")]
    for n in names:
        setattr(dst, n, getattr(src, n))
'''
