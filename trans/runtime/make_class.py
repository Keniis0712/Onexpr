"""_make_class: full metaclass protocol + PEP 560 + PEP 3135 __class__ cell."""

# Sentinel for "cell has no contents yet" — used in _make_class to
# distinguish an empty cell from one that holds None.
_CELL_EMPTY = object()


def _make_class(metaclass, name, bases, body_dict, **kw):
    # PEP 560: resolve __mro_entries__ on non-class bases (NamedTuple,
    # Generic[T], Protocol, TypedDict, …).
    orig_bases = bases
    new_bases = []
    used_mro_entries = False
    for b in bases:
        if not isinstance(b, type) and hasattr(b, '__mro_entries__'):
            used_mro_entries = True
            new_bases.extend(b.__mro_entries__(bases))
        else:
            new_bases.append(b)
    bases = tuple(new_bases)

    if metaclass is type and bases:
        for b in bases:
            mc = type(b)
            if issubclass(mc, metaclass):
                metaclass = mc
            elif not issubclass(metaclass, mc):
                raise TypeError(
                    'metaclass conflict: the metaclass of a derived '
                    'class must be a (non-strict) subclass of the '
                    'metaclasses of all its bases'
                )

    ns = metaclass.__prepare__(name, bases, **kw)

    # CPython injects __module__ before running the class body.
    if '__module__' not in body_dict:
        import sys as _sys
        f = _sys._getframe(1) if hasattr(_sys, '_getframe') else None
        ns['__module__'] = (
            f.f_globals.get('__name__', '__main__') if f is not None
            else '__main__'
        )

    if used_mro_entries:
        ns['__orig_bases__'] = orig_bases

    ns.update(body_dict)
    cls = metaclass(name, bases, ns, **kw)

    # PEP 3135 / zero-arg super(): fill the __class__ cell that the
    # class body lambda captured (initialised to None) with the real
    # class object now that it exists.
    for val in body_dict.values():
        func = val
        if isinstance(val, classmethod):
            func = val.__func__
        elif isinstance(val, staticmethod):
            func = val.__func__
        if callable(func) and hasattr(func, '__closure__') and func.__closure__:
            fvars = getattr(func.__code__, 'co_freevars', ())
            if '__class__' in fvars:
                idx = fvars.index('__class__')
                cell = func.__closure__[idx]
                current = getattr(cell, 'cell_contents', _CELL_EMPTY)
                if current is _CELL_EMPTY or current is None:
                    cell.cell_contents = cls

    return cls
