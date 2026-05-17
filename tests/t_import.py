import math
print(math.pi)
import sys as msys
print(msys.platform)
import os.path
print(os.path.sep)
from builtins import repr as re
print(repr is re)


# Regression: `from m import *` used to emit `module.*` (a syntax
# error). Now it pulls every non-underscore name in dir() (or
# __all__) into globals().
from math import *
print(round(sqrt(81), 3))


# Regression: `from m import sub` for a submodule (e.g. `from tkinter
# import ttk`). Previously the fromlist passed to __import__ was the
# parent module name itself, so the import-loader didn't actually load
# the submodule. Now we pass the names being imported so submodules
# get loaded as attributes.
from urllib import parse as _urllib_parse
print(hasattr(_urllib_parse, 'urlencode'))


# Regression: `import re` (or any plain import) inside a function whose
# body has try/except / with / async-with would land in the per-clause
# lambda's locals instead of the function's helper-attribute, and a
# subsequent read (rewritten by the nonlocal pass to `helper._b_re`)
# would AttributeError. parse_import / parse_import_from now route the
# Store through helper._b_<name> when the surrounding function marked
# the name as boxed.
def _import_in_with():
    from contextlib import nullcontext
    with nullcontext():
        import re
        return re.search(r"x", "axb").group(0)


print(_import_in_with())


def _import_in_try():
    try:
        import json
        return json.dumps({"a": 1})
    except Exception:
        return None


print(_import_in_try())
