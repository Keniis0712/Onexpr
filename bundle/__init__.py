"""Multi-module → single-file Python bundler.

Companion to the `trans` (statements → expression) transformer. The
two compose: `bundle.build` flattens a multi-file package into one
.py, which `trans.parse_root` can then squash into a single
expression.

Public entry point:
    bundle.build(root, package, entry, output)

CLI integration lives in onexpr.py.
"""
from .emit import build

__all__ = ["build"]
