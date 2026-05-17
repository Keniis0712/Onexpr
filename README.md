# Onexpr — squash a Python program into a single expression

[中文 / Chinese](README.zh.md)

**Onexpr** rewrites a whole Python program into one expression. The output
is still valid Python — just one giant expression glued together with
short-circuit `or`/`and`, walrus operators, ternary `if/else`,
comprehensions, and a handful of injected runtime helper classes. Useful
for obfuscation or as a programming-language curiosity.

## Usage

```bash
python onexpr.py --input my_program.py --output obfuscated.py
python obfuscated.py    # runs the same as the input
```

## Requirements

Python **3.13+** at runtime. Function-local `del x` relies on PEP 667
(write-through `f_locals`). The transformer itself also runs on 3.13+.

## What's supported

Most modern Python:

- Control flow: `if` / `for` / `while` / `break` / `continue` / loop `else` / `pass`.
- Functions: `def`, defaults, `*args` / `**kwargs`, decorators, closures,
  `nonlocal`, `global`, `lambda`.
- Classes: inheritance, multi-inheritance, metaclasses,
  `metaclass.__prepare__`, zero-arg `super()`, descriptors,
  `__init_subclass__`, dunder methods.
- Comprehensions: list / set / dict / generator with multiple `for`
  clauses and `if` filters.
- Exceptions: `try` / `except` / `else` / `finally`, exception chaining,
  bare `raise`, `except*` (PEP 654).
- Context managers: `with`, multi-context `with`, `async with`.
- `match` / `case` (PEP 634), all pattern kinds plus guards.
- Generators: `yield` / `yield from`, `send` / `throw` / `close`,
  `return value`, `try` / `with` crossing `yield`,
  `break` / `continue` / `return` through `finally`. PEP 380 send /
  throw / close forwarding through `yield from`.
- async: `async def` / `await`, `async for`, `async with`, async
  generators (with `await` between yields plus `asend` / `athrow` /
  `aclose`), async comprehensions.
- `inspect.isgeneratorfunction` / `iscoroutinefunction` /
  `isasyncgenfunction` recognise transformed functions.
- PEP 695 type parameters: `type X = ...`, `def f[T](...)`,
  `class C[T]:`, `ParamSpec`, `TypeVarTuple`, PEP 696 defaults.
- Imports, `del`, `assert`, runtime annotations, `:=`, augmented
  assignment, starred unpacking.

## Known limitations

- Inside a function, `del x` doesn't raise `NameError` on later access —
  CPython doesn't expose a way to unbind a fast local from Python code.
  The slot is set to `None` instead.
- `for *a, b in ...` (starred tuple-unpack as the loop target) doesn't
  let `a` / `b` escape the loop.
- async comprehensions only work as the right-hand side of an
  assignment or a `return` — not nested inside a larger expression.
- `inspect.isasyncgen(instance)` returns `False` for our async-generator
  instances. The check uses `isinstance(obj, types.AsyncGeneratorType)`
  with a concrete C type, and our wrapper class isn't that type.
  `inspect.isasyncgenfunction(forwarder)` does work via the
  `_has_code_flag` patch.
- We replace `typing.TypeAliasType` with an ABC proxy so
  `isinstance(x, typing.TypeAliasType)` keeps working for both real C
  instances and our duck instances. Side effect:
  `type(x) is typing.TypeAliasType` becomes `False`, and the swap is
  process-wide if the obfuscated code is imported as a library. The
  same is true for the `inspect._has_code_flag` patch we install when
  the user code uses async generators.

## How it works

1. Parse with `ast.parse`.
2. Pre-passes: zero-arg `super()` rewrite, `nonlocal` / try-clause
   boxing, generator-state-machine compilation for `def` / `async def`
   bodies that contain `yield` or `await`.
3. Each remaining statement is rewritten by a `parse_<stmt>` function
   in `trans/parsers.py` that returns nodes closer to a pure
   expression. Repeat until everything is an `ast.expr`.
4. Combine with `or` (short-circuit gives sequencing) and `and False`
   to keep each step falsy.
5. Inject helper classes (`_FuncHelper`, `_ForHelper`, `_WhileHelper`,
   plus optional `_TryHelper`, `_LazyAlias`, `_AsyncGenWrapper`, …)
   at the top of the tree.
6. `ast.unparse`.

## Running the tests

```bash
cd tests
mkdir -p output
PYTHONPATH=.. python test.py
```

Each `t_*.py` fixture is a small program; the harness runs the original
and obfuscated versions and compares stdout.

```bash
PYTHONPATH=.. python -m unittest test.TestSuperTransformer
```

## Layout

```
onexpr.py              CLI wrapper
trans/                 transformation pipeline
  parsers.py             parse_<stmt> dispatch table, the bulk of it
  gen_compile.py         generator / async state-machine compiler
  match_patterns.py      match/case → IfExp chain
  nonlocals.py           nonlocal + try-clause boxing pre-pass
  passes.py              SuperTransformer, name collectors
  helpers.py             runtime injection
  frame.py               temp-name allocation, scope state
  root.py                top-level entry
  runtime/
    core.py              _FuncHelper / _ForHelper / _WhileHelper / async helpers
    try_helper.py        _TryHelper + reentrant event-loop subclass
    typealias.py         _LazyAlias + typing.TypeAliasType proxy
    inspect_patch.py     inspect._has_code_flag monkey-patch
tests/
  t_*.py                 round-trip fixtures
  test.py                unittest entry point
```
