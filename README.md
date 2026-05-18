# Onexpr — squash a Python program into a single expression

[中文 / Chinese](README.zh.md)

**Onexpr** rewrites a whole Python program into one expression. The output is
still valid Python — just one giant expression glued together with short-circuit
`or`/`and`, walrus operators, ternary `if/else`, comprehensions, and a handful
of injected runtime helper classes. Useful for obfuscation or as a
programming-language curiosity.

## Usage

```bash
python onexpr.py --input my_program.py --output obfuscated.py
python obfuscated.py    # runs the same as the input
```

## Requirements

Python **3.13+** at runtime. Function-local `del x` relies on PEP 667
(write-through `f_locals`). The transformer itself also requires 3.13+.

## What's supported

Most modern Python:

- Control flow: `if` / `for` / `while` / `break` / `continue` / loop `else` / `pass`.
- Functions: `def`, defaults, `*args` / `**kwargs`, decorators, closures,
  `nonlocal`, `global`, `lambda`.
- Classes: inheritance, multi-inheritance, metaclasses, `metaclass.__prepare__`,
  zero-arg `super()`, descriptors, `__init_subclass__`, dunder methods.
- Comprehensions: list / set / dict / generator with multiple `for` clauses and
  `if` filters.
- Exceptions: `try` / `except` / `else` / `finally`, exception chaining, bare
  `raise`, `except*` (PEP 654).
- Context managers: `with`, multi-context `with`, `async with`.
- `match` / `case` (PEP 634), all pattern kinds plus guards.
- Generators: `yield` / `yield from`, `send` / `throw` / `close`,
  `return value`, `try` / `with` crossing `yield`,
  `break` / `continue` / `return` through `finally`. PEP 380 send / throw /
  close forwarding through `yield from`.
- async: `async def` / `await`, `async for`, `async with`, async generators
  (with `await` between yields plus `asend` / `athrow` / `aclose`), async
  comprehensions.
- `inspect.isgeneratorfunction` / `iscoroutinefunction` / `isasyncgenfunction` /
  `isasyncgen` recognise transformed functions and instances correctly.
- PEP 695 type parameters: `type X = ...`, `def f[T](...)`, `class C[T]:`,
  `ParamSpec`, `TypeVarTuple`, PEP 696 defaults.
- Imports, `del`, `assert`, runtime annotations, `:=`, augmented assignment,
  starred unpacking.

## Known limitations

- `del x` on a function local does not raise `NameError` on later access —
  CPython doesn't expose a way to unbind a fast local from Python code. The
  slot is set to `None` instead. Module-level `del` works correctly.
- `for *a, b in ...` (starred tuple-unpack as the loop target) doesn't let
  `a` / `b` escape the loop.
- We replace `typing.TypeAliasType` with an ABC proxy so
  `isinstance(x, typing.TypeAliasType)` keeps working for both real C instances
  and our duck instances. Side effect: `type(x) is typing.TypeAliasType` becomes
  `False`, and the swap is process-wide if the obfuscated code is imported as a
  library. The same is true for the `inspect._has_code_flag` patch installed
  when the user code uses async generators or coroutines.

## Testing

**49 round-trip fixtures** cover essentially all supported syntax. Each
`t_*.py` file is a small program; the harness transforms it, runs both the
original and the transformed version, and compares stdout.

```bash
cd tests
mkdir -p output      # only needed on first run
PYTHONPATH=.. py -3.13 test.py
```

Run a single test class or method:

```bash
cd tests
PYTHONPATH=.. py -3.13 -m unittest test.TestSuperTransformer
PYTHONPATH=.. py -3.13 -m unittest test.TestSuperTransformer.test_simple_class
```

**Must use Python 3.13+.** Running with an older interpreter will cause
`t_delete.py` to fail (the `del` GC test requires PEP 667 write-through
`f_locals`).

### Test coverage

| Area | Fixtures |
|------|----------|
| Basic statements | `t_assign`, `t_aug_assign`, `t_ann_assign`, `t_expr`, `t_pass`, `t_delete`, `t_global`, `t_nonlocal`, `t_nonlocal_extra` |
| Control flow | `t_if`, `t_if_chain`, `t_for`, `t_while`, `t_break`, `t_continue`, `t_control_flow`, `t_for_var_leak` |
| Functions & closures | `t_function_def`, `t_lambda_comp`, `t_closure`, `t_decorators`, `t_call_forms`, `t_kwargs_super`, `t_return` |
| Classes | `t_class`, `t_class_extra`, `t_super`, `t_magic`, `t_nesting` |
| Exceptions | `t_raise`, `t_assert`, `t_try`, `t_try_extra`, `t_try_star`, `t_try_top` |
| Context managers | `t_with` |
| Pattern matching | `t_match` |
| Generators | `t_generator` |
| async / await | `t_async` |
| Type system | `t_annotations`, `t_pep695`, `t_type_alias` |
| Misc | `t_import`, `t_literals`, `t_scope_misc`, `t_edge`, `t_algos` |

### Stdlib stress test

All 392 CPython stdlib `test_*.py` files transform successfully (tested on
CPython 3.13). The bulk transform runs in ~60 seconds. This exercises C
extensions, complex metaclass hierarchies, and all syntax the stdlib uses.

## Examples

The `examples/` directory contains runnable programs demonstrating real-world
compatibility:

| File | What it tests |
|------|---------------|
| `calc.py` | Recursive descent parser, closures, exceptions |
| `class_demo.py` | Inheritance, descriptors, `__init_subclass__`, zero-arg `super()` |
| `life.py` | Conway's Game of Life — nested loops, list comprehensions |
| `http_client.py` | `urllib` request to example.com, context managers |
| `async_pipeline.py` | `asyncio` pipeline, async generators, `async for` |
| `numpy_demo.py` | NumPy array ops (avoids `__bool__` on arrays) |
| `fastapi_demo.py` | FastAPI app — `async def` endpoints, Pydantic models, `Depends` |
| `tk_demo.py` | Tkinter GUI — event loop, callbacks, widget classes |

## How it works

1. Parse with `ast.parse`.
2. Pre-passes: `nonlocal` / try-clause boxing, generator-state-machine
   compilation for bodies containing `yield` or `await`.
3. Each remaining statement is rewritten by a `parse_<stmt>` function in
   `trans/parsers/` that returns nodes closer to a pure expression. Repeat
   until everything is an `ast.expr`.
4. Combine with `or` (short-circuit gives sequencing); each step is made falsy
   with `(value, False)[1]` so the `or` chain never short-circuits early.
5. Inject helper classes (`_FuncHelper`, `_ForHelper`, `_WhileHelper`, plus
   conditional `_TryHelper`, `_LazyAlias`, `_AsyncGenWrapper`, …) at the top of
   the tree using a registry with topological dependency ordering.
6. `ast.unparse`.

## Layout

```
onexpr.py                  CLI wrapper
trans/                     transformation pipeline
  root.py                    top-level entry point
  passes.py                  SuperTransformer, NodePresenceDetector, name collectors
  nonlocals.py               nonlocal + try-clause boxing pre-pass
  helpers.py                 runtime helper registry + injection (add_helper)
  frame.py                   temp-name allocation, scope state
  match_patterns.py          match/case → IfExp chain
  parsers/                   parse_<stmt> functions (one module per statement group)
    utils.py                   add_deco, slice_to_callable, _binding_target
    func_def.py                parse_function_def, gen_func, async lowering
    class_def.py               parse_class_def, zero-arg super() detection
    pep695.py                  PEP 695 type-parameter wrapper
    control_flow.py            parse_for/while/if/with/match, loop helpers
    exceptions.py              parse_raise/try/try_star/assert
    imports.py                 parse_import/import_from
    simple.py                  parse_return/delete/assign/aug_assign/…
    dispatch.py                name2func table, parse_stmt, parse_stmts
  gen_compile/               generator / async state-machine compiler
    ir.py                      CFG terminator dataclasses (TGoto, TYield, …)
    anf.py                     A-normal form lift, with→try lowering
    locals.py                  user-local variable discovery
    cfg.py                     _CFGBuilder, build_cfg
    self_rewrite.py            _SelfRewriter (Name → self.<name>)
    emit.py                    emit_state_machine, send/throw/close emission
    compile.py                 compile_generator entry point
  runtime/                   helper source files (transformed alongside user code)
    func_helper.py             _FuncHelper (return, nonlocal boxing)
    for_helper.py              _ForHelper (for loops, break/continue/return)
    while_helper.py            _WhileHelper (while loops, with statements)
    try_helper.py              _TryHelper (try/except/finally, bare raise)
    make_class.py              _make_class (metaclass protocol, __class__ cell)
    del_local.py               _del_local (best-effort function-local del)
    gen_sentinel.py            _GEN_DONE_SENTINEL
    await_iter.py              _await_iter
    async_gen.py               _AsyncGenWrapper, _async_gen_anext/asend/athrow/aclose
    typealias.py               _LazyAlias, typing.TypeAliasType proxy
    inspect_patch.py           inspect._has_code_flag monkey-patch
tests/
  t_*.py                     round-trip fixtures (49 total)
  test.py                    unittest entry point
examples/
  *.py                       runnable demonstration programs
```
