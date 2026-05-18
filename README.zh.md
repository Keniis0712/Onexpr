# Onexpr —— 把 Python 程序压成单个表达式

[English](README.md)

**Onexpr** 把整个 Python 程序重写成一个表达式。输出仍是合法 Python——只是变成一个
用短路 `or` / `and`、海象、三元 `if/else`、推导式和几个注入的 runtime helper class
拼起来的巨大表达式。用途：代码混淆，或者当成一个编程语言玩具。

## 用法

```bash
python onexpr.py --input my_program.py --output obfuscated.py
python obfuscated.py    # 跟原文件一样跑
```

## 环境要求

运行时 **Python 3.13+**。函数内 `del x` 依赖 PEP 667（优化帧的 `f_locals`
写穿）。转换器自身也需要 3.13+。

## 支持的功能

现代 Python 大部分语法：

- 控制流：`if` / `for` / `while` / `break` / `continue` / 循环 `else` / `pass`。
- 函数：`def`、默认参数、`*args` / `**kwargs`、装饰器、闭包、`nonlocal`、`global`、`lambda`。
- 类：继承、多继承、metaclass、`metaclass.__prepare__`、零参 `super()`、描述符、
  `__init_subclass__`、各种 dunder。
- 推导式：list / set / dict / generator，多 `for` 子句和 `if` 过滤。
- 异常：`try` / `except` / `else` / `finally`、异常链、裸 `raise`、`except*`（PEP 654）。
- 上下文管理：`with`、多上下文 `with`、`async with`。
- `match` / `case`（PEP 634），所有 pattern + guard。
- 生成器：`yield` / `yield from`、`send` / `throw` / `close`、`return value`、
  `try` / `with` 跨 `yield`、`break` / `continue` / `return` 穿 `finally`、
  `yield from` 透传 send / throw / close（PEP 380）。
- async：`async def` / `await`、`async for`、`async with`、async generator
  （yield 之间能 await，且支持 `asend` / `athrow` / `aclose`）、async comprehension。
- `inspect.isgeneratorfunction` / `iscoroutinefunction` / `isasyncgenfunction`
  对转换后的函数返回正确结果。
- PEP 695 类型参数：`type X = ...`、`def f[T](...)`、`class C[T]:`、
  `ParamSpec`、`TypeVarTuple`、PEP 696 默认值。
- 导入、`del`、`assert`、运行时注解、`:=`、复合赋值、星号解包。

## 已知边界

- 函数内 `del x` 后续访问不会抛 `NameError`——CPython 不允许 Python 层 unbind
  fast local。槽位被设为 `None`。模块顶层的 `del` 正常工作。
- `for *a, b in ...`（目标里有星号）的循环变量不会逃逸到外层。
- async 推导式只支持出现在赋值右侧或 `return` 顶层——不支持嵌在更大表达式里。
- `inspect.isasyncgen(instance)` 对我们的 async-generator 实例返回 `False`——
  它检查的是 `isinstance(obj, types.AsyncGeneratorType)`，那是具体 C 类型，
  我们的 wrapper 不是。`inspect.isasyncgenfunction(forwarder)` 通过
  `_has_code_flag` patch 正常工作。
- 我们把 `typing.TypeAliasType` 替换成 ABC proxy，`isinstance(x, typing.TypeAliasType)`
  对真 C 实例和我们自己的 duck 实例都工作。副作用：`type(x) is typing.TypeAliasType`
  变 `False`，且替换是**进程级**的——混淆模块被普通程序 import 时该程序的 typing
  也会被改。`inspect._has_code_flag` 的 patch 在用户代码含 async generator 或
  coroutine 时同样进程级注入。

## 测试

**49 个 round-trip fixture** 覆盖了几乎全部支持的语法。每个 `t_*.py` 是一个小程序；
harness 跑原版和混淆版，对比 stdout。

```bash
cd tests
mkdir -p output      # 首次运行需要创建
PYTHONPATH=.. py -3.13 test.py
```

跑单个 TestCase 或方法：

```bash
cd tests
PYTHONPATH=.. py -3.13 -m unittest test.TestSuperTransformer
PYTHONPATH=.. py -3.13 -m unittest test.TestSuperTransformer.test_simple_class
```

**必须用 Python 3.13+**。用旧版解释器跑会导致 `t_delete.py` 失败（`del` 的 GC
测试依赖 PEP 667 写穿 `f_locals`）。

### 覆盖范围

| 领域 | Fixture |
|------|---------|
| 基本语句 | `t_assign`, `t_aug_assign`, `t_ann_assign`, `t_expr`, `t_pass`, `t_delete`, `t_global`, `t_nonlocal`, `t_nonlocal_extra` |
| 控制流 | `t_if`, `t_if_chain`, `t_for`, `t_while`, `t_break`, `t_continue`, `t_control_flow`, `t_for_var_leak` |
| 函数与闭包 | `t_function_def`, `t_lambda_comp`, `t_closure`, `t_decorators`, `t_call_forms`, `t_kwargs_super`, `t_return` |
| 类 | `t_class`, `t_class_extra`, `t_super`, `t_magic`, `t_nesting` |
| 异常 | `t_raise`, `t_assert`, `t_try`, `t_try_extra`, `t_try_star`, `t_try_top` |
| 上下文管理 | `t_with` |
| 模式匹配 | `t_match` |
| 生成器 | `t_generator` |
| async / await | `t_async` |
| 类型系统 | `t_annotations`, `t_pep695`, `t_type_alias` |
| 杂项 | `t_import`, `t_literals`, `t_scope_misc`, `t_edge`, `t_algos` |

### 标准库压力测试

CPython 3.13 的全部 392 个 `test_*.py` 文件全部转换成功，耗时约 60 秒。
这覆盖了 C 扩展交互、复杂 metaclass 层次以及标准库用到的所有语法。

## 示例

`examples/` 目录包含可直接运行的真实场景程序：

| 文件 | 测试内容 |
|------|---------|
| `calc.py` | 递归下降解析器、闭包、异常 |
| `class_demo.py` | 继承、描述符、`__init_subclass__`、零参 `super()` |
| `life.py` | Conway 生命游戏——嵌套循环、列表推导式 |
| `http_client.py` | `urllib` 请求 example.com、上下文管理器 |
| `async_pipeline.py` | `asyncio` pipeline、async generator、`async for` |
| `numpy_demo.py` | NumPy 数组运算（避免对数组调用 `__bool__`） |
| `fastapi_demo.py` | FastAPI 应用——`async def` 端点、Pydantic 模型、`Depends` |
| `tk_demo.py` | Tkinter GUI——事件循环、回调、控件类 |

## 工作原理

1. `ast.parse` 解析。
2. 前置 pass：`nonlocal` / try 子句装箱、含 `yield` 或 `await` 的函数体编译成
   生成器状态机。
3. 剩下的每条语句由 `trans/parsers/` 里对应的 `parse_<stmt>` 返回若干更接近纯
   表达式的节点。反复降级直到全是 `ast.expr`。
4. 用 `or` 串起来（短路求值实现顺序执行）；每一步用 `(value, False)[1]` 强制
   falsy，防止 `or` 链提前短路。
5. 把 helper class（`_FuncHelper`、`_ForHelper`、`_WhileHelper`，按需还有
   `_TryHelper` / `_LazyAlias` / `_AsyncGenWrapper` 等）接到树顶，通过带拓扑
   排序的注册表按依赖顺序注入。
6. `ast.unparse` 输出。

## 仓库结构

```
onexpr.py                  CLI 包装
trans/                     转换流水线
  root.py                    顶层入口
  passes.py                  SuperTransformer、NodePresenceDetector、名字收集
  nonlocals.py               nonlocal + try 子句装箱 pre-pass
  helpers.py                 runtime helper 注册表 + 注入（add_helper）
  frame.py                   临时名分配、作用域状态
  match_patterns.py          match/case → IfExp 链
  parsers/                   parse_<stmt> 函数（按语句类别分模块）
    utils.py                   add_deco、slice_to_callable、_binding_target
    func_def.py                parse_function_def、gen_func、async 降级
    class_def.py               parse_class_def、零参 super() 探测
    pep695.py                  PEP 695 类型参数 wrapper
    control_flow.py            parse_for/while/if/with/match、循环辅助函数
    exceptions.py              parse_raise/try/try_star/assert
    imports.py                 parse_import/import_from
    simple.py                  parse_return/delete/assign/aug_assign/…
    dispatch.py                name2func 表、parse_stmt、parse_stmts
  gen_compile/               生成器 / async 状态机编译器
    ir.py                      CFG terminator dataclass（TGoto、TYield 等）
    anf.py                     A-normal form 提升、with→try 降级
    locals.py                  用户局部变量发现
    cfg.py                     _CFGBuilder、build_cfg
    self_rewrite.py            _SelfRewriter（Name → self.<name>）
    emit.py                    emit_state_machine、send/throw/close 生成
    compile.py                 compile_generator 入口
  runtime/                   helper 源文件（与用户代码一起走转换流水线）
    func_helper.py             _FuncHelper（return、nonlocal 装箱）
    for_helper.py              _ForHelper（for 循环、break/continue/return）
    while_helper.py            _WhileHelper（while 循环、with 语句）
    try_helper.py              _TryHelper（try/except/finally、裸 raise）
    make_class.py              _make_class（metaclass 协议、__class__ cell）
    del_local.py               _del_local（函数局部 del 尽力实现）
    gen_sentinel.py            _GEN_DONE_SENTINEL
    await_iter.py              _await_iter
    async_gen.py               _AsyncGenWrapper、_async_gen_anext/asend/athrow/aclose
    typealias.py               _LazyAlias、typing.TypeAliasType proxy
    inspect_patch.py           inspect._has_code_flag monkey-patch
tests/
  t_*.py                     round-trip fixture（共 49 个）
  test.py                    unittest 入口
examples/
  *.py                       可运行的示例程序
```
