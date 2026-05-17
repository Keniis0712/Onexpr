# Onexpr —— 把 Python 程序压成单个表达式

[English](README.md)

**Onexpr** 把整个 Python 程序重写成一个表达式。输出仍是合法 Python——
只是变成一个用短路 `or` / `and`、海象、三元 `if/else`、推导式和几个
注入的 runtime helper class 拼起来的巨大表达式。用途:代码混淆,或者
当成一个编程语言玩具。

## 用法

```bash
python onexpr.py --input my_program.py --output obfuscated.py
python obfuscated.py    # 跟原文件一样跑
```

## 环境要求

运行时 **Python 3.13+**。函数内 `del x` 依赖 PEP 667(优化帧的
`f_locals` 写穿)。转换器自身也跑在 3.13+ 上。

## 支持的功能

现代 Python 大部分语法:

- 控制流:`if` / `for` / `while` / `break` / `continue` / 循环 `else` / `pass`。
- 函数:`def`、默认参数、`*args` / `**kwargs`、装饰器、闭包、
  `nonlocal`、`global`、`lambda`。
- 类:继承、多继承、metaclass、`metaclass.__prepare__`、零参 `super()`、
  描述符、`__init_subclass__`、各种 dunder。
- 推导式:list / set / dict / generator,多 `for` 子句和 `if` 过滤。
- 异常:`try` / `except` / `else` / `finally`、异常链、裸 `raise`、
  `except*`(PEP 654)。
- 上下文管理:`with`、多上下文 `with`、`async with`。
- `match` / `case`(PEP 634),所有 pattern + guard。
- 生成器:`yield` / `yield from`、`send` / `throw` / `close`、
  `return value`、`try` / `with` 跨 `yield`、
  `break` / `continue` / `return` 穿 `finally`。
- async:`async def` / `await`、`async for`、`async with`、
  async generator(yield 之间能 await)、async comprehension。
- PEP 695 类型参数:`type X = ...`、`def f[T](...)`、`class C[T]:`、
  `ParamSpec`、`TypeVarTuple`、PEP 696 默认值。
- 导入、`del`、`assert`、运行时注解、`:=`、复合赋值、星号解包。

## 已知边界

- 函数内 `del x` 后续访问不会抛 `NameError`——CPython 不允许 Python
  层 unbind fast local。槽位被设为 `None`。
- `for *a, b in ...`(目标里有星号)的循环变量不会逃逸到外层。
- async 推导式只支持出现在赋值右侧或 `return` 顶层——不支持嵌在更大
  表达式里。
- 我们把 `typing.TypeAliasType` 替换成 ABC proxy,
  `isinstance(x, typing.TypeAliasType)` 对真 C 实例和我们自己的 duck
  实例都工作。副作用:`type(x) is typing.TypeAliasType` 变 `False`,
  且替换是**进程级**的——混淆模块被普通程序 import 时该程序的 typing
  也会被改。

## 工作原理

1. `ast.parse` 解析。
2. 前置 pass:零参 `super()` 改写、`nonlocal` / try 子句装箱、含
   `yield` 或 `await` 的 `def` / `async def` 编译成生成器状态机。
3. 剩下的每条语句由 `trans/parsers.py` 里对应的 `parse_<stmt>`
   返回若干"更接近纯表达式"的节点。反复降级直到全是 `ast.expr`。
4. 用 `or` 串起来(短路求值实现顺序执行),每一步加 `and False`
   保持 falsy 让 Or 链不短路。
5. 把 helper class(`_FuncHelper`、`_ForHelper`、`_WhileHelper`,
   按需还有 `_TryHelper` / `_LazyAlias` / `_AsyncGenWrapper` 等)
   接到树顶。
6. `ast.unparse` 输出。

## 跑测试

```bash
cd tests
mkdir -p output
PYTHONPATH=.. python test.py
```

每个 `t_*.py` 是一个小程序;harness 跑原版和混淆版,比较 stdout。

```bash
PYTHONPATH=.. python -m unittest test.TestSuperTransformer
```

## 仓库结构

```
onexpr.py              CLI 包装
trans/                 转换流水线
  parsers.py             parse_<stmt> 调度表,改写器主体
  gen_compile.py         生成器 / async 状态机编译器
  match_patterns.py      match/case → IfExp 链
  nonlocals.py           nonlocal + try 子句装箱 pre-pass
  passes.py              SuperTransformer、名字收集
  helpers.py             runtime 注入
  frame.py               临时名分配、作用域状态
  root.py                顶层入口
  runtime/
    core.py              _FuncHelper / _ForHelper / _WhileHelper / async 辅助
    try_helper.py        _TryHelper + 可重入事件循环子类
    typealias.py         _LazyAlias + typing.TypeAliasType proxy
tests/
  t_*.py                 round-trip fixture
  test.py                unittest 入口
```
