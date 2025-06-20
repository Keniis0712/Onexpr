import _ast
import ast
import dataclasses
from typing import Optional, Callable


@dataclasses.dataclass
class Frame:
    prev: Optional["Frame"]
    nonlocal_vars: list[str]
    global_vars: list[str]
    in_async_def: bool = False
    temp_var_num: Optional[int] = None
    loops: list[str] = dataclasses.field(default_factory=list)

    def get_temp_var_num(self) -> int:
        if self.temp_var_num is None:
            self.temp_var_num = self.prev.get_temp_var_num()

        return self.temp_var_num

    def get_temp_var(self):
        temp_var_num = self.get_temp_var_num()
        self.temp_var_num = self.temp_var_num + 1
        return f"temp_{temp_var_num}"

    def enter_loop(self):
        self.loops.append(self.get_temp_var())

    def get_cur_loop_var(self):
        assert len(self.loops)
        return self.loops[-1]

    def exit_loop(self):
        self.loops.pop()


class SuperTransformer(ast.NodeTransformer):
    def __init__(self):
        self.class_stack = []
        self.first_arg = None

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()
        return node

    def visit_FunctionDef(self, node):
        self._set_first_arg(node)
        self.generic_visit(node)
        self.first_arg = None
        return node

    def visit_AsyncFunctionDef(self, node):
        self._set_first_arg(node)
        self.generic_visit(node)
        self.first_arg = None
        return node

    def _set_first_arg(self, node):
        if node.args.args:
            self.first_arg = node.args.args[0].arg
        else:
            self.first_arg = None

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name)
                and node.func.id == 'super'
                and len(node.args) == 0):
            if not self.class_stack:
                raise RuntimeError("Cannot determine class context for super()")
            if not self.first_arg:
                raise RuntimeError("Cannot determine first argument name for super() call")
            class_name = self.class_stack[-1]
            node.args = [
                ast.Name(id=class_name, ctx=ast.Load()),
                ast.Name(id=self.first_arg, ctx=ast.Load())
            ]
        return self.generic_visit(node)


class NodePresenceDetector(ast.NodeVisitor):
    def __init__(self):
        self.presence = {}

    def visit(self, node):
        node_type = type(node)
        if node_type not in self.presence:
            self.presence[node_type] = True
        self.generic_visit(node)

    def detect(self, tree):
        self.presence.clear()
        self.visit(tree)
        return self.presence


def code2tree(code: str) -> ast.stmt:
    tree = ast.parse(code)
    return tree.body[0]


def add_deco(name: str, decorators: list[ast.expr], func: _ast.expr) -> ast.expr:
    node = func
    for decorator in decorators:
        node = ast.Call(
            func=decorator,
            args=[node],
        )
    return node


def gen_func(stmt: ast.FunctionDef | ast.AsyncFunctionDef, sub_frame):
    temp_func_var = sub_frame.get_temp_var()
    body = stmt.body
    if not isinstance(body[-1], ast.Return):
        body.append(ast.Return(ast.Constant(value=None)))
    return [
        ast.Assign(
            targets=[ast.Name(id=temp_func_var, ctx=ast.Store())],
            value=ast.Lambda(
                args=stmt.args,
                body=ast.Subscript(
                    value=parse_stmts(body, frame=sub_frame),
                    slice=ast.Constant(value=0)
                )
            )
        ),
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=temp_func_var, ctx=ast.Load()),
                    attr='__name__'
                )
            ],
            value=ast.Constant(value=stmt.name)
        ),
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(
                stmt.name,
                stmt.decorator_list,
                ast.Name(id=temp_func_var, ctx=ast.Load()),
            )
        )
    ]


def parse_function_def(stmt: ast.FunctionDef, frame: Frame) -> list[_ast.AST]:
    sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[])
    return gen_func(stmt, sub_frame)


def parse_async_function_def(stmt: ast.AsyncFunctionDef, frame: Frame) -> list[_ast.AST]:
    # sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[], in_async_def=True)
    # stmt.decorator_list.append(
    #     ast.Attribute(
    #         value=ast.Name(id=LIB_TYPES, ctx=ast.Load()),
    #         attr='coroutine'
    #     )
    # )
    # stmts: list[_ast.AST] = gen_func(stmt, sub_frame)
    # stmts.append(
    #     ast.Expr(
    #         ast.Call(
    #             func=ast.Attribute(
    #                 value=ast.Name(id=LIB_INSPECT, ctx=ast.Load()),
    #                 attr='markcoroutinefunction'
    #             ),
    #             args=[ast.Name(id=stmt.name, ctx=ast.Load())],
    #         )
    #     )
    # )
    # return stmts
    return None


def parse_class_def(stmt: ast.ClassDef, frame: Frame) -> list[_ast.AST]:
    sub_frame = Frame(prev=frame, nonlocal_vars=[], global_vars=[])
    cls_body = stmt.body
    cls_body.append(
        ast.Return(
            value=ast.Call(
                func=ast.Name(id='locals', ctx=ast.Load()),
                args=[]
            )
        )
    )
    metaclass = ast.Name(id='type', ctx=ast.Load())
    for kwd in stmt.keywords:
        if kwd.arg == 'metaclass':
            metaclass = kwd.value
            break

    return [
        ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=add_deco(
                stmt.name,
                stmt.decorator_list,
                ast.Call(
                    func=metaclass,
                    args=[
                        ast.Constant(stmt.name),
                        ast.Tuple(elts=stmt.bases),
                        ast.Subscript(
                            value=ast.Call(
                                func=ast.Lambda(
                                    args=ast.arguments(
                                        posonlyargs=[],
                                        args=[],
                                        vararg=None,
                                        kwonlyargs=[],
                                        kw_defaults=[],
                                        defaults=[],
                                    ),
                                    body=parse_stmts(cls_body, frame=sub_frame),
                                ),
                                args=[]
                            ),
                            slice=ast.Constant(value=0)
                        )
                    ]
                )
            )
        )
    ]


def parse_return(stmt: ast.Return, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Tuple(elts=[stmt.value, ast.Constant(value=True)]),
    ]


def parse_delete(stmt: ast.Delete, frame: Frame) -> list[_ast.AST]:
    pass


def parse_assign(stmt: ast.Assign, frame: Frame) -> list[_ast.AST]:
    if len(stmt.targets) != 1:
        # This is look like "a = b = 0"

        # First copy the value to a temp var
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[
                    ast.Name(id=temp_var, ctx=ast.Store())
                ],
                value=stmt.value,
            )
        ]

        # Then copy the temp var to each target
        assigns.extend(
            ast.Assign(
                targets=[target],
                value=ast.Name(id=temp_var, ctx=ast.Load())
            )
            for target in stmt.targets
        )

        return assigns

    target = stmt.targets[0]

    # if not isinstance(target, ast.Name):
    #     # This is look like "a, b = c" or "a[b] = c"
    #     ...
    #     return None

    if isinstance(target, (ast.List, ast.Tuple)):
        # This is look like "a, b = c"
        # We first copy the value to temp var
        temp_var = frame.get_temp_var()
        assigns = [
            ast.Assign(
                targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                value=stmt.value,
            )
        ]
        # Then get each element from the temp var

        # If there is a starred name, we have to get how many elements before and after it
        names = target.elts
        before_star = None
        starred_name = None

        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                before_star = pos

                assert isinstance(name.value, ast.Name)
                starred_name = name.value.id
                break

        if before_star is not None:
            after_star = len(names) - before_star - 1
            assigns.append(
                ast.Assign(
                    targets=[ast.Name(id=starred_name, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Slice(
                            lower=ast.Constant(before_star),
                            upper=ast.Constant(-after_star),
                            col_offset=0,
                            end_col_offset=None,
                            end_lineno=None,
                            lineno=0,
                        )
                    )
                )
            )

        # Finally we start to build the assigns
        parse_before_star = True
        for pos, name in enumerate(names):
            if isinstance(name, ast.Starred):
                # We already parsed it
                parse_before_star = False
                continue
            # assert isinstance(name, ast.Name), f"{type(name)}"
            assigns.append(
                ast.Assign(
                    targets=[name],
                    value=ast.Subscript(
                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                        slice=ast.Constant(pos if parse_before_star else pos - len(names)),
                    )
                )
            )
        return assigns

    if isinstance(target, (ast.Subscript, ast.Attribute)):
        if isinstance(target, ast.Subscript):
            attr = '__setitem__'
            pos = target.slice
        else:
            attr = '__setattr__'
            pos = ast.Constant(target.attr)
        # Use func to change the value
        return [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=target.value,
                        attr=attr,
                        ctx=ast.Load(),
                    ),
                    args=[
                        pos,
                        stmt.value
                    ],
                )
            )
        ]

    # Now is a simple assign like "a = 1"
    assert isinstance(target, ast.Name)
    return [
        ast.Expr(
            value=ast.NamedExpr(
                target=target,
                value=stmt.value,
            )
        )
    ]


def parse_type_alias(stmt: ast.TypeAlias, frame: Frame) -> list[_ast.AST]:
    pass


def parse_aug_assign(stmt: ast.AugAssign, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Assign(
            targets=[stmt.target],
            value=ast.BinOp(
                left=stmt.target,
                op=stmt.op,
                right=stmt.value
            ),
        )
    ]


def parse_ann_assign(stmt: ast.AnnAssign, frame: Frame) -> list[_ast.AST]:
    pass


def parse_for(stmt: ast.For, frame: Frame) -> list[_ast.AST]:
    frame.enter_loop()
    stmts: list[_ast.AST] = [
        ast.Assign(
            targets=[ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=for_helper_name, ctx=ast.Load()),
                args=[stmt.iter]
            )
        ),
        ast.Expr(
            value=ast.ListComp(
                elt=ast.Constant(value=None),
                generators=[
                    ast.comprehension(
                        target=stmt.target,
                        iter=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                        ifs=[parse_stmts(stmt.body, frame)],
                        is_async=0
                    )
                ]
            )
        )
    ]
    if stmt.orelse:
        stmts.append(
            add_orelse(stmt.orelse, frame)
        )
    frame.exit_loop()
    return stmts


def add_orelse(orelse, frame):
    return ast.If(
        test=ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='stopped',
                ctx=ast.Load(),
            )
        ),
        body=orelse
    )


def parse_async_for(stmt: ast.AsyncFor, frame: Frame) -> list[_ast.AST]:
    pass


def parse_while(stmt: ast.While, frame: Frame) -> list[_ast.AST]:
    frame.enter_loop()
    body: list = [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='cond',
            ),
            args=[stmt.test],
        )
    ]
    body += stmt.body

    stmts: list[_ast.AST] = [
        ast.Assign(
            targets=[ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load())],
            value=ast.Call(
                func=ast.Name(id=while_helper_name, ctx=ast.Load()),
                args=[]
            )
        ),
        ast.Expr(
            value=ast.ListComp(
                elt=ast.Constant(value=None),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id='_', ctx=ast.Load()),
                        iter=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                        ifs=[parse_stmts(body, frame)],
                        is_async=0
                    )
                ]
            )
        )
    ]
    if stmt.orelse:
        stmts.append(
            add_orelse(stmt.orelse, frame)
        )
    frame.exit_loop()
    return stmts


def parse_if(stmt: ast.If, frame: Frame) -> list[_ast.AST]:
    return [
        ast.IfExp(
            test=stmt.test,
            body=parse_stmts(stmt.body, frame),
            orelse=parse_stmts(stmt.orelse, frame) if stmt.orelse else ast.Constant(None),
        )
    ]


def parse_with(stmt: ast.With, frame: Frame) -> list[_ast.AST]:
    pass


def parse_async_with(stmt: ast.AsyncWith, frame: Frame) -> list[_ast.AST]:
    pass


def parse_match(stmt: ast.Match, frame: Frame) -> list[_ast.AST]:
    pass


def parse_raise(stmt: ast.Raise, frame: Frame) -> list[_ast.AST]:
    if stmt.cause:
        temp_exc_var = frame.get_temp_var()
        stmts = [
            ast.Assign(
                targets=[ast.Name(id=temp_exc_var)],
                value=stmt.exc
            ),
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id=temp_exc_var, ctx=ast.Load()),
                        attr='__cause__',
                        ctx=ast.Store(),
                    )
                ],
                value=stmt.cause
            )
        ]
    else:
        temp_exc_var = None
        stmts = []
    stmts += [
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.GeneratorExp(
                        elt=ast.Constant(value=None),
                        generators=[
                            ast.comprehension(
                                target=ast.Name(id='_', ctx=ast.Store()),
                                iter=ast.List(ctx=ast.Load()),
                                is_async=0)
                        ]
                    ),
                    attr='throw',
                    ctx=ast.Load()
                ),
                args=[
                    ast.Name(id=temp_exc_var, ctx=ast.Load()) if stmt.cause else stmt.exc,
                ]
            )
        )
    ]
    return stmts


def parse_try(stmt: ast.Try, frame: Frame) -> list[_ast.AST]:
    pass


def parse_try_star(stmt: ast.TryStar, frame: Frame) -> list[_ast.AST]:
    pass


def parse_assert(stmt: ast.Assert, frame: Frame) -> list[_ast.AST]:
    return [
        ast.If(
            test=stmt.test,
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='AssertionError', ctx=ast.Load()),
                        args=[stmt.msg]
                    )
                )
            ],
        )
    ]


def parse_import(stmt: ast.Import, frame: Frame) -> list[_ast.AST]:
    stmts = []
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            as_name = name.split('.')[0]
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=as_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='__import__', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=name),
                    ],
                    keywords=[],
                ),
            )
        )
    return stmts


def parse_import_from(stmt: ast.ImportFrom, frame: Frame) -> list[_ast.AST]:
    stmts = []
    module_var = frame.get_temp_var()
    stmts.append(
        ast.Assign(
            targets=[ast.Name(id=module_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='__import__', ctx=ast.Load()),
                args=[
                    ast.Constant(value=stmt.module),
                ],
                keywords=[
                    ast.keyword(
                        arg='fromlist',
                        value=ast.Constant(
                            [
                                stmt.module,
                            ]
                        ),
                    )
                ],
            ),
        )
    )
    for import_name in stmt.names:
        name, as_name = import_name.name, import_name.asname
        if as_name is None:
            as_name = name
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=as_name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(id=module_var, ctx=ast.Load()),
                    attr=name,
                    ctx=ast.Load(),
                ),
            )
        )
    return stmts


def parse_global(stmt: ast.Global, frame: Frame) -> list[_ast.AST]:
    pass


def parse_nonlocal(stmt: ast.Nonlocal, frame: Frame) -> list[_ast.AST]:
    pass


def parse_expr(stmt: ast.Expr, frame: Frame) -> list[_ast.AST]:
    if frame.in_async_def and isinstance(stmt.value, ast.Await):
        value = ast.YieldFrom(value=stmt.value.value)
    else:
        value = stmt.value
    return [
        ast.BoolOp(
            op=ast.And(),
            values=[
                value,
                ast.Constant(value=False),
            ]
        )
    ]


def parse_pass(_: ast.Pass, __: Frame) -> list[_ast.AST]:
    return [
        ast.Constant(value=...),
    ]


def parse_break(_: ast.Break, frame: Frame) -> list[_ast.AST]:
    return [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=frame.get_cur_loop_var(), ctx=ast.Load()),
                attr='stop',
                ctx=ast.Load(),
            ),
            args=[]
        )
    ]


def parse_continue(stmt: ast.Continue, frame: Frame) -> list[_ast.AST]:
    # Use Constant instead of Expr to avoid adding "and False"
    return [
        ast.Constant(value=True),
    ]


name2func: dict[str, Callable] = {
    "FunctionDef": parse_function_def,
    "AsyncFunctionDef": parse_async_function_def,
    "ClassDef": parse_class_def,
    "Return": parse_return,
    "Delete": parse_delete,
    "Assign": parse_assign,
    "TypeAlias": parse_type_alias,
    "AugAssign": parse_aug_assign,
    "AnnAssign": parse_ann_assign,
    "For": parse_for,
    "AsyncFor": parse_async_for,
    "While": parse_while,
    "If": parse_if,
    "With": parse_with,
    "AsyncWith": parse_async_with,
    "Match": parse_match,
    "Raise": parse_raise,
    "Try": parse_try,
    "TryStar": parse_try_star,
    "Assert": parse_assert,
    "Import": parse_import,
    "ImportFrom": parse_import_from,
    "Global": parse_global,
    "Nonlocal": parse_nonlocal,
    "Expr": parse_expr,
    "Pass": parse_pass,
    "Break": parse_break,
    "Continue": parse_continue,
}

for_helper_name = '_ForHelper'
for_helper_code = f"""
class {for_helper_name}:
    def __init__(self, iterable):
        self.iterable = iter(iterable)
        self.stopped = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            raise StopIteration
        return next(self.iterable)

    def stop(self):
        self.stopped = True
        return True
"""
for_helper = code2tree(for_helper_code)

while_helper_name = '_WhileHelper'
while_helper_code = f"""
class {while_helper_name}:
    def __init__(self):
        self.stopped = False
        self.ended = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped or self.ended:
            raise StopIteration
        return None

    def stop(self):
        self.stopped = True
        return True

    def cond(self, condition):
        if condition:
            return False
        self.ended = True
        return True
"""
while_helper = code2tree(while_helper_code)

LIB_TYPES = '_types'
LIB_COLLECTIONS = '_collections'
LIB_INSPECT = '_inspect'


def parse_stmt(stmt: ast.stmt, frame: Frame) -> list[ast.expr]:
    exprs = [stmt]

    while not all(isinstance(expr, ast.expr) for expr in exprs):
        new_exprs = []

        for expr in exprs:
            if isinstance(expr, ast.expr):
                new_exprs.append(expr)
                continue
            class_name = expr.__class__.__name__
            assert class_name in name2func

            parse_func = name2func[class_name]
            exprs = parse_func(expr, frame)
            if exprs is None:
                raise NotImplementedError(f"Not implemented for {class_name}")

            new_exprs.extend(exprs)

        exprs = new_exprs

    return exprs


def parse_stmts(stmts: list[ast.stmt], frame: Frame) -> _ast.BoolOp:
    exprs = []
    for stmt in stmts:
        expr = parse_stmt(stmt, frame)
        exprs.extend(expr)

    return ast.BoolOp(
        op=ast.Or(),
        values=exprs
    )


def add_helper(tree: ast.AST):
    # Get all used node types
    detector = NodePresenceDetector()
    detector.visit(tree)

    to_insert = []
    if ast.For in detector.presence:
        to_insert.append(for_helper)
    if ast.While in detector.presence:
        to_insert.append(while_helper)
    if ast.AsyncFunctionDef in detector.presence:
        to_insert.append(ast.Import(
            names=[
                ast.alias(
                    name='types',
                    asname=LIB_TYPES,
                ),
                ast.alias(
                    name='collections.abc',
                    asname=LIB_COLLECTIONS,
                ),
                ast.alias(
                    name='inspect',
                    asname=LIB_INSPECT,
                )
            ]
        ))
        to_insert.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=LIB_COLLECTIONS, ctx=ast.Load()),
                                attr='abc',
                                ctx=ast.Load()
                            ),
                            attr='Coroutine',
                            ctx=ast.Load()
                        ),
                        attr='register',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id=LIB_TYPES, ctx=ast.Load()),
                            attr='GeneratorType',
                            ctx=ast.Load()
                        )
                    ]
                )
            )
        )

    tree.body = to_insert + tree.body


def parse_root(tree: ast.Module) -> ast.Module:
    # Init the top frame
    top_frame = Frame(None, [], [])
    top_frame.temp_var_num = 0

    # Transform "super()" to "super(cls, self)"
    super_trans = SuperTransformer()
    tree = super_trans.visit(tree)

    add_helper(tree)

    expr = parse_stmts(tree.body, top_frame)
    return ast.Module(
        body=[
            ast.Expr(
                expr
            )
        ],
        type_ignores=[],
    )
