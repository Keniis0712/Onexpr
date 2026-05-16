import ast

from .passes import NodePresenceDetector


def code2tree(code: str) -> ast.stmt:
    tree = ast.parse(code)
    return tree.body[0]


for_helper_name = '_ForHelper'
for_helper_code = f"""
class {for_helper_name}:
    def __init__(self, iterable, func_helper):
        self.iterable = iter(iterable)
        self.stopped = False
        self.func_helper = func_helper

    def __iter__(self):
        return self

    def __next__(self):
        if self.func_helper.returned:
            self.stopped = True
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
    def __init__(self, func_helper):
        self.stopped = False
        self.ended = False
        self.func_helper = func_helper

    def __iter__(self):
        return self

    def __next__(self):
        if self.func_helper.returned:
            self.stopped = True
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

func_helper_name = '_FuncHelper'
func_helper_code = f"""
class {func_helper_name}:
    def __init__(self):
        self.returned = False
        self.value = None

    def do_return(self, v):
        self.returned = True
        self.value = v
        return True
"""
func_helper = code2tree(func_helper_code)

LIB_TYPES = '_types'
LIB_COLLECTIONS = '_collections'
LIB_INSPECT = '_inspect'


def add_helper(tree: ast.AST, top_func_helper_var: str):
    # Get all used node types
    detector = NodePresenceDetector()
    detector.visit(tree)

    # Collect helper class source as a single exec'able blob, so the helper
    # classes themselves do not get rewritten by parse_class_def. (If they
    # did, _FuncHelper's class body would reference _FuncHelper before
    # _FuncHelper is bound — chicken-and-egg.)
    helper_sources = [func_helper_code]
    if ast.For in detector.presence:
        helper_sources.append(for_helper_code)
    if ast.While in detector.presence:
        helper_sources.append(while_helper_code)
    helper_blob = '\n'.join(helper_sources)

    to_insert = [
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id='exec', ctx=ast.Load()),
                args=[
                    ast.Constant(value=helper_blob),
                    ast.Call(
                        func=ast.Name(id='globals', ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )
        )
    ]

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
                    ],
                    keywords=[],
                )
            )
        )

    to_insert.append(
        ast.Assign(
            targets=[ast.Name(id=top_func_helper_var, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=func_helper_name, ctx=ast.Load()),
                args=[],
                keywords=[],
            ),
        )
    )

    tree.body = to_insert + tree.body
