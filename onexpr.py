import ast
import argparse
import os.path

import consts
from consts import get_temp_var


# stmt = AsyncFunctionDef(identifier name, arguments args, stmt* body, expr* decorator_list, expr? returns, string? type_comment, type_param* type_params)
#      | TypeAlias(expr name, type_param* type_params, expr value)  # Not support
#      | AnnAssign(expr target, expr annotation, expr? value, int simple)  # Not support
#      | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
#      | With(withitem* items, stmt* body, string? type_comment)
#      | AsyncWith(withitem* items, stmt* body, string? type_comment)
#      | Match(expr subject, match_case* cases)
#      | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
#      | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
#      | Global(identifier* names)
#      | Nonlocal(identifier* names)


class ASTNodeFinder(ast.NodeVisitor):
    def __init__(self):
        self.nodes = {}

    def visit(self, node):
        self.nodes[node.__class__] = True
        ast.NodeVisitor.visit(self, node)


class ExprWithoutProcess(ast.Expr):
    ...


def parse_root(node: ast.Module) -> ast.Module:
    assert isinstance(node, ast.Module)  # Top layer should be module
    finder = ASTNodeFinder()
    finder.visit(node)

    body = []
    if finder.nodes.get(ast.Nonlocal, False):
        body.append(consts.nonlocal_inspect_import)
    if finder.nodes.get(ast.For, False):
        body.append(consts.for_class)
    if finder.nodes.get(ast.While, False):
        body.append(consts.while_class)

    body.extend(node.body)

    return ast.Module(
        body=[
            ast.Expr(
                value=parse_sub(body, True, [])
            )
        ],
        type_ignores=[],
    )


def is_using_nonlocal(nodes: list[ast.stmt]) -> list[str]:
    names = []
    for node in nodes:
        match node:
            case ast.Nonlocal(name):
                names.extend(name)
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                continue
            case ast.If(_, body, orelse):
                names.extend(is_using_nonlocal(body))
                names.extend(is_using_nonlocal(orelse))
            case ast.While(_, body, orelse):
                names.extend(is_using_nonlocal(body))
                names.extend(is_using_nonlocal(orelse))
            case ast.For(_, _, body, orelse):
                names.extend(is_using_nonlocal(body))
                names.extend(is_using_nonlocal(orelse))
            case ast.Try(body, handlers, orelse, finalbody):
                names.extend(is_using_nonlocal(body))
                for handler in handlers:
                    names.extend(is_using_nonlocal(handler.body))
                names.extend(is_using_nonlocal(orelse))
                names.extend(is_using_nonlocal(finalbody))
            case ast.With(_, body, _):
                names.extend(is_using_nonlocal(body))
            case ast.Match(_, cases):
                for case in cases:
                    names.extend(is_using_nonlocal(case))
            case _:
                continue
    return names


def parse_sub(nodes: list[ast.stmt], new_scoop: bool, nonlocal_name: list[str]) -> ast.BoolOp:
    exprs = []
    pos = 0
    if new_scoop:
        nonlocal_name = is_using_nonlocal(nodes)
    while pos < len(nodes):
        node = nodes[pos]

        match node:
            case ast.Expr(value) | ExprWithoutProcess(value):
                expr = ast.BoolOp(
                    op=ast.And(),
                    values=[
                        value,
                        ast.Constant(value=False),
                    ],
                ) if type(node) is ast.Expr else value
                exprs.append(expr)
            case ast.Assign(targets, value):
                if len(targets) > 1:
                    # The assign is like `a = b = val`
                    # We change it to 't = val; a = t; b = t'
                    temp_var = get_temp_var()

                    nodes[pos:pos + 1] = [
                        ast.Assign(
                            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                            value=value,
                        ),
                    ]
                    nodes[pos + 1:pos + 1] = [
                        ast.Assign(
                            targets=[target],
                            value=ast.Name(id=temp_var, ctx=ast.Load()),
                        )
                        for target in targets
                    ]
                    continue

                target = targets[0]
                if isinstance(target, ast.Tuple):
                    # The assign is like `a, b = var`
                    # We want to change it to several expressions like 't = var; a = t[0]; b = t[1]'
                    # However, var can be an iterator
                    # So we change all the sequences to iterator first
                    # Then we can use the same way

                    # If the iterator is infinity, sequence unpacking will be stuck forever
                    # So we can get all items of the iterator safely -- It won't be infinity

                    # Now we change the value to iterator the get all the values immediately
                    temp_var = get_temp_var()
                    nodes[pos:pos + 1] = [
                        ast.Assign(
                            targets=[ast.Name(id=temp_var, ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id='list', ctx=ast.Load()),
                                args=[
                                    ast.Call(
                                        func=ast.Name(id='iter', ctx=ast.Load()),
                                        args=[value],
                                        keywords=[],
                                    )
                                ],
                                keywords=[],
                            ),
                        ),
                    ]

                    items = target.elts

                    # First process the items before the star, and find the position of the star as the same time
                    new_assigns = []
                    for i, item in enumerate(items):
                        if isinstance(item, ast.Starred):
                            has_star, star_pos = True, i
                            break
                        elif isinstance(item, ast.Name):
                            if item.id == '_':
                                continue
                            new_assigns.append(
                                ast.Assign(
                                    targets=[item],
                                    value=ast.Subscript(
                                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                                        slice=ast.Constant(value=i),
                                        ctx=ast.Load(),
                                    ),
                                )
                            )
                        elif isinstance(item, ast.Attribute):
                            new_assigns.append(
                                ast.Call(
                                    func=ast.Name(id='setattr', ctx=ast.Load()),
                                    args=[
                                        item.value,
                                        ast.Constant(value=item.attr),
                                        ast.Subscript(
                                            value=ast.Name(id=temp_var, ctx=ast.Load()),
                                            slice=ast.Constant(value=i),
                                            ctx=ast.Load(),
                                        ),
                                    ],
                                    keywords=[],
                                )
                            )
                        else:
                            raise NotImplementedError
                    else:
                        # If there's no star, we have done all the job
                        nodes[pos + 1:pos + 1] = new_assigns
                        continue

                    # Then we parse the items after the star from tail to head
                    for i, item in enumerate(items[len(items):star_pos:-1]):
                        if isinstance(item, ast.Name):
                            if item.id == '_':
                                continue
                            new_assigns.append(
                                ast.Assign(
                                    targets=[item],
                                    value=ast.Subscript(
                                        value=ast.Name(id=temp_var, ctx=ast.Load()),
                                        slice=ast.Constant(value=-i - 1),
                                        ctx=ast.Load(),
                                    ),
                                )
                            )
                        elif isinstance(item, ast.Attribute):
                            new_assigns.append(
                                ast.Call(
                                    func=ast.Name(id='setattr', ctx=ast.Load()),
                                    args=[
                                        item.value,
                                        ast.Constant(value=item.attr),
                                        ast.Subscript(
                                            value=ast.Name(id=temp_var, ctx=ast.Load()),
                                            slice=ast.Constant(value=-i - 1),
                                            ctx=ast.Load(),
                                        ),
                                    ],
                                    keywords=[],
                                )
                            )
                        else:
                            raise NotImplementedError

                    # Last we parse the star

                    # noinspection PyTypeChecker
                    star: ast.Starred = items[star_pos]
                    # noinspection PyUnresolvedReferences
                    if star.value.id != '_':
                        # noinspection PyArgumentList
                        new_assigns.append(
                            ast.Assign(
                                targets=[star.value],
                                value=ast.Call(
                                    func=ast.Name(id='list', ctx=ast.Load()),
                                    args=[
                                        ast.Subscript(
                                            value=ast.Name(id=temp_var, ctx=ast.Load()),
                                            slice=ast.Slice(
                                                lower=ast.Constant(value=star_pos),
                                                upper=ast.Constant(value=-(len(items) - star_pos - 1)),
                                            ),
                                            ctx=ast.Load(),
                                        )
                                    ],
                                    keywords=[],
                                ),
                            )
                        )
                    nodes[pos + 1:pos + 1] = new_assigns
                    continue

                # Now the assign is like `a = 1`
                if isinstance(target, ast.Name):
                    if target.id not in nonlocal_name:
                        expr = ast.BoolOp(
                            op=ast.And(),
                            values=[
                                ast.NamedExpr(
                                    target=target,
                                    value=value,
                                ),
                                ast.Constant(value=False),
                            ],
                        )
                    else:
                        expr = ast.Call(
                            func=ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Name(id=consts.nonlocal_inspect_name, ctx=ast.Load()),
                                                attr='currentframe',
                                                ctx=ast.Load()
                                            ),
                                            args=[],
                                            keywords=[]
                                        ),
                                        attr='f_back',
                                        ctx=ast.Load()
                                    ),
                                    attr='f_locals',
                                    ctx=ast.Load()
                                ),
                                attr='__setitem__',
                                ctx=ast.Load()
                            ),
                            args=[
                                ast.Constant(value=target.id),
                                value,
                            ],
                            keywords=[],
                        )
                elif isinstance(target, ast.Attribute):
                    expr = ast.Call(
                        func=ast.Name(id='setattr', ctx=ast.Load()),
                        args=[
                            target.value,
                            ast.Constant(value=target.attr),
                            value,
                        ],
                        keywords=[],
                    )
                elif isinstance(target, ast.Subscript):
                    expr = ast.Call(
                        func=ast.Attribute(
                            value=target.value,
                            attr='__setitem__',
                            ctx=ast.Load(),
                        ),
                        args=[
                            target.slice,
                            value,
                        ],
                        keywords=[],
                    )
                else:
                    raise NotImplementedError(type(target))
                exprs.append(expr)
            case ast.FunctionDef(name, args, body, decorators):
                if not isinstance(body[-1], ast.Return):
                    body.append(ast.Return(value=ast.Constant(value=None)))
                new_node = [
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Lambda(
                            args=args,
                            body=ast.Subscript(
                                value=parse_sub(body, True, []),
                                slice=ast.Constant(value=1),
                                ctx=ast.Load(),
                            ),
                        ),
                    )
                ]
                new_node.extend(add_decorators(decorators, name))
                nodes[pos:pos + 1] = new_node
                continue
            case ast.Return(value):
                exprs.append(
                    ast.Tuple(
                        elts=[
                            ast.Constant(value=True),
                            value,
                        ],
                        ctx=ast.Load(),
                    )
                )
            case ast.If(test, body, orelse):
                if not orelse:
                    orelse.append(ast.Expr(value=ast.Constant(value=None)))
                exprs.append(
                    ast.IfExp(
                        test=test,
                        body=parse_sub(body, False, nonlocal_name),
                        orelse=parse_sub(orelse, False, nonlocal_name),
                    )
                )
            case ast.ClassDef(name, bases, keywords, body, decorators):
                body.append(
                    ast.Return(
                        ast.Call(
                            func=ast.Name(id='locals', ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    )
                )
                new_node = [
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id='__build_class__', ctx=ast.Load()),
                            args=[
                                     ast.Lambda(
                                         args=ast.arguments(
                                             posonlyargs=[],
                                             args=[],
                                             kwonlyargs=[],
                                             kw_defaults=[],
                                             defaults=[],
                                         ),
                                         body=parse_sub(
                                             body,
                                             True,
                                             []
                                         ),
                                     ),
                                     ast.Constant(value=name),
                                 ] + bases,
                            keywords=keywords,
                        ),
                    )
                ]
                new_node.extend(add_decorators(decorators, name))
                nodes[pos:pos + 1] = new_node
                continue
            case ast.AugAssign(target, op, value):
                nodes[pos:pos + 1] = [
                    ast.Assign(
                        targets=[target],
                        value=ast.BinOp(
                            left=target,
                            op=op,
                            right=value,
                        ),
                    )
                ]
                continue
            case ast.For(target, iter_, body, orelse):
                new_node = [
                    ast.Assign(
                        targets=[ast.Name(id=consts.for_obj_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id=consts.for_name, ctx=ast.Load()),
                            args=[iter_],
                            keywords=[],
                        ),
                    ),
                    ast.Expr(
                        value=ast.ListComp(
                            elt=ast.Constant(value=None),
                            generators=[
                                ast.comprehension(
                                    target=target,
                                    iter=ast.Name(id=consts.for_obj_name, ctx=ast.Load()),
                                    ifs=[
                                        parse_sub(body, False, nonlocal_name)
                                    ],
                                    is_async=0,
                                )
                            ]
                        )
                    )
                ]

                if orelse:
                    new_node.append(
                        gen_orelse(orelse, nonlocal_name)
                    )

                nodes[pos:pos + 1] = new_node
                continue
            case ast.Break():
                exprs.append(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=consts.for_obj_name, ctx=ast.Load()),
                            attr='break_',
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    )
                )
            case ast.Continue():
                exprs.append(
                    ast.Constant(value=True)
                )
            case ast.While(test, body, orelse):
                body_ast = [
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=consts.while_obj_name, ctx=ast.Load()),
                                attr='cond',
                                ctx=ast.Load(),
                            ),
                            args=[test],
                            keywords=[],
                        )
                    )
                ]
                body_ast.extend(body)
                new_node = [
                    ast.Assign(
                        targets=[ast.Name(id=consts.while_obj_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id=consts.while_name, ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                    ),
                    ast.Expr(
                        value=ast.ListComp(
                            elt=ast.Constant(value=None),
                            generators=[
                                ast.comprehension(
                                    target=ast.Name(id='_', ctx=ast.Store()),
                                    iter=ast.Name(id=consts.while_obj_name, ctx=ast.Load()),
                                    ifs=[
                                        parse_sub(body_ast, False, nonlocal_name)
                                    ],
                                    is_async=0,
                                )
                            ]
                        )
                    )
                ]
                if orelse:
                    orelse.append(
                        gen_orelse(orelse, nonlocal_name)
                    )

                nodes[pos:pos + 1] = new_node
                continue
            case ast.Import(names):
                new_node = []
                for import_name in names:
                    name, asname = import_name.name, import_name.asname
                    if asname is None:
                        asname = name
                    new_node.append(
                        ast.Assign(
                            targets=[ast.Name(id=asname, ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Name(id='__import__', ctx=ast.Load()),
                                args=[
                                    ast.Constant(value=name),
                                ],
                                keywords=[],
                            ),
                        )
                    )
                nodes[pos:pos + 1] = new_node
                continue
            case ast.ImportFrom(module, names, _):
                new_node = []
                temp_node_var = get_temp_var()
                new_node.append(
                    ast.Assign(
                        targets=[ast.Name(id=temp_node_var, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id='__import__', ctx=ast.Load()),
                            args=[
                                ast.Constant(value=module),
                            ],
                            keywords=[
                                ast.keyword(
                                    arg='fromlist',
                                    value=ast.Constant(
                                        [
                                            module,
                                        ]
                                    ),
                                )
                            ],
                        ),
                    )
                )
                for import_name in names:
                    name, asname = import_name.name, import_name.asname
                    if asname is None:
                        asname = name
                    new_node.append(
                        ast.Assign(
                            targets=[ast.Name(id=asname, ctx=ast.Store())],
                            value=ast.Attribute(
                                value=ast.Name(id=temp_node_var, ctx=ast.Load()),
                                attr=name,
                                ctx=ast.Load(),
                            ),
                        )
                    )
                nodes[pos:pos + 1] = new_node
                continue
            case ast.Pass():
                exprs.append(
                    ast.Constant(value=...)
                )
            case ast.Raise(exc, cause):
                if cause is None:
                    exprs.append(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.GeneratorExp(
                                        elt=ast.Constant(value=0),
                                        generators=[
                                            ast.comprehension(
                                                target=ast.Name(id='_', ctx=ast.Store()),
                                                iter=ast.Tuple(elts=[], ctx=ast.Load()),
                                                ifs=[],
                                                is_async=0
                                            )
                                        ]
                                    ),
                                    attr='throw',
                                    ctx=ast.Load()
                                ),
                                args=[
                                    ast.Call(
                                        func=ast.Attribute(
                                            value=exc,
                                            attr='with_traceback',
                                            ctx=ast.Load()
                                        ),
                                        args=[
                                            ast.Attribute(
                                                value=exc,
                                                attr='__traceback__',
                                                ctx=ast.Load()
                                            )
                                        ] if exc else [
                                            ast.Constant(value=None)
                                        ],
                                        keywords=[]
                                    )
                                ],
                                keywords=[]
                            )
                        )
                    )
            case ast.Assert(test, msg):
                nodes[pos:pos + 1] = [
                    ast.If(
                        test=ast.UnaryOp(
                            op=ast.Not(),
                            operand=test,
                        ),
                        body=[
                            ast.Raise(
                                exc=ast.Call(
                                    func=ast.Name(id='AssertionError', ctx=ast.Load()),
                                    args=[
                                        msg,
                                    ],
                                    keywords=[],
                                )
                            )
                        ],
                        orelse=[
                            ast.Expr(
                                ast.Constant(value=None)
                            )
                        ],
                    )
                ]
                continue
            case ast.Delete(targets):
                new_node = []
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        new_node.append(
                            ast.Call(
                                func=ast.Name(
                                    id='delattr',
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    target.value,
                                    ast.Constant(value=target.attr),
                                ],
                                keywords=[],
                            )
                        )
                    elif isinstance(target, ast.Subscript):
                        new_node.append(
                            ast.Assign(
                                targets=[
                                    ast.Subscript(
                                        value=target.value,
                                        slice=ast.Slice(
                                            lower=target.slice.value,
                                            upper=target.slice.value + 1
                                        ),
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Constant(value=[]),
                            )
                        )
                    elif isinstance(target, ast.Name):
                        new_node.append(
                            ast.Expr(
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Call(
                                            func=ast.Name(id='locals', ctx=ast.Load()),
                                            args=[],
                                            keywords=[],
                                        ),
                                        attr='pop',
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.Constant(value=target.id),
                                    ],
                                    keywords=[],
                                )
                            )
                        )
                    else:
                        raise NotImplementedError(f"Not implemented for type {type(target)}")
                nodes[pos:pos + 1] = new_node
                continue
            case ast.Nonlocal(_):
                ...

            case _:
                raise NotImplementedError(f"Not implemented for type {type(node)}")

        pos += 1

    return ast.BoolOp(
        op=ast.Or(),
        values=exprs,
    )


def gen_orelse(orelse, use_nonlocal):
    return ast.Expr(
        value=ast.IfExp(
            test=ast.Attribute(
                value=ast.Name(id=consts.for_obj_name, ctx=ast.Load()),
                attr='_break',
                ctx=ast.Load(),
            ),
            body=ast.Constant(value=None),
            orelse=parse_sub(orelse, False, use_nonlocal),
        )
    )


def add_decorators(decorators, name):
    return [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=decorator,
                args=[
                    ast.Name(id=name, ctx=ast.Store()),
                ],
                keywords=[],
            ),
        )
        for decorator in decorators
    ]


def main():
    parser = argparse.ArgumentParser(description='Cover a python program to one expression')
    parser.add_argument('--input', type=str, help='The input file', required=True)
    parser.add_argument('--output', type=str, help='The output file', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'File {args.input} does not exist')
        return
    with open(args.input) as f:
        old_code = f.read()
    ast_tree = ast.parse(old_code)

    new_tree = parse_root(ast_tree)

    if os.path.exists(args.output):
        choice = input(f'File {args.output} already exist, overwrite it?(y/N)')
        if choice.lower() != 'y':
            return
    new_code = ast.unparse(new_tree)
    with open(args.output, 'w') as f:
        f.write(new_code)


if __name__ == '__main__':
    main()
