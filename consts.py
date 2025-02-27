import ast

debug_mode = 1

for_name = '__onexpr_for_helper' if not debug_mode else 'for_cls'
for_class = ast.ClassDef(
    name=for_name,
    bases=[],
    keywords=[],
    body=[
        ast.FunctionDef(
            name='__init__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self'),
                    ast.arg(arg='iterator')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='iterator',
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Call(
                        func=ast.Name(id='iter', ctx=ast.Load()),
                        args=[ast.Name(id='iterator', ctx=ast.Load())],
                        keywords=[]
                    )
                ),
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='_break',
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Constant(value=False)
                )
            ],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='__iter__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]),
            body=[
                ast.Return(
                    value=ast.Name(id='self', ctx=ast.Load())
                )
            ],
            decorator_list=[],
            type_params=[]),
        ast.FunctionDef(
            name='__next__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]),
            body=[
                ast.If(
                    test=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr='_break',
                        ctx=ast.Load()),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='next', ctx=ast.Load()),
                                args=[
                                    ast.Call(
                                        func=ast.Name(id='iter', ctx=ast.Load()),
                                        args=[
                                            ast.List(elts=[], ctx=ast.Load())
                                        ],
                                        keywords=[]
                                    )
                                ],
                                keywords=[]
                            )
                        )
                    ],
                    orelse=[]
                ),
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id='next', ctx=ast.Load()),
                        args=[
                            ast.Attribute(
                                value=ast.Name(id='self', ctx=ast.Load()),
                                attr='iterator',
                                ctx=ast.Load()
                            )
                        ],
                        keywords=[]
                    )
                )
            ],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='break_',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]),
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='_break',
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Constant(value=True)
                ),
                ast.Return(
                    value=ast.Constant(value=True)
                )
            ],
            decorator_list=[],
            type_params=[]
        )
    ],
    decorator_list=[],
    type_params=[]
)
for_obj_name = '__onexpr_for_helper_obj' if not debug_mode else 'for_obj'

while_name = '__onexpr_while_helper' if not debug_mode else 'while_cls'
while_class = ast.ClassDef(
    name=while_name,
    bases=[],
    keywords=[],
    body=[
        ast.FunctionDef(
            name='__init__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='_break',
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Constant(value=False))],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='__iter__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Return(
                    value=ast.Name(id='self', ctx=ast.Load()))
            ],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='__next__',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.If(
                    test=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr='_break',
                        ctx=ast.Load()),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='next', ctx=ast.Load()),
                                args=[
                                    ast.Call(
                                        func=ast.Name(id='iter', ctx=ast.Load()),
                                        args=[
                                            ast.List(elts=[], ctx=ast.Load())
                                        ],
                                        keywords=[])
                                ],
                                keywords=[]
                            )
                        )
                    ],
                    orelse=[]
                ),
                ast.Return(
                    value=ast.Constant(value=None)
                )
            ],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='break_',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.Assign(
                    targets=[
                        ast.Attribute(
                            value=ast.Name(id='self', ctx=ast.Load()),
                            attr='_break',
                            ctx=ast.Store()
                        )
                    ],
                    value=ast.Constant(value=True)
                ),
                ast.Return(
                    value=ast.Constant(value=True)
                )
            ],
            decorator_list=[],
            type_params=[]
        ),
        ast.FunctionDef(
            name='cond',
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg='self'),
                    ast.arg(arg='cond')
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[
                ast.If(
                    test=ast.UnaryOp(
                        op=ast.Not(),
                        operand=ast.Name(id='cond', ctx=ast.Load())
                    ),
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='setattr', ctx=ast.Load()),
                                args=[
                                    ast.Name(id='self', ctx=ast.Load()),
                                    ast.Constant(value='_break'),
                                    ast.Constant(value=True)
                                ],
                                keywords=[]
                            )
                        ),
                        ast.Return(
                            value=ast.Constant(value=True)
                        )
                    ],
                    orelse=[]
                )
            ],
            decorator_list=[],
            type_params=[]
        )
    ],
    decorator_list=[],
    type_params=[]
)
while_obj_name = '__onexpr_while_helper_obj' if not debug_mode else 'while_obj'

nonlocal_inspect_name = '__onexpr_nonlocal_inspect' if not debug_mode else 'inspect_lib'
nonlocal_inspect_import = ast.Import(
    names=[
        ast.alias(
            name='inspect',
            asname=nonlocal_inspect_name
        )
    ]
)

temp_var_counter = 0


def get_temp_var():
    global temp_var_counter
    temp_var_counter += 1
    return f"__onexpr_temp_{temp_var_counter}" if not debug_mode else f"_v{temp_var_counter}"
