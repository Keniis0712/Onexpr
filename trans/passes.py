import ast


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


def collect_user_names(tree: ast.AST) -> set:
    """Collect every identifier already used in the source tree, so that
    generated temp_N names won't collide with user code (e.g. a user
    function parameter that happens to be called `temp_1`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names
