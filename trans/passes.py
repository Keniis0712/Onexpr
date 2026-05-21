import ast


class SuperTransformer(ast.NodeTransformer):
    def __init__(self):
        self.class_stack = []
        # Stack of first-arg names of every function we're currently
        # visiting. Inner-function visits push and pop their own
        # entries so a nested zero-arg `def helper():` doesn't clobber
        # the outer method's `self` binding.
        self.arg_stack = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        # A class body is its own scope — nested zero-arg super()
        # calls there shouldn't see an outer function's first arg.
        self.arg_stack.append(None)
        self.generic_visit(node)
        self.arg_stack.pop()
        self.class_stack.pop()
        return node

    def visit_FunctionDef(self, node):
        self.arg_stack.append(self._first_arg(node))
        self.generic_visit(node)
        self.arg_stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node):
        self.arg_stack.append(self._first_arg(node))
        self.generic_visit(node)
        self.arg_stack.pop()
        return node

    def visit_Lambda(self, node):
        self.arg_stack.append(self._first_arg(node))
        self.generic_visit(node)
        self.arg_stack.pop()
        return node

    def _first_arg(self, node):
        # posonlyargs come before args; either could carry self/cls.
        if node.args.posonlyargs:
            return node.args.posonlyargs[0].arg
        if node.args.args:
            return node.args.args[0].arg
        return None

    @property
    def first_arg(self):
        # Walk outward to find the nearest function with a usable
        # first arg. A nested `def helper():` that doesn't take any
        # arguments shouldn't shadow the enclosing method's self.
        for a in reversed(self.arg_stack):
            if a is not None:
                return a
        return None

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name)
                and node.func.id == 'super'
                and len(node.args) == 0
                and len(node.keywords) == 0):
            if self.class_stack:
                # Inside a class: leave super() as-is. parse_class_def
                # injects a `__class__=None` kwonly default on the body
                # lambda so the compiler creates a __class__ cell; all
                # nested method lambdas closure-capture it. _make_class
                # fills the cell with the actual class object after
                # construction. This matches CPython's PEP 3135 semantics
                # without any static rewrite.
                return self.generic_visit(node)
            # Outside a class (e.g. a function defined at module level
            # that will later be monkey-patched into a class): we can't
            # determine the class statically. Leave super() as-is and
            # let the runtime raise if __class__ isn't available.
            return self.generic_visit(node)
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
        elif isinstance(node, ast.Attribute):
            # Pre-mangling can leave temp_N names as attribute names
            # (e.g. `self.temp_3 = ...`); the post-bundle onexpr pass
            # must not allocate the same name to a helper class.
            names.add(node.attr)
    return names
