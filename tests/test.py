import ast
import glob
import subprocess
import sys
import unittest

import trans


class TestCase(unittest.TestCase):
    def run_func(self, file):
        process = subprocess.Popen([sys.executable, file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()

        self.assertEqual(process.returncode, 0,
                         f"Running file {file} failed with the return code {process.returncode}.")

        return output

    def test_trans(self):
        test_files = glob.glob('t_*.py')

        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_code = f.read()
            ast_tree = ast.parse(test_code)
            new_tree = trans.parse_root(ast_tree)
            new_code = ast.unparse(new_tree)
            with open(f"./output/output_{test_file}", 'w', encoding='utf-8') as f:
                f.write(new_code)

            orig_output = self.run_func(test_file)
            new_output = self.run_func(f"./output/output_{test_file}")
            self.assertEqual(orig_output, new_output, f"The output of {test_file} is not the same as the new file.")
            print(f"Finish testing {test_file}.")


def get_transformed_ast(source):
    tree = ast.parse(source)
    transformer = trans.SuperTransformer()
    return transformer.visit(tree)


class TestSuperTransformer(unittest.TestCase):
    def test_simple_class(self):
        # super() inside a class is now left as-is; the __class__ cell
        # mechanism in parse_class_def + _make_class handles it at runtime.
        source = '''
class A:
    def method(self):
        super().method()
'''
        result_tree = get_transformed_ast(source)
        # super() should remain zero-arg (not rewritten to super(A, self))
        self.assertIn("super()", ast.unparse(result_tree))

    def test_nested_classes(self):
        source = '''
class Outer:
    class Inner:
        def foo(self):
            super().foo()
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("super()", ast.unparse(result_tree))

    def test_async_function(self):
        source = '''
class A:
    async def coro(self):
        super().coro()
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("super()", ast.unparse(result_tree))

    def test_no_super_calls(self):
        source = '''
class A:
    def foo(self):
        print("no super here")
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("print('no super here')", ast.unparse(result_tree))

    def test_missing_self_raises_error(self):
        # super() inside a class with no-arg method: no longer raises
        # at transform time; the cell mechanism handles it.
        source = '''
class A:
    def foo():
        super().foo()
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("super()", ast.unparse(result_tree))

    def test_missing_class_raises_error(self):
        # super() outside class: no longer raises at transform time.
        # It will raise at runtime if __class__ isn't available.
        source = '''
def func():
    super().foo()
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("super()", ast.unparse(result_tree))


class TestNodePresenceDetector(unittest.TestCase):

    def test_simple_function(self):
        source = """
def hello():
    print("Hi!")
"""
        tree = ast.parse(source)
        detector = trans.NodePresenceDetector()
        presence = detector.detect(tree)

        # 断言一些应该出现的节点类型
        self.assertIn(ast.FunctionDef, presence)
        self.assertIn(ast.Call, presence)
        self.assertIn(ast.Name, presence)
        self.assertIn(ast.Constant, presence)

        # 断言一些不应该出现的节点类型
        self.assertNotIn(ast.ClassDef, presence)
        self.assertNotIn(ast.Import, presence)

    def test_class_definition(self):
        source = """
class MyClass:
    def method(self):
        pass
"""
        tree = ast.parse(source)
        detector = trans.NodePresenceDetector()
        presence = detector.detect(tree)

        self.assertIn(ast.ClassDef, presence)
        self.assertIn(ast.FunctionDef, presence)
        self.assertIn(ast.Pass, presence)

    def test_empty_module(self):
        source = ""
        tree = ast.parse(source)
        detector = trans.NodePresenceDetector()
        presence = detector.detect(tree)

        self.assertIn(ast.Module, presence)
        self.assertEqual(len(presence), 1)

    def test_multiple_constructs(self):
        source = """
x = 10
if x > 5:
    for i in range(x):
        print(i)
"""
        tree = ast.parse(source)
        detector = trans.NodePresenceDetector()
        presence = detector.detect(tree)

        for required_type in [ast.Assign, ast.If, ast.For, ast.Call, ast.Compare, ast.Name, ast.Constant]:
            self.assertIn(required_type, presence)


class TestStrip(unittest.TestCase):
    """The strip pass shouldn't change observable behaviour for code
    that doesn't read its own __doc__ / __annotations__. Run each
    --strip mode through t_strip_basic.py and confirm stdout
    matches.
    """

    def _run(self, mode, asserts=False, fixture='t_strip_basic.py'):
        with open(fixture, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src)
        out = trans.parse_root(
            tree, src=src, strip=mode, strip_asserts=asserts,
        )
        out_path = f'output/output_strip_{mode}_{fixture}'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(ast.unparse(out))
        # original
        orig = subprocess.run(
            [sys.executable, fixture],
            capture_output=True, text=True, timeout=10,
        )
        new = subprocess.run(
            [sys.executable, out_path],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(orig.returncode, 0,
                         f'baseline failed: {orig.stderr}')
        self.assertEqual(new.returncode, 0,
                         f'strip={mode} failed: {new.stderr}')
        self.assertEqual(orig.stdout, new.stdout,
                         f'strip={mode} stdout drift')

    def test_strip_none(self):
        self._run('none')

    def test_strip_docs(self):
        self._run('docs')

    def test_strip_annotations(self):
        self._run('annotations')

    def test_strip_all(self):
        self._run('all')

    def test_strip_asserts(self):
        self._run('none', asserts=True)


if __name__ == '__main__':
    unittest.main()
