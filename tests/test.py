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
        source = '''
class A:
    def method(self):
        super().method()
'''
        expected = '''
class A:
    def method(self):
        super(A, self).method()
'''
        result_tree = get_transformed_ast(source)
        expected_tree = ast.parse(expected)
        self.assertEqual(
            ast.dump(result_tree),
            ast.dump(expected_tree)
        )

    def test_nested_classes(self):
        source = '''
class Outer:
    class Inner:
        def foo(self):
            super().foo()
'''
        expected = '''
class Outer:
    class Inner:
        def foo(self):
            super(Inner, self).foo()
'''
        result_tree = get_transformed_ast(source)
        expected_tree = ast.parse(expected)
        self.assertEqual(
            ast.dump(result_tree),
            ast.dump(expected_tree)
        )

    def test_async_function(self):
        source = '''
class A:
    async def coro(self):
        super().coro()
'''
        expected = '''
class A:
    async def coro(self):
        super(A, self).coro()
'''
        result_tree = get_transformed_ast(source)
        expected_tree = ast.parse(expected)
        self.assertEqual(
            ast.dump(result_tree),
            ast.dump(expected_tree)
        )

    def test_no_super_calls(self):
        source = '''
class A:
    def foo(self):
        print("no super here")
'''
        result_tree = get_transformed_ast(source)
        self.assertIn("print('no super here')", ast.unparse(result_tree))

    def test_missing_self_raises_error(self):
        source = '''
class A:
    def foo():
        super().foo()
'''
        with self.assertRaises(RuntimeError) as context:
            get_transformed_ast(source)
        self.assertIn("first argument name", str(context.exception))

    def test_missing_class_raises_error(self):
        # super() outside class — this should also raise
        source = '''
def func():
    super().foo()
'''
        with self.assertRaises(RuntimeError) as context:
            get_transformed_ast(source)
        self.assertIn("class context", str(context.exception))


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


if __name__ == '__main__':
    unittest.main()
