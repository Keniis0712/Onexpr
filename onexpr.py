import ast
import argparse
import os.path

import trans


def main():
    parser = argparse.ArgumentParser(description='Cover a python program to one expression')
    parser.add_argument('--input', type=str, help='The input file', required=True)
    parser.add_argument('--output', type=str, help='The output file', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'File {args.input} does not exist')
        return
    with open(args.input, encoding='utf-8') as f:
        old_code = f.read()
    ast_tree = ast.parse(old_code)

    new_tree = trans.parse_root(ast_tree)

    if os.path.exists(args.output):
        choice = input(f'File {args.output} already exist, overwrite it?(y/N)')
        if choice.lower() != 'y':
            return
    new_code = ast.unparse(new_tree)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new_code)


if __name__ == '__main__':
    main()
