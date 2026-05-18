import ast
import argparse
import os.path
import sys
import tempfile
from pathlib import Path

import bundle
import trans


def _confirm_overwrite(path: str) -> bool:
    if not os.path.exists(path):
        return True
    choice = input(f'File {path} already exist, overwrite it?(y/N)')
    return choice.lower() == 'y'


def _transform_file(input_path: str, output_path: str, replace_name: str = 'none') -> None:
    with open(input_path, encoding='utf-8') as f:
        old_code = f.read()
    ast_tree = ast.parse(old_code)
    new_tree = trans.parse_root(ast_tree, replace_name=replace_name)
    new_code = ast.unparse(new_tree)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_code)


def main():
    parser = argparse.ArgumentParser(
        description='Squash a Python program into a single expression. '
                    'With --bundle, first flattens a multi-module package '
                    'into a single file, then squashes that.'
    )
    parser.add_argument('--input', type=str,
                        help='Input file (single-file mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file')
    parser.add_argument('--bundle', type=str, metavar='ROOT',
                        help='Bundle mode: source root containing the package')
    parser.add_argument('-p', '--package', type=str,
                        help='Bundle mode: package name to bundle')
    parser.add_argument('-e', '--entry', type=str,
                        help='Bundle mode: dotted entry module (e.g. pkg.__main__)')
    parser.add_argument('--bundle-only', action='store_true',
                        help='Bundle mode: stop after bundling (skip the '
                             'expression transform). Output is plain '
                             'single-file Python.')
    parser.add_argument('--replace-name', choices=['none', 'global'],
                        default='none',
                        help='Identifier mangling. "none" (default) keeps '
                             'helper class / member names readable. '
                             '"global" assumes the output is a self-'
                             'contained file and renames every helper '
                             'class + member to a fresh temp_N — useful '
                             'for obfuscation, breaks any tooling that '
                             'introspects the helpers by name.')
    args = parser.parse_args()

    if args.bundle is not None:
        if not args.package or not args.entry:
            parser.error('--bundle requires -p/--package and -e/--entry')
        if args.input is not None:
            parser.error('--bundle is incompatible with --input')
        _run_bundle(args)
    else:
        if args.input is None:
            parser.error('--input is required (or use --bundle for multi-module)')
        _run_single(args)


def _run_single(args) -> None:
    if not os.path.exists(args.input):
        print(f'File {args.input} does not exist')
        return
    if not _confirm_overwrite(args.output):
        return
    _transform_file(args.input, args.output, replace_name=args.replace_name)


def _run_bundle(args) -> None:
    if not _confirm_overwrite(args.output):
        return

    if args.bundle_only:
        # Bundle straight to the requested output, skip the expression pass.
        bundle.build(Path(args.bundle), args.package, args.entry, Path(args.output))
        return

    # Bundle to a temp file first, then run the onexpr transform on it.
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as tmp:
        tmp_path = tmp.name
    try:
        bundle.build(Path(args.bundle), args.package, args.entry, Path(tmp_path))
        _transform_file(tmp_path, args.output, replace_name=args.replace_name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == '__main__':
    main()
