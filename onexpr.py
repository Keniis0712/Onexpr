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


def _transform_file(input_path: str, output_path: str,
                    replace_name: str = 'none',
                    strip: str = 'none',
                    strip_asserts: bool = False) -> None:
    with open(input_path, encoding='utf-8') as f:
        old_code = f.read()
    ast_tree = ast.parse(old_code)
    new_tree = trans.parse_root(
        ast_tree, replace_name=replace_name, src=old_code,
        strip=strip, strip_asserts=strip_asserts,
    )
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
    parser.add_argument('--replace-name', default='none',
                        help='Identifier mangling. CSV of tags: '
                             'helper (rename runtime helper class / '
                             'member names), toplevel (top-level def / '
                             'class names), imports (import bindings), '
                             'locals (function params + local vars), '
                             'methods (class method names + class-defined '
                             'self.X / cls.X attrs; consults mypy when '
                             'available to avoid mangling stdlib calls '
                             'that happen to share a method name), '
                             'attrs (every non-dunder Attribute.attr; '
                             'REQUIRES mypy — uses inferred receiver '
                             'types to leave stdlib API names alone). '
                             'Aliases: none (default), '
                             'safe = helper,toplevel,imports,locals,methods, '
                             'all = safe + attrs.')
    parser.add_argument('--strip', default='none',
                        help='Strip cosmetic AST elements. CSV of '
                             'tags: docs (drop module / class / '
                             'function docstrings), annotations '
                             '(drop function param + return '
                             'annotations and module / function-body '
                             'AnnAssign annotations; class-body '
                             'AnnAssigns are auto-preserved for '
                             'dataclass / pydantic). Aliases: none '
                             '(default), all = docs,annotations. '
                             'Use a `# obfuscate: keep` comment '
                             'above or on the same line as a node '
                             'to protect that single node from '
                             'stripping.')
    parser.add_argument('--strip-asserts', action='store_true',
                        help='Drop `assert` statements. Separate '
                             'flag because asserts have runtime '
                             'effects, unlike docstrings / '
                             'annotations.')
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
    _transform_file(
        args.input, args.output,
        replace_name=args.replace_name,
        strip=args.strip, strip_asserts=args.strip_asserts,
    )


def _run_bundle(args) -> None:
    if not _confirm_overwrite(args.output):
        return

    # User-code mangle tags (`locals`, `methods`, `attrs`) apply to
    # the original source files: doing them after bundling would
    # confuse mypy (it would see the synthetic _M_xxx functions, not
    # the user's classes). Pre-mangle in a temp tree, then bundle
    # that. `toplevel` / `imports` stay disabled in this stage —
    # they would rewrite the export surface the bundler relies on.
    from trans.mangle import expand_tags, premangle_package
    tags = expand_tags(args.replace_name)
    premangle_tags = tags & {'locals', 'methods', 'attrs'}

    src_root = args.bundle
    cleanup_premangle: list[str] = []
    if premangle_tags:
        premangle_dir = tempfile.mkdtemp(prefix='onexpr-mangle-')
        cleanup_premangle.append(premangle_dir)
        premangle_package(args.bundle, args.package, premangle_tags,
                          premangle_dir)
        src_root = premangle_dir

    # After pre-mangling user code we don't want the post-bundle
    # onexpr step to re-run user-code mangling — it would see the
    # synthetic _M_xxx functions and get confused. Restrict the
    # remaining tags to `helper` only.
    post_replace_name = 'helper' if 'helper' in tags else 'none'

    try:
        if args.bundle_only:
            # Bundle straight to the requested output, skip the
            # expression pass. Helper / toplevel / etc. were never
            # going to apply in --bundle-only either.
            bundle.build(Path(src_root), args.package, args.entry, Path(args.output))
            return

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, encoding='utf-8'
        ) as tmp:
            tmp_path = tmp.name
        try:
            bundle.build(Path(src_root), args.package, args.entry, Path(tmp_path))
            _transform_file(
                tmp_path, args.output,
                replace_name=post_replace_name,
                strip=args.strip, strip_asserts=args.strip_asserts,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    finally:
        import shutil
        for d in cleanup_premangle:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


if __name__ == '__main__':
    main()
