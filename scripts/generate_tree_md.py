import os
import argparse


def generate_tree_markdown(
    root_dir: str,
    max_depth: int = None,
    include_files: bool = True,
    extensions: list[str] | None = None,
) -> str:
    """Generate a markdown-formatted directory tree."""
    root_dir = os.path.abspath(root_dir)
    lines = [f"# Directory tree for `{root_dir}`\n"]

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Path not found: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath.replace(root_dir, '').count(os.sep)
        if max_depth is not None and depth > max_depth:
            del dirnames[:]
            continue

        indent = '  ' * depth
        folder_name = os.path.basename(dirpath) or root_dir
        if depth == 0:
            lines.append(f"- **{folder_name}/**")
        else:
            lines.append(f"{indent}- **{folder_name}/**")

        if include_files:
            if extensions is not None:
                filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in extensions]
            for filename in sorted(filenames):
                lines.append(f"{indent}  - {filename}")

    return '\n'.join(lines) + '\n'


def write_markdown(
    root_dir: str,
    output_path: str,
    max_depth: int = None,
    include_files: bool = True,
    extensions: list[str] | None = None,
):
    content = generate_tree_markdown(root_dir, max_depth, include_files, extensions)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Markdown tree generated: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a Markdown directory tree.')
    parser.add_argument('root', nargs='?', default='.', help='Root directory to scan')
    parser.add_argument('-o', '--output', default='directory_tree.md', help='Output Markdown file')
    parser.add_argument('-d', '--depth', type=int, default=None, help='Max directory depth')
    parser.add_argument('--no-files', action='store_true', help='Exclude files from the tree')
    parser.add_argument(
        '-e', '--extensions',
        default=None,
        help='Comma-separated list of file extensions to include, e.g. .py,.json',
    )
    args = parser.parse_args()

    extensions = None
    if args.extensions:
        extensions = [ext.strip().lower() for ext in args.extensions.split(',') if ext.strip()]

    write_markdown(
        root_dir=args.root,
        output_path=args.output,
        max_depth=args.depth,
        include_files=not args.no_files,
        extensions=extensions,
    )
