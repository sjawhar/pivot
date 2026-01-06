#!/usr/bin/env python3
"""Generate type stubs for third-party packages without type annotations.

This script helps create and maintain type stubs in the `typings/` directory.

## Background

Some packages (loky, pygtrie, dvc) don't ship with type annotations or have
incomplete stubs. We create minimal, hand-tuned stubs covering only the APIs
we actually use, rather than generating comprehensive stubs for entire packages.

## Why Manual Stubs?

1. **Precision**: stubgen uses `Incomplete` (Any) for many types; we write exact types
2. **Minimal surface**: We only stub APIs we use (~30 lines vs 500+ from stubgen)
3. **Stability**: Third-party APIs we use rarely change
4. **Maintainability**: Small stubs are easy to update when needed

## Usage

    # Generate baseline stubs for a package (requires mypy installed)
    uv add --dev mypy
    python scripts/generate_stubs.py --generate loky

    # List current stub packages
    python scripts/generate_stubs.py --list

    # Validate stubs work with basedpyright
    python scripts/generate_stubs.py --validate

## Stub Location

Stubs live in `typings/<package>/` and are configured via pyproject.toml:

    [tool.basedpyright]
    stubPath = "typings"

## Adding a New Stub

1. Run stubgen to get a baseline (optional):
       stubgen -p <package> -o /tmp/stubs

2. Create `typings/<package>/__init__.pyi` with only the APIs you need

3. Run `basedpyright` to verify

## Example: Minimal Stub

Instead of stubgen's verbose output:

    def get_reusable_executor(max_workers=None, context=None, timeout=10, ...): ...

Write precise types:

    def get_reusable_executor(
        max_workers: int | None = None,
        context: BaseContext | None = None,
        timeout: int = 10,
        ...
    ) -> ProcessPoolExecutor: ...
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

TYPINGS_DIR = pathlib.Path(__file__).parent.parent / "typings"


def list_stubs() -> None:
    """List all stub packages in typings/."""
    if not TYPINGS_DIR.exists():
        print("No typings/ directory found")
        return

    packages = [p.name for p in TYPINGS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not packages:
        print("No stub packages found in typings/")
        return

    print("Stub packages in typings/:")
    for pkg in sorted(packages):
        stub_files = list((TYPINGS_DIR / pkg).glob("*.pyi"))
        print(f"  {pkg}/ ({len(stub_files)} stub files)")
        for f in stub_files:
            lines = len(f.read_text().splitlines())
            print(f"    - {f.name} ({lines} lines)")


def generate_baseline(package: str) -> None:
    """Generate baseline stubs using stubgen."""
    output_dir = pathlib.Path("/tmp/stubs")

    print(f"Generating baseline stubs for '{package}'...")
    print(f"Output directory: {output_dir}")
    print()

    # Check if stubgen is available
    check = subprocess.run(
        ["uv", "run", "python", "-c", "import mypy.stubgen"],
        capture_output=True,
    )
    if check.returncode != 0:
        print("stubgen requires mypy. Install it with:")
        print("  uv add --dev mypy")
        print()
        print("Then re-run this command.")
        sys.exit(1)

    result = subprocess.run(
        ["uv", "run", "stubgen", "-p", package, "-o", str(output_dir)],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    print(output)

    # Check if stubs were actually generated
    pkg_stub_dir = output_dir / package
    pkg_stub_file = output_dir / f"{package}.pyi"

    if not pkg_stub_dir.exists() and not pkg_stub_file.exists():
        print()
        print(f"No stubs generated for '{package}'.")
        print("This usually means stubgen couldn't import the package.")
        print()
        print("Try running manually to debug:")
        print(f"  python -c 'import {package}'")
        sys.exit(1)

    print()
    print("Baseline stubs generated. Now:")
    if pkg_stub_dir.exists():
        print(f"1. Review {pkg_stub_dir}/")
    else:
        print(f"1. Review {pkg_stub_file}")
    print(f"2. Copy relevant types to typings/{package}/__init__.pyi")
    print("3. Simplify to only the APIs you use")
    print("4. Add precise type annotations (replace Incomplete with real types)")
    print("5. Run: basedpyright")


def validate_stubs() -> None:
    """Validate stubs work with basedpyright."""
    print("Running basedpyright to validate stubs...")
    result = subprocess.run(["uv", "run", "basedpyright"], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("All stubs valid!")
    else:
        print("Stub validation failed")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage type stubs for third-party packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List current stub packages")
    parser.add_argument("--generate", metavar="PKG", help="Generate baseline stubs for a package")
    parser.add_argument("--validate", action="store_true", help="Validate stubs with basedpyright")

    args = parser.parse_args()

    if args.list:
        list_stubs()
    elif args.generate:
        generate_baseline(args.generate)
    elif args.validate:
        validate_stubs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
