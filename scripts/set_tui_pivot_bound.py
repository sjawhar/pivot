#!/usr/bin/env python3
"""Write pivot-tui's requirement on pivot as ``>=<version>``.

pivot-tui imports pivot modules (`pivot.types`, `pivot.display`), so the pivot-tui a release
publishes must require the pivot released with it: pivot-tui vX needs pivot>=vX. Both packages
take one version from ``[tool.semantic_release] version_toml``, so the bound is that version,
and a hand-written one is wrong as soon as another pull request releases first.

The release runs this from ``[tool.semantic_release] build_command``, between semantic-release's
version bump and its release commit, and stages the file, so every published pivot-tui carries
the bound of its own release.

## Usage

    python scripts/set_tui_pivot_bound.py              # version from $NEW_VERSION
    python scripts/set_tui_pivot_bound.py 1.2.3
    python scripts/set_tui_pivot_bound.py 1.2.3 --pyproject other/pyproject.toml
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import tomllib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_PYPROJECT = _REPO_ROOT / "packages" / "pivot-tui" / "pyproject.toml"

# A dependency line holding the requirement on pivot: `"pivot"`, `"pivot>=1.2.3"`, and so on.
# `pivot-tui` and `pivot_tui` are other distributions, so the name ends at the closing quote,
# a comparison operator, an extra or a marker.
_PIVOT_REQUIREMENT_LINE = re.compile(
    r'^(?P<indent>[ \t]*)"pivot(?![\w.\-])[^"]*"(?P<trailer>,?)[ \t]*$', re.MULTILINE
)


def set_bound(pyproject: pathlib.Path, version: str) -> str:
    """Rewrite ``pyproject``'s requirement on pivot to ``pivot>=version``; return the new text."""
    text = pyproject.read_text("utf-8")
    matches = list(_PIVOT_REQUIREMENT_LINE.finditer(text))
    if len(matches) != 1:
        found = [match.group(0).strip() for match in matches]
        raise SystemExit(
            f"{pyproject}: expected exactly one requirement on pivot, found {len(matches)}: {found}"
        )
    match = matches[0]
    requirement = f'{match["indent"]}"pivot>={version}"{match["trailer"]}'
    updated = text[: match.start()] + requirement + text[match.end() :]

    dependencies = tomllib.loads(updated)["project"]["dependencies"]
    if f"pivot>={version}" not in dependencies:
        raise SystemExit(f"{pyproject}: rewriting the requirement did not produce pivot>={version}")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        default=os.environ.get("NEW_VERSION", ""),
        help="the version being released; defaults to $NEW_VERSION, which semantic-release sets",
    )
    parser.add_argument(
        "--pyproject",
        type=pathlib.Path,
        default=_DEFAULT_PYPROJECT,
        help=f"the pyproject.toml to rewrite (default: {_DEFAULT_PYPROJECT})",
    )
    args = parser.parse_args(argv)
    if not args.version:
        parser.error("no version given and NEW_VERSION is unset")

    updated = set_bound(args.pyproject, args.version)
    args.pyproject.write_text(updated, "utf-8")
    print(f"{args.pyproject}: pivot>={args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
