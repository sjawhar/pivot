"""A directory input's filtered hash against git: for random trees and random pattern sets, the
files the hash keeps are the files ``git ls-files -o --exclude-standard`` keeps.

The seed and the number of cases are fixed, so a failure reproduces. Set
``PIVOT_IGNORE_GIT_CASES`` to run more cases (the review ran 5,100).
"""

from __future__ import annotations

import collections
import os
import random
import subprocess
from typing import TYPE_CHECKING

from pivot import ignore
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib

_SEED = 539
_CASES = int(os.environ.get("PIVOT_IGNORE_GIT_CASES", "1000"))
_INPUT = "data/raw/in"

_PATTERNS = [
    "*.pyc",
    "__pycache__/",
    "!keep.pyc",
    "build/",
    "!build/keep.txt",
    "**/foo/**",
    "!**/foo/keep.txt",
    "foo/",
    "/data/raw/in/sub/",
    "sub/*.txt",
    "*.txt",
    "!a.txt",
    "a*",
    "**/b",
    "x/**/y",
    "[ab].bin",
    "sub/",
    "!sub/",
    "!sub/a.txt",
    "*.pyc.txt",
    "**/__pycache__/**",
    "data/raw/in/foo",
    "in/",
    "*/keep.txt",
    "!*/keep.txt",
    "b/",
    "!b",
    "y",
    "**",
    "!**/",
    "*~",
    ".#*",
    "logs/**",
    "!logs/.gitkeep",
    "**/logs/**",
    "!**/logs/.gitkeep",
    "/data/raw/in/foo/**",
    "!/data/raw/in/foo/keep.txt",
    "deep/**/x",
    "x/**/",
    "[!a]*.bin",
    "?.txt",
    "\\#x",
    "**/",
    "***",
    "deep/*",
    "!deep/b",
    "*/",
    "!*/",
    "**/x/*",
    "sub/**",
    "!sub/**/",
    "data/raw/in/**/keep.txt",
    "a**b",
]
# A pattern and a later negation that undoes part of it, the case a matcher most often gets
# wrong (``X/**`` must not name X itself, or the parent rule keeps the negation from working).
_PAIRS = [
    ("**/foo/**", "!**/foo/keep.txt"),
    ("logs/**", "!logs/.gitkeep"),
    ("**/logs/**", "!**/logs/.gitkeep"),
    ("/data/raw/in/foo/**", "!/data/raw/in/foo/keep.txt"),
    ("sub/**", "!sub/a.txt"),
    ("x/**", "!x/keep.txt"),
    ("build/", "!build/keep.txt"),
    ("*.txt", "!a.txt"),
    ("sub/", "!sub/"),
    ("*/", "!*/"),
    ("deep/*", "!deep/b"),
    ("*.pyc", "!keep.pyc"),
]
_NAMES = [
    "a.txt",
    "x.pyc",
    "x.pyc.txt",
    "keep.pyc",
    "keep.txt",
    "a.bin",
    "c.bin",
    "b",
    "y",
    "f~",
    ".#lock",
    "plain",
    ".gitkeep",
    "#x",
    "azb",
]
_DIRS = ["__pycache__", "__pycache__x", "build", "foo", "sub", "x", "b", "y", "deep", "logs", "a"]


def _helper_tree(root: pathlib.Path, rng: random.Random, depth: int) -> None:
    """Random files and directories under ``root``, ``depth`` levels deep."""
    root.mkdir(parents=True, exist_ok=True)
    for name in rng.sample(_NAMES, rng.randint(0, 4)):
        (root / name).write_text(f"{root.name}/{name}\n")
    if depth > 0:
        for name in rng.sample(_DIRS, rng.randint(0, 3)):
            if not (root / name).is_file():
                _helper_tree(root / name, rng, depth - 1)


def _helper_git_kept(repo: pathlib.Path, home: pathlib.Path) -> dict[str, set[str]]:
    """By case directory, the files under its input that git keeps, each case's patterns being
    its own ``.gitignore``. No global or system git configuration is read."""
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True)
    listed = subprocess.run(
        ["git", "ls-files", "-o", "--exclude-standard", "-z"],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    ).stdout
    kept = collections.defaultdict[str, set[str]](set)
    for raw in listed.split(b"\0"):
        case, _, path = os.fsdecode(raw).partition("/")
        if path.startswith(f"{_INPUT}/"):
            kept[case].add(path.removeprefix(f"{_INPUT}/"))
    return kept


def test_an_input_hash_keeps_exactly_the_files_git_keeps(tmp_path: pathlib.Path) -> None:
    """The pruning walk, ``filter_listing`` over the whole listing and ``check_ignore`` all keep
    the files git keeps. A case whose input directory the patterns ignore is left out: Pivot
    hashes such an input whole, on purpose (``IgnoreFilter.member_filter``)."""
    rng = random.Random(_SEED)
    repo = tmp_path / "repo"
    patterns_by_case = dict[str, list[str]]()
    for index in range(_CASES):
        case = repo / f"c{index}"
        _helper_tree(case / _INPUT, rng, 3)
        (case / _INPUT / "always.dat").write_text("x\n")
        patterns = rng.sample(_PATTERNS, rng.randint(1, 6))
        if rng.random() < 0.5:
            # A negation only matters after a pattern it undoes, so put one such pair in order.
            patterns.extend(rng.choice(_PAIRS))
        text = "\n".join(patterns) + "\n"
        (case / ".pivotignore").write_text(text)
        (case / ".gitignore").write_text(text + "/.pivotignore\n/.gitignore\n")
        patterns_by_case[case.name] = patterns
    kept_by_git = _helper_git_kept(repo, tmp_path)

    compared = 0
    disagreements = list[str]()
    for name, patterns in patterns_by_case.items():
        case = repo / name
        rules = ignore.IgnoreFilter(case)
        member_filter = rules.member_filter(_INPUT)
        if member_filter is None:
            continue
        compared += 1
        git_kept = kept_by_git[name]
        walked = cache.hash_directory(case / _INPUT, ignored=member_filter)
        kept = {entry["relpath"] for entry in walked["listing"]["files"]}
        if kept != git_kept:
            disagreements.append(
                f"{name} {patterns}: pivot only {sorted(kept - git_kept)}, "
                + f"git only {sorted(git_kept - kept)}"
            )
        whole = cache.hash_directory(case / _INPUT)
        if cache.filter_listing(whole["listing"], member_filter) != walked:
            disagreements.append(f"{name} {patterns}: filter_listing differs from the walk")
        checked = {
            entry["relpath"]
            for entry in whole["listing"]["files"]
            if not rules.check_ignore(f"{_INPUT}/{entry['relpath']}").ignored
        }
        if checked != git_kept:
            disagreements.append(
                f"{name} {patterns}: check_ignore keeps {sorted(checked ^ git_kept)} unlike git"
            )

    assert compared > _CASES // 2, "most cases filter their input"
    assert disagreements == [], f"{len(disagreements)} disagreements: {disagreements[:10]}"
