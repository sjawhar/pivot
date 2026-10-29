"""A directory input's filtered hash against git: for random trees and generated ``.pivotignore``
bytes, the files the hash keeps are the files ``git ls-files -o --exclude-standard`` keeps.

Each pattern is built from tokens: literals, ``*``, ``**`` and ``?`` in every position, classes
(ranges, ``]`` first, ``!`` and ``^``, every ``[:name:]``, unknown and unclosed ones), escapes,
leading, middle and trailing ``/``, trailing spaces escaped or not, negations of a variant of an
earlier pattern, comments; the file gets CRLF endings, a byte order mark, a NUL inside a line or
no final newline now and then. Names hold spaces, tabs, ``\\v``, ``\\f``, wildcards, backslashes,
bytes that are not UTF-8, and NFC and NFD forms.

The seed and the number of cases are fixed, so a failure reproduces. Set
``PIVOT_IGNORE_GIT_CASES`` to run more cases.
"""

from __future__ import annotations

import collections
import os
import random
import subprocess
import unicodedata
from typing import TYPE_CHECKING

from pivot import ignore
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib

_SEED = 539
_CASES = int(os.environ.get("PIVOT_IGNORE_GIT_CASES", "1000"))
_INPUT = "data/raw/in"

_E_NFC = unicodedata.normalize("NFC", "é").encode()
_E_NFD = unicodedata.normalize("NFD", "é").encode()
_FILE_NAMES = [
    *(b"a", b"b", b"ab", b"abc", b"a.txt", b"b.txt", b"x.pyc", b"keep.txt", b"keep.pyc"),
    *(b"foo", b"foobar", b"fo", b"a b", b"a ", b" a", b"#c", b"!n", b"*", b"?", b"[a]", b"]"),
    *(b"-", b"a\\b", b"\\", _E_NFC, _E_NFD, b"caf" + _E_NFC, b"\xff", b"\xffx", b".hidden"),
    *(b"x~", b"A.TXT", b"a.txt ", b"z9", b"Z", b"_", b"a-b", b"a\tb", b"a\vb", b"\f", b"q?"),
    *(b".#l", b"x.pyc.txt", b"**", b"[ab]", b".gitkeep"),
]
_DIR_NAMES = [
    *(b"foo", b"foobar", b"fo", b"fox", b"a", b"b", b"sub", b"deep", b"x y", b"**", b"[ab]"),
    *(_E_NFC, b"__pycache__", b"build", b"logs", b"x", b"y", b"#d", b"!d", b"d "),
]
_COMPONENTS = [
    # literals
    *(b"a", b"b", b"ab", b"foo", b"fo", b"foobar", b"sub", b"deep", b"x", b"y", b"a.txt"),
    *(b"keep.txt", b"__pycache__", b"build", b"logs", b"x y", b"a b", _E_NFC, _E_NFD),
    *(b"caf" + _E_NFC, b"\xff", b"\xff*", b".gitkeep"),
    # stars in every position, a "**" right after a literal included
    *(b"*", b"**", b"***", b"*a", b"a*", b"f*o", b"fo*", b"fo**", b"foo**", b"**o", b"**oo"),
    *(b"a**b", b"*.txt", b"**.txt", b"*.pyc", b"x**", b"**x", b"a*b*", b"*-*-*"),
    # ?
    *(b"?", b"??", b"?.txt", b"a?", b"f?o"),
    # classes
    *(b"[ab]", b"[!a]", b"[^a]", b"[a-c]*", b"[]a]", b"[!]a]", b"[a-]", b"[-a]", b"[z-a]"),
    *(b"[[:alpha:]]*", b"[[:digit:][:alpha:]]*", b"[[:space:]]", b"a[[:space:]]b"),
    *(b"[[:punct:]]", b"[[:upper:]]*", b"[[:cntrl:]]", b"[[:blank:]]*", b"[[:xdigit:]]*"),
    *(b"[[:bogus:]]", b"[a", b"[\\]]", b"[\\a-c]", b"[[:alpha:]-z]", b"[[:al]", b"[[]"),
    *(b"[a-[:]", b"[!/]", b"*[ab]", b"[ab]*"),
    # escapes
    *(b"\\*", b"\\?", b"\\[a]", b"\\[ab]", b"a\\ ", b"\\#c", b"\\!n", b"\\\\", b"a\\\\b"),
    *(b"a\\b", b"\\a", b"f\\*o", b"\\**", b"*\\*"),
]
_PREFIXES = [
    *(b"", b"", b"", b"/", b"data/raw/in/", b"/data/raw/in/", b"**/", b"raw/in/"),
    *(b"data/**/", b"data/raw/in/**/", b"data/raw/i*/", b"d**/raw/in/", b"data**/"),
]
_SUFFIXES = [
    *(b"", b"", b"", b"", b"/", b"/**", b"/**/", b"**", b" ", b"  ", b"\\ ", b"\\  "),
    *(b" \\ ", b"\t"),
]


def _helper_pattern(rng: random.Random) -> bytes:
    """One random line: a prefix, one to three components joined by "/", and a suffix."""
    body = b"/".join(rng.choice(_COMPONENTS) for _ in range(rng.choice((1, 1, 1, 2, 2, 3))))
    line = rng.choice(_PREFIXES) + body + rng.choice(_SUFFIXES)
    roll = rng.random()
    if roll < 0.25:
        line = b"!" + line
    elif roll < 0.28:
        line = b"\\!" + line
    elif roll < 0.30:
        line = b"\\#" + line
    elif roll < 0.32:
        line = b"#" + line  # a comment
    return line


def _helper_ignore_bytes(rng: random.Random) -> bytes:
    """A random ignore file: lines, often a negation that undoes part of an earlier one."""
    lines = [_helper_pattern(rng) for _ in range(rng.randint(1, 6))]
    if rng.random() < 0.5:
        variant = rng.choice(lines).lstrip(b"!")
        lines.append(b"!" + variant + rng.choice([b"", b"/**", b"/keep.txt", b"/a", b"/*"]))
    if rng.random() < 0.05:
        at = rng.randrange(len(lines))
        lines[at] = lines[at] + b"\0" + rng.choice(_COMPONENTS)  # git reads a line up to a NUL
    eol = b"\r\n" if rng.random() < 0.15 else b"\n"
    data = eol.join(lines)
    if rng.random() < 0.8:
        data += eol
    if rng.random() < 0.1:
        data = b"\xef\xbb\xbf" + data
    return data


def _helper_tree(root: pathlib.Path, rng: random.Random, depth: int) -> None:
    """Random files and directories under ``root``, ``depth`` levels deep."""
    root.mkdir(parents=True, exist_ok=True)
    for name in rng.sample(_FILE_NAMES, rng.randint(0, 5)):
        path = root / os.fsdecode(name)
        if not path.exists():
            path.write_bytes(b"x")
    if depth > 0:
        for name in rng.sample(_DIR_NAMES, rng.randint(0, 3)):
            path = root / os.fsdecode(name)
            if not path.exists() or path.is_dir():
                _helper_tree(path, rng, depth - 1)


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
    the files git keeps. An input directory the patterns ignore is hashed whole, on purpose
    (``IgnoreFilter.member_filter``); there git keeps none of its files."""
    rng = random.Random(_SEED)
    repo = tmp_path / "repo"
    patterns_by_case = dict[str, bytes]()
    for index in range(_CASES):
        case = repo / f"c{index}"
        _helper_tree(case / _INPUT, rng, 3)
        (case / _INPUT / "always.dat").write_text("x\n")
        data = _helper_ignore_bytes(rng)
        (case / ".pivotignore").write_bytes(data)
        ending = b"" if data.endswith(b"\n") else b"\n"
        (case / ".gitignore").write_bytes(data + ending + b"/.pivotignore\n/.gitignore\n")
        patterns_by_case[case.name] = data
    kept_by_git = _helper_git_kept(repo, tmp_path)

    filtered = 0
    disagreements = list[str]()
    for name, data in patterns_by_case.items():
        case = repo / name
        rules = ignore.IgnoreFilter(case)
        git_kept = kept_by_git[name]
        whole = cache.hash_directory(case / _INPUT)
        files = {entry["relpath"] for entry in whole["listing"]["files"]}
        member_filter = rules.member_filter(_INPUT)
        if member_filter is None:
            expected = set[str]() if rules.check_ignore(_INPUT).ignored else files
            if git_kept != expected:
                disagreements.append(f"{name} {data!r}: hashed whole, git keeps {git_kept}")
            continue
        filtered += 1
        walked = cache.hash_directory(case / _INPUT, ignored=member_filter)
        kept = {entry["relpath"] for entry in walked["listing"]["files"]}
        if kept != git_kept:
            disagreements.append(
                f"{name} {data!r}: pivot only {sorted(kept - git_kept)}, "
                + f"git only {sorted(git_kept - kept)}"
            )
        if cache.filter_listing(whole["listing"], member_filter) != walked:
            disagreements.append(f"{name} {data!r}: filter_listing differs from the walk")
        checked = {
            relpath for relpath in files if not rules.check_ignore(f"{_INPUT}/{relpath}").ignored
        }
        if checked != git_kept:
            disagreements.append(
                f"{name} {data!r}: check_ignore keeps {sorted(checked ^ git_kept)} unlike git"
            )

    assert filtered > _CASES // 2, "most cases filter their input"
    assert disagreements == [], f"{len(disagreements)} disagreements: {disagreements[:10]}"
