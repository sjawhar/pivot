"""The project's ``.pivotignore``, read and matched as git reads and matches a root
``.gitignore``: a port of git's own code (``dir.c`` and ``wildmatch.c``, git 2.52).

- The file is bytes, a UTF-8 byte order mark at its start dropped. Lines split on ``\\n``; a
  ``\\r`` before it is dropped, and a line ends at a NUL, as a C string does. An empty line or
  one starting with ``#`` is skipped, and trailing spaces are dropped unless a backslash
  escapes them. A leading ``!`` negates the pattern; a trailing ``/`` makes it match
  directories only.
- A pattern with no other ``/`` matches an entry's own name at any depth. Any other pattern
  matches the whole path from the project root, a leading ``/`` dropped: its literal prefix (up
  to the first of ``*?[\\``) byte for byte, and wildmatch with ``WM_PATHNAME`` the rest, from
  the prefix's last byte. ``*``, ``?`` and ``[...]`` never match ``/``; ``**`` matches any
  number of directories only as a whole path component (``**/``, ``/**/``, a trailing
  ``/**``), and is a ``*`` anywhere else: ``foo**/bar`` ignores ``foo/bar`` and ``foox/bar``,
  not ``foobar`` or ``foo/x/bar``. A backslash makes the next byte literal.
- The last pattern that matches a path decides it. A path is ignored when that pattern is not
  a negation, or when a directory above it is ignored: git never looks inside an ignored
  directory, so no later negation can bring a file back from one.

Matching is on bytes and case-sensitive, as git's is with ``core.ignorecase`` false. On macOS,
where git precomposes the names it reads (``core.precomposeunicode``, true by default there), a
path is first put in Unicode NFC; a pattern never is. Wildmatch is ``dowild`` itself, not a
regular expression, so it gives up where git gives up and no pattern backtracks further than in
git (each step is Python's, not C's); consecutive whole ``**/`` components are first merged into
one, which matches the same paths. Only the project's own file is read: a ``.pivotignore`` in a
subdirectory is data like any other file.

Pivot departs from git in three places. ``pipeline.py`` and ``.pivot/`` at the project root are
never ignored (``PROTECTED_PATHS``). The project root itself is never ignored either, though
``git check-ignore .`` matches it, as an empty name, against a pattern such as ``*``; git's own
walk never tests it. And a line with more than ``_MAX_STAR_RUNS`` runs of ``*`` that make
``dowild`` recurse (``_recursing_runs``: not one at the end of the line, nor one that is not a
whole ``**`` right before a ``/``; after that merge) is not matched as git would, with a
warning: a negation is taken to match every path, so it keeps every file git might keep, and
any other line is not used.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import stat
import sys
import threading
import unicodedata
from collections.abc import Callable
from typing import NamedTuple, override

logger = logging.getLogger(__name__)

# Paths that can never be ignored regardless of patterns
PROTECTED_PATHS = frozenset({"pipeline.py", ".pivot/"})

# Default patterns for starter .pivotignore file
_DEFAULT_PATTERNS = [
    "# Python bytecode",
    "*.pyc",
    "*.pyo",
    "__pycache__/",
    "",
    "# Virtual environments",
    ".venv/",
    "venv/",
    "",
    "# Version control",
    ".git/",
    ".hg/",
    "",
    "# IDE/editors",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    "*~",
    ".#*",
    "",
    "# Build outputs",
    "*.egg-info/",
    "dist/",
    "build/",
    "node_modules/",
    "",
    "# Pivot internals",
    ".pivot/",
]

# git precomposes the names it reads on macOS (core.precomposeunicode defaults to true there).
_PRECOMPOSE = sys.platform == "darwin"


def get_default_patterns() -> list[str]:
    """Return default patterns for starter .pivotignore file."""
    return list(_DEFAULT_PATTERNS)


# Whether .pivotignore names one entry of a directory dependency, given its relpath under that
# directory and whether it is a directory. It tests that entry alone; ``member_ignored`` adds
# the directories above it.
type MemberFilter = Callable[[str, bool], bool]


def member_ignored(ignored: MemberFilter, relpath: str, *, is_dir: bool) -> bool:
    """Whether ``ignored`` names ``relpath`` or a directory above it.

    A walk that never enters an ignored directory (``cache.walk_artifact_members``) decides the
    same for every entry it meets; this decides it for a path met without that walk: a watched
    change, an entry of a recorded manifest.
    """
    end = relpath.find("/")
    while end != -1:
        if ignored(relpath[:end], True):
            return True
        end = relpath.find("/", end + 1)
    return ignored(relpath, is_dir)


class CheckIgnoreResult(NamedTuple):
    """Result of checking if a path is ignored."""

    path: str
    ignored: bool
    pattern: str | None
    source: str | None  # e.g., ".pivotignore:5"


class _Pattern(NamedTuple):
    """One ``.pivotignore`` line, parsed as git's ``parse_path_pattern`` parses it."""

    text: str  # the line as written, a byte that is not UTF-8 shown as a \x escape
    source: str  # e.g. ".pivotignore:5"
    negated: bool
    dir_only: bool  # a trailing "/": it matches a directory only
    basename: bool  # no "/" in it: it matches an entry's own name, at any depth
    match: Callable[[bytes], bool]  # tests the entry's name (basename) or its whole path


def _encode(path: str) -> bytes:
    """A path's bytes as git compares them (precomposed on macOS, as git reads names there)."""
    return os.fsencode(unicodedata.normalize("NFC", path) if _PRECOMPOSE else path)


def _deciding(patterns: tuple[_Pattern, ...], path: bytes, is_dir: bool) -> _Pattern | None:
    """The last pattern that matches ``path`` alone (git's ``last_matching_pattern_from_list``),
    if any."""
    name = path[path.rfind(b"/") + 1 :]
    for pattern in reversed(patterns):
        if pattern.dir_only and not is_dir:
            continue
        if pattern.match(name if pattern.basename else path):
            return pattern
    return None


def _excluding(patterns: tuple[_Pattern, ...], path: bytes, is_dir: bool) -> _Pattern | None:
    """The pattern that excludes ``path`` itself, or None when no pattern does."""
    deciding = _deciding(patterns, path, is_dir)
    return None if deciding is None or deciding.negated else deciding


def _ignored_at(patterns: tuple[_Pattern, ...], path_str: str, is_dir: bool) -> _Pattern | None:
    """The pattern that ignores ``path_str``: the one that ignores the first directory above it
    that is ignored, else the one that ignores it, else None (git's parent rule). The project
    root, ``.``, is never ignored: git's walk never tests it (the module docstring names how
    ``git check-ignore .`` differs)."""
    if path_str == ".":
        return None
    path = _encode(path_str)
    end = path.find(b"/", 1)
    while end != -1:
        if (excluding := _excluding(patterns, path[:end], True)) is not None:
            return excluding
        end = path.find(b"/", end + 1)
    return _excluding(patterns, path, is_dir)


class IgnoreFilter:
    """Thread-safe filter over the project's ``.pivotignore``, reloaded when its mtime changes.

    Matching is git's, as the module docstring says. This one rule decides ``pivot
    check-ignore``, the files a directory input's hash leaves out (``member_filter``) and the
    input changes watch mode skips. Hashing differs from ``check-ignore`` for an input
    directory that the file itself ignores, or that lies inside a directory it ignores:
    ``check-ignore`` reports its files ignored, as git does, but ``member_filter`` returns None
    and the input is hashed whole, since leaving out every file would hide every change to it.
    ``check-ignore`` also takes a relative path from the project root, not the working
    directory; the module docstring names where matching itself departs from git. No
    user-level file is read: an ignore rule changes dependency hashes, and lock files carry
    those to every machine.
    """

    _lock: threading.Lock
    _project_root: pathlib.Path | None
    _patterns: tuple[_Pattern, ...] | None
    _project_mtime: float | None
    _warned_dirs: set[str]

    def __init__(self, project_root: pathlib.Path | None = None) -> None:
        self._lock = threading.Lock()
        self._project_root = project_root

        # Cached state
        self._patterns = None
        self._project_mtime = None
        self._warned_dirs = set()

    @override
    def __reduce__(self) -> tuple[type[IgnoreFilter], tuple[pathlib.Path | None]]:
        """A copy (a store pickled for another process) is a filter for the same project that
        reads the file again; the lock and the loaded patterns stay with this process."""
        return IgnoreFilter, (self._project_root,)

    def is_ignored(self, path: str | pathlib.Path, *, is_dir: bool | None = None) -> bool:
        """Check if path should be ignored.

        Args:
            path: Path to check (relative to the project root, or absolute)
            is_dir: Override directory detection. If None, auto-detects.

        Returns:
            True if path, or a directory above it, is ignored and path is not protected.
        """
        path_str = self._normalize_path(path)
        if self._is_protected(path_str):
            return False
        if is_dir is None:
            is_dir = self._detect_is_dir(path)
        return _ignored_at(self._snapshot(), path_str, is_dir) is not None

    def check_ignore(self, path: str | pathlib.Path) -> CheckIgnoreResult:
        """Check if path is ignored and return detailed match info.

        Args:
            path: Path to check

        Returns:
            CheckIgnoreResult with path, ignored status, matching pattern, and source. When a
            directory above path is what is ignored, the pattern is the one that ignores it.
        """
        path_str = self._normalize_path(path)
        if self._is_protected(path_str):
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)
        excluding = _ignored_at(self._snapshot(), path_str, self._detect_is_dir(path))
        if excluding is None:
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)
        return CheckIgnoreResult(
            path=path_str, ignored=True, pattern=excluding.text, source=excluding.source
        )

    def member_filter(self, directory: str | pathlib.Path) -> MemberFilter | None:
        """The rule for the members of the directory dependency at ``directory``.

        ``directory`` is relative to the project root, or absolute under it. The filter takes a
        member's relpath under ``directory`` and whether it is a directory, and tests that entry
        alone (``member_ignored`` adds the directories above it). Patterns match the member's
        path inside the project, ``directory/relpath``, never a symlink target, so a directory
        linked in from outside the project gets the answer a local one gets. The patterns are
        read once here: the filter does not reload them.

        None when no member can be left out: ``.pivotignore`` holds no pattern, or it ignores
        ``directory`` itself or a directory above it. Such a directory is hashed whole, with a
        warning, since leaving out every file in it would hide every change to it.
        """
        directory_str = self._normalize_path(directory)
        patterns = self._snapshot()
        if not patterns or self._is_protected(directory_str):
            return None
        at_root = directory_str == "."
        excluding = None if at_root else _ignored_at(patterns, directory_str, True)
        if excluding is not None:
            self._warn_ignored_directory(directory_str, excluding)
            return None

        text_prefix = "" if at_root else directory_str + "/"
        prefix = b"" if at_root else _encode(directory_str) + b"/"
        is_protected = self._is_protected

        def ignored(relpath: str, is_dir: bool) -> bool:
            if is_protected(text_prefix + relpath):
                return False
            return _excluding(patterns, prefix + _encode(relpath), is_dir) is not None

        return ignored

    def invalidate(self) -> None:
        """Force reload of patterns on next access."""
        with self._lock:
            self._patterns = None
            self._project_mtime = None

    def _warn_ignored_directory(self, directory: str, excluding: _Pattern) -> None:
        """Warn once per directory that .pivotignore ignores it, so its hash keeps every file."""
        with self._lock:
            if directory in self._warned_dirs:
                return
            self._warned_dirs.add(directory)
        logger.warning(
            "Input directory %s is ignored by .pivotignore (pattern %r at %s), so Pivot hashes "
            + "every file in it: .pivotignore leaves files out only inside a directory it does "
            + "not ignore. Narrow the pattern to name only files inside it.",
            directory,
            excluding.text,
            excluding.source,
        )

    def _snapshot(self) -> tuple[_Pattern, ...]:
        """The parsed patterns, reloaded first if stale."""
        with self._lock:
            if self._is_stale():
                self._reload()
            assert self._patterns is not None
            return self._patterns

    def _is_stale(self) -> bool:
        """Check if cached patterns are outdated."""
        if self._patterns is None:
            return True

        if self._project_root is not None:
            project_path = self._project_root / ".pivotignore"
            try:
                current_mtime = project_path.stat().st_mtime
                if current_mtime != self._project_mtime:
                    return True
            except OSError:
                # File deleted, permissions changed, or other access error - reload to handle gracefully
                if self._project_mtime is not None:
                    return True

        return False

    def _reload(self) -> None:
        """Reload patterns from the project .pivotignore."""
        if self._project_root is None:
            self._patterns, self._project_mtime = (), None
            return
        self._patterns, self._project_mtime = _load_ignore_file(
            self._project_root / ".pivotignore", ".pivotignore"
        )

    def _normalize_path(self, path: str | pathlib.Path) -> str:
        """The path relative to the project root, as given otherwise (a backslash stays in a
        name; no Unicode normalization: git compares bytes)."""
        path_obj = path if isinstance(path, pathlib.Path) else pathlib.Path(path)

        # Convert absolute paths to relative (for watch mode which receives absolute paths)
        if self._project_root is not None and path_obj.is_absolute():
            with contextlib.suppress(ValueError):  # Path not under project root, use as-is
                path_obj = path_obj.relative_to(self._project_root)

        return str(path_obj)

    def _is_protected(self, path_str: str) -> bool:
        """Check if path is protected (should never be ignored)."""
        # Check exact match
        if path_str in PROTECTED_PATHS:
            return True

        # Check if path starts with protected directory
        for protected in PROTECTED_PATHS:
            if protected.endswith("/") and (
                path_str.startswith(protected) or path_str == protected.rstrip("/")
            ):
                return True

        return False

    def _detect_is_dir(self, path: str | pathlib.Path) -> bool:
        """Whether ``path`` is a directory, as git and the hash walk see it: by ``lstat``, so a
        symlink to a directory is not one. A relative path is taken from the project root, as
        matching takes it. A path that does not exist is a directory when written with a
        trailing ``/``."""
        path_obj = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        if self._project_root is not None and not path_obj.is_absolute():
            path_obj = self._project_root / path_obj
        try:
            return stat.S_ISDIR(path_obj.lstat().st_mode)
        except OSError:
            pass
        # Check original path for trailing slash (pathlib strips it during normalization)
        original = path if isinstance(path, str) else str(path)
        return original.endswith("/")


def _load_ignore_file(
    path: pathlib.Path, source_name: str
) -> tuple[tuple[_Pattern, ...], float | None]:
    """The patterns of the ignore file at ``path`` and its mtime (from fstat, so the two agree).

    The file is read as bytes, as git reads it, so no content can make it fail: a byte that is
    not UTF-8 matches the same byte in a name.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
            mtime = os.fstat(f.fileno()).st_mtime
    except FileNotFoundError:
        return (), None
    except OSError as e:
        logger.warning(f"Failed to read {source_name}: {e}")
        return (), None
    return _parse(data, source_name), mtime


def _parse(data: bytes, source_name: str) -> tuple[_Pattern, ...]:
    """Parse an ignore file's bytes as git's ``add_patterns_from_buffer`` and
    ``parse_path_pattern`` do."""
    data = data.removeprefix(b"\xef\xbb\xbf")
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    patterns = list[_Pattern]()
    for line_number, raw in enumerate(lines, start=1):
        # git ends a line at the CR before its LF, then reads it as a C string: up to a NUL
        line = _trim_trailing_spaces(raw.removesuffix(b"\r").partition(b"\0")[0])
        if not line or line.startswith(b"#"):
            continue
        text = line.decode("utf-8", "backslashreplace")
        negated = line.startswith(b"!")
        body = line[1:] if negated else line
        dir_only = body.endswith(b"/")
        if dir_only:
            body = body[:-1]
        basename = b"/" not in body
        if not basename:
            body = _merge_starstar_components(body.removeprefix(b"/"))
        compiled = _compile(body)
        if compiled is None:
            logger.warning(
                f"Invalid pattern on line {line_number} of {source_name}: '{text}' - git never "
                + "matches it (a lone trailing backslash, or an unclosed or unknown [...] class)"
            )
            continue
        classes, star_runs = compiled
        start = 0 if basename else _wildmatch_start(_literal_length(body))
        recursing = _recursing_runs(body, star_runs, start, not basename)
        if recursing <= _MAX_STAR_RUNS:
            match = _name_matcher(body, classes) if basename else _path_matcher(body, classes)
        elif negated:
            # Taken to match every path, it can only keep a file git would leave out (a rerun
            # too many), never leave out a file git keeps (a stage that should rerun)
            logger.warning(
                f"Negation on line {line_number} of {source_name} ('{text}') has {recursing} "
                + "runs of '*' that try more than one place; Pivot matches at most "
                + f"{_MAX_STAR_RUNS}, so it takes it to match every path. Shorten it."
            )
            match = _matches_everything
        else:
            logger.warning(
                f"Pattern on line {line_number} of {source_name} ('{text}') has {recursing} "
                + "runs of '*' that try more than one place; Pivot matches at most "
                + f"{_MAX_STAR_RUNS} and does not use it. Shorten it."
            )
            continue
        patterns.append(
            _Pattern(
                text=text,
                source=f"{source_name}:{line_number}",
                negated=negated,
                dir_only=dir_only,
                basename=basename,
                match=match,
            )
        )
    return tuple(patterns)


def _matches_everything(_path: bytes) -> bool:
    """The match of a negation too long to match as git does (``_MAX_STAR_RUNS``)."""
    return True


def _merge_starstar_components(body: bytes) -> bytes:
    """``body``, a pattern holding a ``/``, with each run of consecutive whole ``**/``
    components cut to one.

    A whole ``**/`` (one starting the pattern or a component, and followed by ``/``) matches
    any number of directories, none included, so ``**/**/`` matches what ``**/`` matches. Each
    one more costs ``dowild`` another level of trying every split of the path: 256 of them took
    minutes on one deep path. A run of ``*`` inside a class, after a backslash, or not filling
    its component is left as it is.
    """
    out = bytearray()
    size = len(body)
    index = 0
    component_start = True  # the next byte starts the pattern or a component
    after_starstar = False  # ``out`` ends with a whole "**" component and its "/"
    while index < size:
        byte = body[index]
        if byte == 0x2A:  # "*"
            end = index
            while end < size and body[end] == 0x2A:
                end += 1
            if end - index >= 2 and component_start and end < size and body[end] == 0x2F:
                if not after_starstar:
                    out += body[index : end + 1]
                after_starstar = True
                index = end + 1
                continue
            out += body[index:end]
            index = end
        elif byte == 0x5C:  # "\": it and the byte it escapes
            out += body[index : index + 2]
            index += 2
        elif byte == 0x5B and (compiled := _compile_class(body, index)) is not None:  # "["
            out += body[index : compiled.end + 1]
            index = compiled.end + 1
        else:
            out.append(byte)
            index += 1
            component_start = byte == 0x2F
            after_starstar = False
            continue
        component_start = False
        after_starstar = False
    return bytes(out)


def _trim_trailing_spaces(line: bytes) -> bytes:
    """git's ``trim_trailing_spaces``: drop trailing spaces a backslash does not escape."""
    last_space: int | None = None
    index = 0
    while index < len(line):
        byte = line[index]
        if byte == 0x20:
            if last_space is None:
                last_space = index
        elif byte == 0x5C:
            index += 1
            if index >= len(line):
                return line
            last_space = None
        else:
            last_space = None
        index += 1
    return line if last_space is None else line[:last_space]


# The bytes git's ``is_glob_special`` names: a literal prefix ends at the first of them.
_GLOB_SPECIAL = b"*?[\\"

# A line with more runs of "*" that recurse than this is not matched as git would (see the
# module docstring): each is one level of recursion in ``_dowild``, and Python's recursion
# limit is far below what git's C stack allows.
_MAX_STAR_RUNS = 256


def _literal_length(pattern: bytes, start: int = 0) -> int:
    """git's ``simple_length``: the index of the first of ``*?[\\`` at or after ``start``."""
    for index in range(start, len(pattern)):
        if pattern[index] in _GLOB_SPECIAL:
            return index
    return len(pattern)


def _name_matcher(body: bytes, classes: dict[int, _Class]) -> Callable[[bytes], bool]:
    """git's ``match_basename``, for a pattern with no ``/``: a pattern with no wildcard is
    compared whole, ``*`` and a literal tail is a suffix test (``PATTERN_FLAG_ENDSWITH``), and
    any other pattern is wildmatch without ``WM_PATHNAME``."""
    if _literal_length(body) == len(body):
        return lambda name: name == body
    if body[0] == 0x2A and _literal_length(body, 1) == len(body):
        tail = body[1:]
        return lambda name: name.endswith(tail)
    return lambda name: _dowild(body, classes, 0, name, 0, False) == _WM_MATCH


def _path_matcher(body: bytes, classes: dict[int, _Class]) -> Callable[[bytes], bool]:
    """git's ``match_pathname``, for a pattern holding a ``/`` (its leading ``/`` already
    dropped), against a path from the project root.

    The literal prefix (up to the first of ``*?[\\``) is compared as is, and wildmatch with
    ``WM_PATHNAME`` sees the rest and the prefix's last byte, as git 2.52 gives it (git commit
    1940a02dc1, "match_pathname(): give fnmatch one char of prefix context"): so a ``**`` right
    after a byte that is not ``/`` starts no whole component, and ``foo**/bar`` does not name
    ``foobar``. Before 2.52, git gave wildmatch no byte of the prefix, and that ``**`` spanned
    directories.
    """
    literal = _literal_length(body)
    if literal == len(body):
        return lambda path: path == body
    prefix = body[:literal]
    start = _wildmatch_start(literal)
    return lambda path: (
        path.startswith(prefix) and _dowild(body, classes, start, path, start, True) == _WM_MATCH
    )


def _wildmatch_start(literal: int) -> int:
    """Where git 2.52's ``match_pathname`` starts wildmatch in a pattern whose literal prefix is
    ``literal`` bytes long: at the prefix's last byte, so it sees where its components start."""
    return literal - 1 if literal else 0


class _Class(NamedTuple):
    """A ``[...]`` class of a wildmatch pattern: the bytes it names, whether it is negated, and
    the index of its closing ``]``."""

    members: frozenset[int]
    negated: bool
    end: int


def _compile(pattern: bytes) -> tuple[dict[int, _Class], list[tuple[int, int]]] | None:
    """The classes of a wildmatch pattern by the index of their ``[``, and the span of each of
    its runs of ``*``; or None when git never matches it: a lone trailing backslash, or a class
    that is unclosed or names an unknown ``[:name:]``.

    A match has to pass every element of the pattern, so wildmatch gives up on such a pattern
    (``WM_ABORT_ALL``) wherever it reaches the element, and nothing can match it. A class's
    members do not depend on the byte it is tested against, so each is read once here, as
    ``dowild`` reads it.
    """
    classes = dict[int, _Class]()
    star_runs = list[tuple[int, int]]()
    index = 0
    size = len(pattern)
    while index < size:
        byte = pattern[index]
        if byte == 0x5C:  # "\"
            if index + 1 >= size:
                return None
            index += 2
        elif byte == 0x5B:  # "["
            compiled = _compile_class(pattern, index)
            if compiled is None:
                return None
            classes[index] = compiled
            index = compiled.end + 1
        elif byte == 0x2A:  # "*"
            run_start = index
            while index < size and pattern[index] == 0x2A:
                index += 1
            star_runs.append((run_start, index))
        else:
            index += 1
    return classes, star_runs


def _recursing_runs(
    pattern: bytes, star_runs: list[tuple[int, int]], start: int, pathname: bool
) -> int:
    """How many of ``pattern``'s runs of ``*`` make ``_dowild`` recurse when it starts at
    ``start``, which bounds the depth of its recursion: each nested call starts after a later
    run.

    A run at the end of the pattern returns at once. With ``WM_PATHNAME``, a run that is not a
    whole ``**`` component and is followed by ``/`` jumps to the next ``/`` in the text. Every
    other run tries the rest of the pattern at each place in the text (a whole ``**/`` also
    tries matching no directory), one level deeper.
    """
    size = len(pattern)
    count = 0
    for run_start, run_end in star_runs:
        if run_end >= size:
            continue
        if pathname:
            after = pattern[run_end]
            whole = (
                run_end - run_start >= 2
                and (run_start == start or pattern[run_start - 1] == 0x2F)
                and (
                    after == 0x2F
                    or (after == 0x5C and run_end + 1 < size and pattern[run_end + 1] == 0x2F)
                )
            )
            if not whole and after == 0x2F:
                continue
        count += 1
    return count


# git's ASCII character classes (``sane_ctype``), as wildmatch's ``[:name:]`` tests them.
_ALNUM = frozenset(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_GRAPH = frozenset(range(0x21, 0x7F))
_CTYPE_CLASSES = {
    b"alnum": _ALNUM,
    b"alpha": frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    b"blank": frozenset(b" \t"),
    b"cntrl": frozenset([*range(0x20), 0x7F]),
    b"digit": frozenset(b"0123456789"),
    b"graph": _GRAPH,
    b"lower": frozenset(b"abcdefghijklmnopqrstuvwxyz"),
    b"print": frozenset(range(0x20, 0x7F)),
    b"punct": _GRAPH - _ALNUM,
    b"space": frozenset(b" \t\n\r"),  # not \v or \f: git's isspace is not C's
    b"upper": frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    b"xdigit": frozenset(b"0123456789ABCDEFabcdef"),
}


def _compile_class(pattern: bytes, start: int) -> _Class | None:
    """The ``[...]`` class at ``pattern[start]``, read as ``dowild``'s ``case '['`` reads it, or
    None where it gives up (an unclosed class, an unknown ``[:name:]``).

    A leading ``!`` or ``^`` negates it, its first byte is a member even when it is ``]``,
    ``a-z`` is a range of bytes, a backslash makes the next byte a member, and ``[`` not
    followed by ``:name:]`` is a member like any other byte.
    """
    size = len(pattern)
    p = start + 1
    p_ch = pattern[p] if p < size else 0
    if p_ch == 0x5E:  # "^" is "!"
        p_ch = 0x21
    negated = p_ch == 0x21
    if negated:
        p += 1
        p_ch = pattern[p] if p < size else 0
    members = set[int]()
    prev_ch = 0
    while True:
        if p_ch == 0:
            return None
        if p_ch == 0x5C:  # "\"
            p += 1
            p_ch = pattern[p] if p < size else 0
            if p_ch == 0:
                return None
            members.add(p_ch)
        elif p_ch == 0x2D and prev_ch and p + 1 < size and pattern[p + 1] != 0x5D:  # "a-z"
            p += 1
            p_ch = pattern[p]
            if p_ch == 0x5C:
                p += 1
                p_ch = pattern[p] if p < size else 0
                if p_ch == 0:
                    return None
            members.update(range(prev_ch, p_ch + 1))
            p_ch = 0  # the next byte starts no range
        elif p_ch == 0x5B and p + 1 < size and pattern[p + 1] == 0x3A:  # "[:"
            name_start = p + 2
            close = pattern.find(b"]", name_start)
            if close == -1:
                return None
            if close - name_start - 1 < 0 or pattern[close - 1] != 0x3A:
                members.add(0x5B)  # no ":]": the "[" is a member, and ":" is read next
            else:
                named = _CTYPE_CLASSES.get(pattern[name_start : close - 1])
                if named is None:
                    return None
                members.update(named)
                p = close
                p_ch = 0
        else:
            members.add(p_ch)
        prev_ch = p_ch
        p += 1
        p_ch = pattern[p] if p < size else 0
        if p_ch == 0x5D:  # "]"
            return _Class(frozenset(members), negated, p)


# wildmatch's results: a match, no match here, or no match anywhere a "*" (or, only past a
# "/", a "**") further out could try.
_WM_MATCH = 0
_WM_NOMATCH = 1
_WM_ABORT_ALL = -1
_WM_ABORT_TO_STARSTAR = -2


def _dowild(
    pattern: bytes, classes: dict[int, _Class], start: int, text: bytes, t: int, pathname: bool
) -> int:
    """git's ``dowild`` (wildmatch.c, git 2.52, as in 2.43 but for a refactor), line for line:
    ``pattern[start:]`` against ``text[t:]``, with ``WM_PATHNAME`` when ``pathname``.

    Its abort results are what keep it fast: a ``*`` that cannot match by taking the rest of the
    text tells every ``*`` further out to stop (``WM_ABORT_ALL``), so no pattern backtracks
    through every way of splitting the text, as a regular expression would. Each nested call
    starts after a later run of ``*``, so the depth is at most the pattern's runs that recurse
    (``_recursing_runs``, at most ``_MAX_STAR_RUNS``).
    """
    p = start
    size = len(pattern)
    end = len(text)
    while p < size:
        p_ch = pattern[p]
        t_ch = text[t] if t < end else 0
        if t_ch == 0 and p_ch != 0x2A:
            return _WM_ABORT_ALL
        if p_ch == 0x5C:  # "\": the next byte is literal
            p += 1
            if p >= size or t_ch != pattern[p]:
                return _WM_NOMATCH
        elif p_ch == 0x3F:  # "?": any byte but "/"
            if pathname and t_ch == 0x2F:
                return _WM_NOMATCH
        elif p_ch == 0x5B:  # "["
            members, negated, close = classes[p]
            if (t_ch in members) == negated or (pathname and t_ch == 0x2F):
                return _WM_NOMATCH
            p = close
        elif p_ch == 0x2A:  # "*"
            p += 1
            if p < size and pattern[p] == 0x2A:
                prev_p = p
                p += 1
                while p < size and pattern[p] == 0x2A:
                    p += 1
                if not pathname:
                    match_slash = True  # without WM_PATHNAME, "*" is "**"
                elif (prev_p - start < 2 or pattern[prev_p - 2] == 0x2F) and (
                    p >= size
                    or pattern[p] == 0x2F
                    or (pattern[p] == 0x5C and p + 1 < size and pattern[p + 1] == 0x2F)
                ):
                    # A whole "**" component: "**/" first tries matching no directory at all
                    if (
                        p < size
                        and pattern[p] == 0x2F
                        and _dowild(pattern, classes, p + 1, text, t, pathname) == _WM_MATCH
                    ):
                        return _WM_MATCH
                    match_slash = True
                else:
                    match_slash = False
            else:
                match_slash = not pathname
            if p >= size:
                # A trailing "**" matches everything; a trailing "*" only what holds no "/"
                if not match_slash and text.find(b"/", t) != -1:
                    return _WM_ABORT_TO_STARSTAR
                return _WM_MATCH
            if not match_slash and pattern[p] == 0x2F:
                # One "*" and then "/" matches the rest of this directory's name
                slash = text.find(b"/", t)
                if slash == -1:
                    return _WM_ABORT_ALL
                t = slash
            else:
                while True:
                    if t_ch == 0:
                        break
                    if pattern[p] not in _GLOB_SPECIAL:
                        # The text up to the next byte the literal after "*" names is the "*"'s
                        p_ch = pattern[p]
                        while True:
                            t_ch = text[t] if t < end else 0
                            if t_ch in (0, p_ch) or (not match_slash and t_ch == 0x2F):
                                break
                            t += 1
                        if t_ch != p_ch:
                            return _WM_ABORT_ALL if match_slash else _WM_ABORT_TO_STARSTAR
                    matched = _dowild(pattern, classes, p, text, t, pathname)
                    if matched != _WM_NOMATCH:
                        if not match_slash or matched != _WM_ABORT_TO_STARSTAR:
                            return matched
                    elif not match_slash and t_ch == 0x2F:
                        return _WM_ABORT_TO_STARSTAR
                    t += 1
                    t_ch = text[t] if t < end else 0
                return _WM_ABORT_ALL
        elif t_ch != p_ch:
            return _WM_NOMATCH
        p += 1
        t += 1
    return _WM_NOMATCH if t < end else _WM_MATCH
