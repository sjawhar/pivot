"""The project's ``.pivotignore``, read and matched exactly as git reads and matches a root
``.gitignore``.

git's rules (``dir.c`` and ``wildmatch.c``):

- The file is bytes. Lines split on ``\\n`` (a ``\\r`` before it is dropped); an empty line or one
  starting with ``#`` is skipped, and trailing spaces are dropped unless a backslash escapes
  them. A leading ``!`` negates the pattern; a trailing ``/`` makes it match directories only.
- A pattern with no other ``/`` matches an entry's own name at any depth. Any other pattern
  matches the whole path from the project root, a leading ``/`` dropped, by wildmatch with
  ``WM_PATHNAME``: ``*``, ``?`` and ``[...]`` never match ``/``; ``**`` matches any number of
  directories only as a whole path component (``**/``, ``/**/``, a trailing ``/**``), and is a
  ``*`` anywhere else. A backslash makes the next byte literal.
- The last pattern that matches a path decides it. A path is ignored when that pattern is not
  a negation, or when a directory above it is ignored: git never looks inside an ignored
  directory, so no later negation can bring a file back from one.

Matching is on bytes, as git matches. On macOS, where git precomposes the names it reads
(``core.precomposeunicode``, true by default there), a path is first put in Unicode NFC; a
pattern never is.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import re
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
    """One ``.pivotignore`` line, parsed as git parses a gitignore line."""

    text: str  # the line as written, a byte that is not UTF-8 shown as a \x escape
    source: str  # e.g. ".pivotignore:5"
    negated: bool
    dir_only: bool  # a trailing "/": it matches a directory only
    basename: bool  # no "/" in it: it matches an entry's own name, at any depth
    regex: re.Pattern[bytes]


def _encode(path: str) -> bytes:
    """A path's bytes as git compares them (precomposed on macOS, as git reads names there)."""
    return os.fsencode(unicodedata.normalize("NFC", path) if _PRECOMPOSE else path)


def _deciding(patterns: tuple[_Pattern, ...], path: bytes, is_dir: bool) -> _Pattern | None:
    """The last pattern that matches ``path`` alone (git's ``last_matching_pattern``), if any."""
    name = path[path.rfind(b"/") + 1 :]
    for pattern in reversed(patterns):
        if pattern.dir_only and not is_dir:
            continue
        if pattern.regex.fullmatch(name if pattern.basename else path) is not None:
            return pattern
    return None


def _excluding(patterns: tuple[_Pattern, ...], path: bytes, is_dir: bool) -> _Pattern | None:
    """The pattern that excludes ``path`` itself, or None when no pattern does."""
    deciding = _deciding(patterns, path, is_dir)
    return None if deciding is None or deciding.negated else deciding


def _ignored_at(patterns: tuple[_Pattern, ...], path_str: str, is_dir: bool) -> _Pattern | None:
    """The pattern that ignores ``path_str``: the one that ignores the first directory above it
    that is ignored, else the one that ignores it, else None (git's parent rule)."""
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
    input changes watch mode skips. The one difference is an input directory that the file
    itself ignores, or that lies inside a directory it ignores: ``check-ignore`` reports its
    files ignored, as git does, but ``member_filter`` returns None and the input is hashed
    whole, since leaving out every file would hide every change to it. No user-level file is
    read: an ignore rule changes dependency hashes, and lock files carry those to every machine.
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
        """Auto-detect if path is a directory."""
        try:
            path_obj = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
            if path_obj.exists():
                return path_obj.is_dir()
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
    """Parse an ignore file's bytes as git's ``add_patterns_from_buffer`` does."""
    data = data.removeprefix(b"\xef\xbb\xbf")
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    patterns = list[_Pattern]()
    for line_number, raw in enumerate(lines, start=1):
        line = _trim_trailing_spaces(raw.removesuffix(b"\r"))
        if not line or line.startswith(b"#"):
            continue
        text = line.decode("utf-8", "backslashreplace")
        negated = line.startswith(b"!")
        body = line[1:] if negated else line
        dir_only = body.endswith(b"/")
        if dir_only:
            body = body[:-1]
        basename = b"/" not in body
        if not basename and body.startswith(b"/"):
            body = body[1:]
        regex = _wildmatch_regex(body)
        if regex is None:
            logger.warning(
                f"Invalid pattern on line {line_number} of {source_name}: '{text}' - git never "
                + "matches it (a lone trailing backslash, or an unclosed or unknown [...] class)"
            )
            continue
        patterns.append(
            _Pattern(
                text=text,
                source=f"{source_name}:{line_number}",
                negated=negated,
                dir_only=dir_only,
                basename=basename,
                regex=re.compile(regex, re.DOTALL),
            )
        )
    return tuple(patterns)


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


def _wildmatch_regex(pattern: bytes) -> bytes | None:
    """git's ``wildmatch(pattern, subject, WM_PATHNAME)`` as a regular expression the whole
    subject must match, or None when no subject can match it.

    For a pattern matched against a name alone, which holds no ``/``, git uses wildmatch
    without ``WM_PATHNAME``; on such a subject the two agree.
    """
    out = list[bytes]()
    index = 0
    size = len(pattern)
    while index < size:
        byte = pattern[index]
        if byte == 0x5C:  # "\": the next byte is literal; a trailing one never matches
            if index + 1 >= size:
                return None
            out.append(re.escape(pattern[index + 1 : index + 2]))
            index += 2
        elif byte == 0x3F:  # "?"
            out.append(b"[^/]")
            index += 1
        elif byte == 0x2A:  # "*"
            start = index
            while index < size and pattern[index] == 0x2A:
                index += 1
            rest = pattern[index:]
            whole_component = (start == 0 or pattern[start - 1] == 0x2F) and (
                not rest or rest.startswith((b"/", b"\\/"))
            )
            if index - start == 1 or not whole_component:
                out.append(b"[^/]*")
            elif rest.startswith(b"/"):
                out.append(b"(?:.*/)?")  # "**/": no directory, or any number of them
                index += 1
            else:
                out.append(b".*")  # a trailing "**" (or "**" before an escaped "/")
        elif byte == 0x5B:  # "["
            bracket = _bracket_regex(pattern, index)
            if bracket is None:
                return None
            regex, index = bracket
            out.append(regex)
        else:
            out.append(re.escape(pattern[index : index + 1]))
            index += 1
    return b"".join(out)


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
    b"space": frozenset(b" \t\n\r"),
    b"upper": frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    b"xdigit": frozenset(b"0123456789ABCDEFabcdef"),
}


def _bracket_regex(pattern: bytes, start: int) -> tuple[bytes, int] | None:
    """The ``[...]`` class at ``pattern[start]`` as a regular expression, and the index after
    it, or None when git never matches it (unclosed, or an unknown ``[:name:]``).

    As wildmatch reads it: a leading ``!`` or ``^`` negates it, its first byte is a member even
    when it is ``]``, ``a-z`` is a range of bytes, a backslash makes the next byte literal, and
    it never matches ``/``.
    """
    size = len(pattern)
    index = start + 1
    negated = index < size and pattern[index] in b"!^"
    if negated:
        index += 1
    members = set[int]()
    previous = 0
    first = True
    while True:
        if index >= size:
            return None
        byte = pattern[index]
        if byte == 0x5D and not first:  # "]"
            break
        first = False
        if byte == 0x5C:
            index += 1
            if index >= size:
                return None
            byte = pattern[index]
            members.add(byte)
        elif byte == 0x2D and previous and index + 1 < size and pattern[index + 1] != 0x5D:
            index += 1
            high = pattern[index]
            if high == 0x5C:
                index += 1
                if index >= size:
                    return None
                high = pattern[index]
            members.update(range(previous, high + 1))
            byte = 0
        elif byte == 0x5B and index + 1 < size and pattern[index + 1] == 0x3A:  # "[:"
            close = pattern.find(b"]", index + 2)
            if close == -1:
                return None
            if close - (index + 2) - 1 < 0 or pattern[close - 1] != 0x3A:
                members.add(0x5B)  # no ":]": the "[" is a member, and ":" is read next
            else:
                named = _CTYPE_CLASSES.get(pattern[index + 2 : close - 1])
                if named is None:
                    return None
                members.update(named)
                index = close
                byte = 0
        else:
            members.add(byte)
        previous = byte
        index += 1
    matched = (set(range(1, 256)) - members) if negated else members
    matched.discard(0x2F)
    if not matched:
        return b"(?!)", index + 1
    return b"[" + b"".join(
        re.escape(bytes([member])) for member in sorted(matched)
    ) + b"]", index + 1
