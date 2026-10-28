from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import threading
import unicodedata
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, override

if TYPE_CHECKING:
    import pathspec

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


class _PatternInfo(NamedTuple):
    """Internal: pattern with its source location."""

    pattern: str
    source: str
    line_number: int


class IgnoreFilter:
    """Thread-safe filter over the project's ``.pivotignore``, reloaded when its mtime changes.

    Matching follows gitignore: a path is ignored when the last pattern that matches it is not a
    negation, or when a directory above it is ignored, which no negation undoes. This one rule
    decides ``pivot check-ignore``, the files a directory input's hash leaves out
    (``member_filter``) and the input changes watch mode skips. No user-level file is read: an
    ignore rule changes dependency hashes, and lock files carry those to every machine.
    """

    _lock: threading.Lock
    _project_root: pathlib.Path | None
    _spec: pathspec.PathSpec | None
    _pattern_infos: list[_PatternInfo]
    _project_mtime: float | None
    _warned_dirs: set[str]

    def __init__(self, project_root: pathlib.Path | None = None) -> None:
        self._lock = threading.Lock()
        self._project_root = project_root

        # Cached state
        self._spec = None
        self._pattern_infos = []
        self._project_mtime = None
        self._warned_dirs = set()

    @override
    def __reduce__(self) -> tuple[type[IgnoreFilter], tuple[pathlib.Path | None]]:
        """A copy (a store sent to a worker) is a filter for the same project that reads the
        file again; the lock and the loaded patterns stay with this process."""
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

        # Check protected paths first
        if self._is_protected(path_str):
            return False

        # Auto-detect is_dir if not provided
        if is_dir is None:
            is_dir = self._detect_is_dir(path)

        return _ignored_at(self._get_spec(), path_str, is_dir) is not None

    def check_ignore(self, path: str | pathlib.Path) -> CheckIgnoreResult:
        """Check if path is ignored and return detailed match info.

        Args:
            path: Path to check

        Returns:
            CheckIgnoreResult with path, ignored status, matching pattern, and source. When a
            directory above path is what is ignored, the pattern is the one that ignores it.
        """
        path_str = self._normalize_path(path)

        # Check protected paths first
        if self._is_protected(path_str):
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)

        spec, pattern_infos = self._snapshot()
        match_path = _ignored_at(spec, path_str, self._detect_is_dir(path))
        if match_path is None:
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)

        pattern, source = _deciding_pattern(pattern_infos, match_path)
        return CheckIgnoreResult(path=path_str, ignored=True, pattern=pattern, source=source)

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
        spec, pattern_infos = self._snapshot()
        if not spec.patterns or self._is_protected(directory_str):
            return None
        at_root = directory_str == "."
        match_path = None if at_root else _ignored_at(spec, directory_str, True)
        if match_path is not None:
            pattern, source = _deciding_pattern(pattern_infos, match_path)
            self._warn_ignored_directory(directory_str, pattern, source)
            return None

        prefix = "" if at_root else directory_str + "/"
        match_file = spec.match_file
        is_protected = self._is_protected

        def ignored(relpath: str, is_dir: bool) -> bool:
            member = prefix + unicodedata.normalize("NFC", relpath)
            if is_protected(member):
                return False
            return match_file(member + "/" if is_dir else member)

        return ignored

    def invalidate(self) -> None:
        """Force reload of patterns on next access."""
        with self._lock:
            self._spec = None
            self._pattern_infos = []
            self._project_mtime = None

    def _warn_ignored_directory(
        self, directory: str, pattern: str | None, source: str | None
    ) -> None:
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
            pattern,
            source,
        )

    def _snapshot(self) -> tuple[pathspec.PathSpec, list[_PatternInfo]]:
        """The compiled patterns and their sources, reloaded first if stale, read together."""
        with self._lock:
            if self._is_stale():
                self._reload_spec()
            assert self._spec is not None
            return self._spec, list(self._pattern_infos)

    def _get_spec(self) -> pathspec.PathSpec:
        """Get compiled PathSpec, reloading if stale."""
        with self._lock:
            if self._is_stale():
                self._reload_spec()
            assert self._spec is not None
            return self._spec

    def _is_stale(self) -> bool:
        """Check if cached patterns are outdated."""
        if self._spec is None:
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

    def _reload_spec(self) -> None:
        """Reload patterns from the project .pivotignore."""
        patterns: list[str] = []
        pattern_infos: list[_PatternInfo] = []

        if self._project_root is not None:
            project_path = self._project_root / ".pivotignore"
            patterns, pattern_infos, self._project_mtime = self._load_ignore_file(
                project_path, ".pivotignore"
            )
        else:
            self._project_mtime = None

        # Compile PathSpec
        try:
            self._spec = _compile(patterns)
        except ValueError as e:
            logger.warning(f"Failed to compile ignore patterns: {e}")
            self._spec = _compile([])

        self._pattern_infos = pattern_infos

    def _load_ignore_file(
        self, path: pathlib.Path, source_name: str
    ) -> tuple[list[str], list[_PatternInfo], float | None]:
        """Load patterns from an ignore file using fstat for atomic mtime.

        Returns:
            Tuple of (patterns, pattern_infos, mtime)
        """
        patterns: list[str] = []
        pattern_infos: list[_PatternInfo] = []
        mtime: float | None = None

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
                # Use fstat on open file handle for atomic mtime
                mtime = os.fstat(f.fileno()).st_mtime

            for line_num, line in enumerate(content.splitlines(), start=1):
                line = line.rstrip("\r\n")
                # Normalize Unicode to NFC (consistent with path normalization)
                line = unicodedata.normalize("NFC", line)

                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue

                # Validate pattern
                try:
                    _compile([line])
                    patterns.append(line)
                    pattern_infos.append(
                        _PatternInfo(pattern=line, source=source_name, line_number=line_num)
                    )
                except ValueError as e:
                    logger.warning(
                        f"Invalid pattern on line {line_num} of {source_name}: '{line}' - {e}"
                    )

        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Failed to read {source_name}: {e}")

        return patterns, pattern_infos, mtime

    def _normalize_path(self, path: str | pathlib.Path) -> str:
        """Normalize path: Unicode NFC, relative to project root (a backslash stays in a name)."""
        path_obj = path if isinstance(path, pathlib.Path) else pathlib.Path(path)

        # Convert absolute paths to relative (for watch mode which receives absolute paths)
        if self._project_root is not None and path_obj.is_absolute():
            with contextlib.suppress(ValueError):  # Path not under project root, use as-is
                path_obj = path_obj.relative_to(self._project_root)

        # Normalize Unicode to NFC (handles macOS NFD vs composed characters)
        return unicodedata.normalize("NFC", str(path_obj))

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


def _ignored_at(spec: pathspec.PathSpec, path_str: str, is_dir: bool) -> str | None:
    """The path gitignore's rule finds ignored for ``path_str``, or None if it is not ignored.

    That is the first directory above ``path_str`` the patterns match (with its trailing slash),
    else ``path_str`` itself (with a trailing slash for a directory) if they match it.
    """
    end = path_str.find("/", 1)
    while end != -1:
        parent = path_str[: end + 1]
        if spec.match_file(parent):
            return parent
        end = path_str.find("/", end + 1)
    match_path = path_str + "/" if is_dir and not path_str.endswith("/") else path_str
    return match_path if spec.match_file(match_path) else None


def _deciding_pattern(
    pattern_infos: list[_PatternInfo], match_path: str
) -> tuple[str | None, str | None]:
    """The last pattern, in declaration order, that matches ``match_path``, and its source."""
    for info in reversed(pattern_infos):
        pattern_to_check = info.pattern.removeprefix("!")
        try:
            single_spec = _compile([pattern_to_check])
        except ValueError:
            continue
        if single_spec.match_file(match_path):
            return info.pattern, f"{info.source}:{info.line_number}"
    return None, None


def _compile(lines: list[str]) -> pathspec.PathSpec:
    """Compile gitwildmatch patterns.

    pathspec is imported here rather than with this module: importing it costs about 10 ms in
    each process, and only a command that hashes a directory input, plans watch changes or
    runs ``check-ignore`` compiles patterns.
    """
    import pathspec

    return pathspec.PathSpec.from_lines("gitwildmatch", lines)
