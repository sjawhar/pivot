from __future__ import annotations

import logging
import os
import pathlib
import threading
import unicodedata
from typing import TYPE_CHECKING, NamedTuple

import pathspec

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

logger = logging.getLogger(__name__)

# Paths that can never be ignored regardless of patterns
PROTECTED_PATHS = frozenset({"pivot.yaml", "pivot.yml", ".pivot/"})

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


class CheckIgnoreResult(NamedTuple):
    """Result of checking if a path is ignored."""

    path: str
    ignored: bool
    pattern: str | None
    source: str | None  # e.g., ".pivotignore:5" or "~/.pivotignore:3"


class _PatternInfo(NamedTuple):
    """Internal: pattern with its source location."""

    pattern: str
    source: str
    line_number: int


class IgnoreFilter:
    """Thread-safe ignore filter with mtime-based caching.

    Loads patterns from project .pivotignore and user ~/.pivotignore files.
    Patterns are additive - a file is ignored if it matches ANY pattern.
    """

    _lock: threading.Lock
    _project_root: pathlib.Path | None
    _user_ignore_path: pathlib.Path
    _spec: pathspec.PathSpec | None
    _pattern_infos: list[_PatternInfo]
    _project_mtime: float | None
    _user_mtime: float | None

    def __init__(
        self,
        project_root: pathlib.Path | None = None,
        user_ignore_path: pathlib.Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._project_root = project_root
        self._user_ignore_path = user_ignore_path or pathlib.Path.home() / ".pivotignore"

        # Cached state
        self._spec = None
        self._pattern_infos = []
        self._project_mtime = None
        self._user_mtime = None

    def is_ignored(self, path: str | pathlib.Path, *, is_dir: bool | None = None) -> bool:
        """Check if path should be ignored.

        Args:
            path: Path to check (relative or absolute)
            is_dir: Override directory detection. If None, auto-detects.

        Returns:
            True if path matches any ignore pattern and is not protected.
        """
        path_str = self._normalize_path(path)

        # Check protected paths first
        if self._is_protected(path_str):
            return False

        # Auto-detect is_dir if not provided
        if is_dir is None:
            is_dir = self._detect_is_dir(path)

        spec = self._get_spec()

        # Add trailing slash for directory matching
        match_path = path_str
        if is_dir and not match_path.endswith("/"):
            match_path += "/"

        return spec.match_file(match_path)

    def check_ignore(self, path: str | pathlib.Path) -> CheckIgnoreResult:
        """Check if path is ignored and return detailed match info.

        Args:
            path: Path to check

        Returns:
            CheckIgnoreResult with path, ignored status, matching pattern, and source.
        """
        path_str = self._normalize_path(path)

        # Check protected paths first
        if self._is_protected(path_str):
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)

        # Auto-detect is_dir
        is_dir = self._detect_is_dir(path)

        # Use is_ignored() for the authoritative decision (single algorithm)
        ignored = self.is_ignored(path, is_dir=is_dir)

        # If not ignored, no need to find the pattern
        if not ignored:
            return CheckIgnoreResult(path=path_str, ignored=False, pattern=None, source=None)

        # Get pattern infos to find which pattern caused the match
        with self._lock:
            if self._is_stale():
                self._reload_spec()
            pattern_infos = list(self._pattern_infos)

        # Add trailing slash for directory matching
        match_path = path_str
        if is_dir and not match_path.endswith("/"):
            match_path += "/"

        # Find the last matching pattern in declaration order (the one that caused this path to be ignored)
        matched_pattern: str | None = None
        matched_source: str | None = None

        for info in reversed(pattern_infos):
            try:
                pattern_to_check = info.pattern
                is_negation = pattern_to_check.startswith("!")
                if is_negation:
                    pattern_to_check = pattern_to_check[1:]

                single_spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern_to_check])
                if single_spec.match_file(match_path):
                    # Found the last matching pattern
                    matched_pattern = info.pattern
                    matched_source = f"{info.source}:{info.line_number}"
                    break
            except ValueError:
                continue

        return CheckIgnoreResult(
            path=path_str,
            ignored=True,
            pattern=matched_pattern,
            source=matched_source,
        )

    def filter_entries(
        self,
        entries: Iterable[os.DirEntry[str]],
        base_path: pathlib.Path,
    ) -> Generator[os.DirEntry[str]]:
        """Filter directory entries, yielding non-ignored ones.

        Args:
            entries: Directory entries from os.scandir()
            base_path: Base path for computing relative paths

        Yields:
            Entries that are not ignored.
        """
        spec = self._get_spec()

        for entry in entries:
            try:
                rel_path = pathlib.Path(entry.path).relative_to(base_path)
            except ValueError:
                yield entry
                continue

            path_str = str(rel_path).replace(os.sep, "/")

            # Check protected paths
            if self._is_protected(path_str):
                yield entry
                continue

            # Add trailing slash for directories
            if entry.is_dir(follow_symlinks=False):
                path_str += "/"

            if not spec.match_file(path_str):
                yield entry

    def invalidate(self) -> None:
        """Force reload of patterns on next access."""
        with self._lock:
            self._spec = None
            self._pattern_infos = []
            self._project_mtime = None
            self._user_mtime = None

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

        # Check project .pivotignore
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

        # Check user .pivotignore
        try:
            current_mtime = self._user_ignore_path.stat().st_mtime
            if current_mtime != self._user_mtime:
                return True
        except OSError:
            # File deleted, permissions changed, or other access error - reload to handle gracefully
            if self._user_mtime is not None:
                return True

        return False

    def _reload_spec(self) -> None:
        """Reload patterns from all sources."""
        patterns: list[str] = []
        pattern_infos: list[_PatternInfo] = []

        # Load project .pivotignore
        if self._project_root is not None:
            project_path = self._project_root / ".pivotignore"
            project_patterns, project_infos, project_mtime = self._load_ignore_file(
                project_path, ".pivotignore"
            )
            patterns.extend(project_patterns)
            pattern_infos.extend(project_infos)
            self._project_mtime = project_mtime
        else:
            self._project_mtime = None

        # Load user .pivotignore
        user_patterns, user_infos, user_mtime = self._load_ignore_file(
            self._user_ignore_path, "~/.pivotignore"
        )
        patterns.extend(user_patterns)
        pattern_infos.extend(user_infos)
        self._user_mtime = user_mtime

        # Compile PathSpec
        try:
            self._spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except ValueError as e:
            logger.warning(f"Failed to compile ignore patterns: {e}")
            self._spec = pathspec.PathSpec.from_lines("gitwildmatch", [])

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
                # Normalize Windows path separators in patterns
                line = line.replace("\\", "/")
                # Normalize Unicode to NFC (consistent with path normalization)
                line = unicodedata.normalize("NFC", line)

                # Skip comments and blank lines
                if not line or line.startswith("#"):
                    continue

                # Validate pattern
                try:
                    pathspec.PathSpec.from_lines("gitwildmatch", [line])
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
        """Normalize path: forward slashes, Unicode NFC, relative to project root."""
        path_obj = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        path_str = str(path_obj).replace("\\", "/")

        # Convert absolute paths to relative (for watch mode which receives absolute paths)
        if self._project_root is not None and path_obj.is_absolute():
            try:
                rel_path = path_obj.relative_to(self._project_root)
                path_str = str(rel_path).replace("\\", "/")
            except ValueError:
                pass  # Path not under project root, use as-is

        # Normalize Unicode to NFC (handles macOS NFD vs composed characters)
        return unicodedata.normalize("NFC", path_str)

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
        return original.endswith("/") or original.endswith("\\")
