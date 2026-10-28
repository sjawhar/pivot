import os
import pathlib
from difflib import get_close_matches
from typing import override

# Fuzzy matching constants
_FUZZY_CUTOFF = 0.6
_FUZZY_MIN_LENGTH = 3


def _fuzzy_suggest(query: str, candidates: list[str]) -> str | None:
    """Return best fuzzy match if found, else None."""
    if not candidates or len(query) < _FUZZY_MIN_LENGTH:
        return None
    matches = get_close_matches(query, candidates, n=1, cutoff=_FUZZY_CUTOFF)
    return matches[0] if matches else None


class PivotError(Exception):
    """Base exception for Pivot errors."""

    def format_user_message(self) -> str:
        """Format a user-friendly error message."""
        return str(self)

    def get_suggestion(self) -> str | None:
        """Return actionable suggestion for resolving the error."""
        return None


class ValidationError(PivotError):
    """Raised when stage validation fails."""


class StageDefinitionError(ValidationError):
    """Raised when a stage function has invalid annotations (return type, deps, etc)."""


class SecurityValidationError(PivotError):
    """Raised for security-sensitive validation failures (path traversal, injection attacks).

    Inherits from PivotError (not ValidationError) to ensure security errors
    are never accidentally caught by broad ValidationError handlers.
    """


class InvalidPathError(ValidationError):
    """Raised when a path is invalid (e.g., resolves outside project root)."""


class OutputPathConflictError(ValidationError):
    """Raised when two outputs resolve to the same workspace path, or one lies inside another."""

    @override
    def get_suggestion(self) -> str:
        return "Rename a stage, pipeline or output so each output has its own path"


class DAGError(PivotError):
    """Base class for DAG-related errors."""


class CyclicGraphError(DAGError):
    """Raised when DAG contains cycles."""

    @override
    def get_suggestion(self) -> str:
        return "Check stage dependencies for circular references"


class DependencyNotFoundError(DAGError):
    """Raised when a dependency doesn't exist."""

    _stage: str
    _dep: str
    _available: list[str]
    _input_path: pathlib.Path | None
    _input_path_display: str | None

    def __init__(
        self,
        stage: str,
        dep: str,
        available_outputs: list[str] | None = None,
        input_path: pathlib.Path | None = None,
        input_path_display: str | None = None,
    ) -> None:
        self._stage = stage
        self._dep = dep
        self._available = available_outputs or []
        self._input_path = input_path
        self._input_path_display = input_path_display
        if input_path is None:
            message = (
                f"Stage '{stage}' depends on '{dep}' which is not produced by any stage "
                + "and does not exist on disk"
            )
        else:
            display_path = input_path_display or str(input_path)
            message = (
                f"Stage '{stage}' depends on input '{dep}' (path {display_path}), "
                + "which does not exist"
            )
        super().__init__(message)

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def input_path(self) -> pathlib.Path | None:
        """The missing input's filesystem path; None for a missing stage output."""
        return self._input_path

    @override
    def format_user_message(self) -> str:
        msg = str(self)
        if match := _fuzzy_suggest(self._dep, self._available):
            msg += f"\n  Did you mean: '{match}'?"
        return msg

    @override
    def get_suggestion(self) -> str:
        return "Ensure the file exists or is produced by another stage"

    @override
    def __reduce__(
        self,
    ) -> tuple[type, tuple[str, str, list[str], pathlib.Path | None, str | None]]:
        return (
            self.__class__,
            (self._stage, self._dep, self._available, self._input_path, self._input_path_display),
        )


class MultipleDependencyError(DAGError):
    """Raised when multiple dependencies are missing."""

    _errors: list[DependencyNotFoundError]

    def __init__(self, errors: list[DependencyNotFoundError]) -> None:
        self._errors = errors
        super().__init__(f"{len(errors)} missing dependency error(s)")

    @property
    def errors(self) -> list[DependencyNotFoundError]:
        return self._errors

    @override
    def format_user_message(self) -> str:
        lines = [f"{len(self._errors)} missing dependency error(s):"]
        for err in self._errors:
            lines.append(f"  - {err.format_user_message()}")
        return "\n".join(lines)

    @override
    def get_suggestion(self) -> str:
        return "Ensure all dependency files exist or are produced by other stages"

    @override
    def __reduce__(self) -> tuple[type, tuple[list[DependencyNotFoundError]]]:
        return (self.__class__, (self._errors,))


class StageNotFoundError(DAGError):
    """Raised when a requested stage doesn't exist."""

    _unknown: list[str]
    _available: list[str]

    def __init__(
        self,
        unknown_stages: list[str],
        available_stages: list[str] | None = None,
    ) -> None:
        self._unknown = unknown_stages
        self._available = available_stages or []
        super().__init__(f"Unknown stage(s): {', '.join(unknown_stages)}")

    @override
    def format_user_message(self) -> str:
        msg = str(self)
        if self._available:
            # Limit to 3 suggestions to avoid overwhelming output
            suggestions = [
                f"'{u}' -> '{m}'"
                for u in self._unknown[:3]
                if (m := _fuzzy_suggest(u, self._available))
            ]
            if suggestions:
                msg += f"\n  Did you mean: {', '.join(suggestions)}?"
                if len(self._unknown) > 3:
                    msg += f"\n  (showing first 3 of {len(self._unknown)} unknown stages)"
        return msg

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot list' to see available stages"

    @override
    def __reduce__(self) -> tuple[type, tuple[list[str], list[str]]]:
        return (self.__class__, (self._unknown, self._available))


class PivotDBWriteTimeoutError(PivotError):
    """Raised when a StateDB write lock cannot be acquired in time."""

    @override
    def get_suggestion(self) -> str:
        return "Another pivot process may be holding the write lock. Check for zombie processes."


class ExecutionError(PivotError):
    """Raised when pipeline execution fails."""


class CacheError(PivotError):
    """Base class for cache-related errors."""


class OutputMissingError(CacheError):
    """Raised when a stage did not produce a declared output."""


class CacheRestoreError(CacheError):
    """Raised when restoring outputs from cache fails."""


class UncachedIncrementalOutputError(CacheError):
    """Raised when an incremental output exists without a cache entry."""


class CacheModeCollisionError(CacheError):
    """A HARDLINK/SYMLINK checkout shares the cache object's own permissions and found it
    already recorded with the wrong executable bit, with no COPY fallback available.

    This happens when two files with identical content disagree on their executable bit (the
    cache key is content-only) or when a cache object was populated by a path that does not
    know a file's intended executable bit (e.g. `pivot pull`, which downloads by hash alone).
    Pivot never changes a published cache object, so only a private copy, or the object deleted
    and published again from bytes that carry the right bit, can give the file its bit.
    """

    @override
    def get_suggestion(self) -> str:
        return (
            "Add CheckoutMode.COPY to checkout_modes (cache: checkout_mode: [..., copy] in "
            ".pivot/config.yaml) so checkout can fall back to a private, correctly-permissioned "
            "copy instead of sharing the cache object's own permissions; or delete the cache "
            "object the error names, then re-run what produced the file (`pivot repro --force "
            "<stage>`, `pivot track --force <path>`) to publish it with the right bit; until it "
            "is published again, any workspace file symlinked to it is broken"
        )


class ParamsError(PivotError):
    """Raised when parameter validation or loading fails."""


class PlotsError(PivotError):
    """Raised when plot processing fails."""


class TrackedFileMissingError(CacheError):
    """Raised when .pvt tracked files are missing (user should run pivot checkout)."""

    _missing: list[str]
    _checkout_attempted: bool

    def __init__(self, missing_files: list[str], checkout_attempted: bool = False) -> None:
        self._missing = missing_files
        self._checkout_attempted = checkout_attempted
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        missing_list = "\n".join(f"  - {p}" for p in self._missing)
        if self._checkout_attempted:
            return (
                f"The following tracked files are missing:\n{missing_list}\n\n"
                + "These files are not in local cache."
            )
        return (
            f"The following tracked files are missing:\n{missing_list}\n\n"
            + "These files are tracked by Pivot but don't exist on disk."
        )

    @override
    def get_suggestion(self) -> str:
        if self._checkout_attempted:
            return "Run 'pivot pull' to fetch from remote storage"
        if len(self._missing) == 1:
            return f"Run 'pivot checkout {self._missing[0]}' to restore, or 'pivot checkout --only-missing' to restore all missing files"
        return "Run 'pivot checkout --only-missing' to restore missing files, or 'pivot run --checkout-missing' to restore and run"

    @override
    def __reduce__(self) -> tuple[type, tuple[list[str], bool]]:
        return (self.__class__, (self._missing, self._checkout_attempted))


class GetError(PivotError):
    """Base class for get command errors."""


class RevisionNotFoundError(GetError):
    """Raised when git revision cannot be resolved."""


class TargetNotFoundError(GetError):
    """Raised when target is not found at specified revision."""


class CacheMissError(GetError):
    """Raised when file cannot be retrieved from cache, git, or remote."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot pull' to fetch from remote, or re-run the stage to regenerate"


class RemoteError(PivotError):
    """Base class for remote storage errors."""


class RemoteNotFoundError(RemoteError):
    """Raised when a named remote doesn't exist in configuration."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot remote list' to see available remotes"


class RemoteConnectionError(RemoteError):
    """Raised when connection to remote storage fails."""

    @override
    def get_suggestion(self) -> str:
        return "Check network connection and remote credentials"


class RemoteUnreachableError(RemoteConnectionError):
    """Raised when a connection to the remote cannot be made after the retries: a DNS failure, a
    refused connection, a connect timeout, a TLS failure or a proxy that cannot be connected to.
    Every object of the remote fails the same way, unlike a failure of one object on a remote
    that answered."""


class InvalidRemoteURLError(RemoteError):
    """Raised when remote URL is malformed or uses unsupported scheme."""

    @override
    def get_suggestion(self) -> str:
        return "Use format: s3://bucket/prefix or gs://bucket/prefix"


class RemoteTransferError(RemoteError):
    """Raised when file transfer to/from remote fails."""

    @override
    def get_suggestion(self) -> str:
        return "Check network connection and remote permissions"


class RemoteNotConfiguredError(RemoteError):
    """Raised when no remote is configured."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot config set remotes.<name> <url>' to configure a remote"


class ConfigError(PivotError):
    """Base class for configuration errors."""


class EmptyStateDirError(ConfigError):
    """Raised when a project config's ``core.state_dir`` is the empty string."""


class StateDirTargetError(ConfigError):
    """``core.state_dir`` names a path that cannot become a directory this process writes.

    A state-directory record keeps it as the root's resolution, with the outcome decided once
    with the value: the stages keep their state in ``.pivot``, and ``fallback`` says what a
    command that needs the configured directory gets. True: ``.pivot`` as well, so every reader
    uses ``.pivot``; it is true only where Pivot 0.5.2 ran such a command. False: ``path``,
    where the command fails as 0.5.2 failed. ``path`` is the directory the value names.
    """

    path: pathlib.Path
    fallback: bool

    def __init__(self, message: str, path: pathlib.Path, fallback: bool) -> None:
        super().__init__(message)
        self.path = path
        self.fallback = fallback

    @override
    def __reduce__(self) -> tuple[type, tuple[str, pathlib.Path, bool]]:
        return (self.__class__, (str(self), self.path, self.fallback))


class ConfigValidationError(ConfigError):
    """Raised when config value fails validation."""


class ConfigKeyError(ConfigError):
    """Raised when config key is unknown or invalid."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot config list' to see available config keys"


class InitError(PivotError):
    """Base class for initialization errors."""


class AlreadyInitializedError(InitError):
    """Raised when project is already initialized."""

    @override
    def get_suggestion(self) -> str:
        return "Use --force to reinitialize"


class ProjectNotInitializedError(InitError):
    """Raised when no .pivot directory exists above the current directory."""

    @override
    def get_suggestion(self) -> str:
        return "Run 'pivot init' in your project root to initialize Pivot"


class PipelineConfigError(PivotError):
    """Raised when pipeline configuration is invalid."""


class PipelineNotFoundError(PivotError):
    """Raised when no pipeline is found (no pipeline.py)."""

    @override
    def get_suggestion(self) -> str:
        return "Create pipeline.py to define your pipeline"


def describe_error(error: BaseException) -> str:
    """The text a user reads for ``error``, never a Python repr.

    A PivotError keeps its own message. An OSError is the operating system's message and the
    path or paths it names, with no exception class: ``<strerror>: <path>``,
    ``<strerror>: <path> -> <path2>``, or ``<strerror>`` when it names no path; one raised with
    its own message and no error number keeps that message. Any other exception is one pivot did
    not expect, named by its type: ``Unexpected <type>: <message>``.
    """
    message = str(error)
    if isinstance(error, PivotError):
        return message or type(error).__name__
    if isinstance(error, OSError):
        if error.strerror:
            names = (error.filename, error.filename2)
            paths = [_path_text(name) for name in names if name is not None]
            return f"{error.strerror}: {' -> '.join(paths)}" if paths else error.strerror
        if message:
            return message
    kind = type(error).__name__
    return f"Unexpected {kind}: {message}" if message else f"Unexpected {kind}"


def _path_text(name: object) -> str:
    """A path an OSError names, as text; one that is empty or would not print on one line is
    quoted with its escapes, as the exception's own text quotes it."""
    text = os.fsdecode(name) if isinstance(name, bytes) else str(name)
    return text if text and text.isprintable() else repr(text)
