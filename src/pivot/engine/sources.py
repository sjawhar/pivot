"""Event source implementations for the engine."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import watchfiles

from pivot.engine.types import CodeOrConfigChanged, DataArtifactChanged, RunRequested
from pivot.types import OnError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pivot.engine.types import InputEvent

__all__ = ["FilesystemSource", "OneShotSource"]

_logger = logging.getLogger(__name__)

# File patterns that trigger code reload (same as watch/engine.py)
_CODE_FILE_SUFFIXES = (".py",)
_CONFIG_FILE_NAMES = (
    "pivot.yaml",
    "pivot.yml",
    "pipeline.py",
    "params.yaml",
    "params.yml",
    ".pivotignore",
)


def _is_code_or_config(path: str) -> bool:
    """Check if a path is a code or config file."""
    # Use string operations to avoid Path object overhead
    if path.endswith(_CODE_FILE_SUFFIXES):
        return True
    # Extract filename from path (everything after last /)
    slash_idx = path.rfind("/")
    name = path[slash_idx + 1 :] if slash_idx >= 0 else path
    return name in _CONFIG_FILE_NAMES


class FilesystemSource:
    """Event source that watches filesystem for changes.

    Wraps watchfiles to detect file changes and emit DataArtifactChanged
    or CodeOrConfigChanged events.
    """

    _watch_paths: list[Path]
    _debounce: int | None
    _submit: Callable[[InputEvent], None] | None
    _running: bool
    _shutdown_event: threading.Event
    _watcher_thread: threading.Thread | None

    def __init__(self, watch_paths: list[Path], debounce: int | None = None) -> None:
        """Initialize with paths to watch.

        Args:
            watch_paths: Directories/files to watch for changes.
            debounce: Debounce delay in milliseconds for watchfiles. None uses watchfiles default.
        """
        self._watch_paths = list(watch_paths)
        self._debounce = debounce
        self._submit = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._watcher_thread = None

    @property
    def watch_paths(self) -> list[Path]:
        """Current paths being watched."""
        return list(self._watch_paths)

    @property
    def debounce(self) -> int | None:
        """Debounce delay in milliseconds, or None for watchfiles default (1600ms)."""
        return self._debounce

    def set_watch_paths(self, paths: list[Path]) -> None:
        """Update watched paths.

        Called by engine when DAG changes and watch scope needs updating.
        Note: Changes only take effect on the next start(). If the watcher
        is currently running, stop() and start() must be called to apply
        the new paths.
        """
        self._watch_paths = list(paths)

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Begin producing events. Call submit() for each event."""
        self._submit = submit
        self._running = True
        self._shutdown_event.clear()

        # Don't spawn a thread if there are no paths to watch
        if not self._watch_paths:
            return

        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
        )
        self._watcher_thread.start()

    def stop(self) -> None:
        """Stop producing events."""
        self._running = False
        self._shutdown_event.set()

        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=3.0)
            if self._watcher_thread.is_alive():
                _logger.warning("FilesystemSource watcher thread did not terminate within timeout")
            self._watcher_thread = None

        # Always clear submit callback, even if no thread was started
        self._submit = None

    def _watch_loop(self) -> None:
        """Watch for file changes and submit events."""
        debounce = self._debounce if self._debounce is not None else 1600  # watchfiles default
        for changes in watchfiles.watch(
            *self._watch_paths,
            stop_event=self._shutdown_event,
            debounce=debounce,
        ):
            # Capture submit callback to avoid race with stop()
            submit = self._submit
            if submit is None:
                break

            # Classify changes as code/config or data
            code_paths = list[str]()
            data_paths = list[str]()

            for _change_type, path in changes:
                if _is_code_or_config(path):
                    code_paths.append(path)
                else:
                    data_paths.append(path)

            # Emit appropriate events
            if code_paths:
                submit(
                    CodeOrConfigChanged(
                        type="code_or_config_changed",
                        paths=code_paths,
                    )
                )

            if data_paths:
                submit(
                    DataArtifactChanged(
                        type="data_artifact_changed",
                        paths=data_paths,
                    )
                )


class OneShotSource:
    """Event source that emits a single RunRequested event.

    Used for 'pivot run' without --watch. Emits the run request
    immediately when start() is called, then becomes inactive.

    This source is single-use: once start() has been called and the
    event emitted, subsequent calls to start() will be no-ops.
    Create a new instance for each run request.
    """

    _stages: list[str] | None
    _force: bool
    _reason: str
    _single_stage: bool
    _parallel: bool
    _max_workers: int | None
    _no_commit: bool
    _no_cache: bool
    _on_error: OnError
    _cache_dir: Path | None
    _allow_uncached_incremental: bool
    _checkout_missing: bool
    _emitted: bool

    def __init__(
        self,
        stages: list[str] | None,
        force: bool,
        reason: str,
        *,
        single_stage: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        no_commit: bool = False,
        no_cache: bool = False,
        on_error: OnError = OnError.FAIL,
        cache_dir: Path | None = None,
        allow_uncached_incremental: bool = False,
        checkout_missing: bool = False,
    ) -> None:
        """Initialize with run parameters.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run.
            reason: Description of why this run was requested.
            single_stage: If True, run only the specified stages.
            parallel: If True, run stages in parallel.
            max_workers: Maximum worker processes.
            no_commit: If True, don't update lockfiles.
            no_cache: If True, disable run cache.
            on_error: Error handling mode.
            cache_dir: Directory for lock files.
            allow_uncached_incremental: Allow incremental outputs without cache.
            checkout_missing: Checkout missing dependency files from cache.
        """
        self._stages = stages
        self._force = force
        self._reason = reason
        self._single_stage = single_stage
        self._parallel = parallel
        self._max_workers = max_workers
        self._no_commit = no_commit
        self._no_cache = no_cache
        self._on_error = on_error
        self._cache_dir = cache_dir
        self._allow_uncached_incremental = allow_uncached_incremental
        self._checkout_missing = checkout_missing
        self._emitted = False

    def start(self, submit: Callable[[InputEvent], None]) -> None:
        """Emit a single RunRequested event."""
        if self._emitted:
            return

        event = RunRequested(
            type="run_requested",
            stages=self._stages,
            force=self._force,
            reason=self._reason,
            single_stage=self._single_stage,
            parallel=self._parallel,
            max_workers=self._max_workers,
            no_commit=self._no_commit,
            no_cache=self._no_cache,
            on_error=self._on_error,
            cache_dir=self._cache_dir,
            allow_uncached_incremental=self._allow_uncached_incremental,
            checkout_missing=self._checkout_missing,
        )
        submit(event)
        self._emitted = True

    def stop(self) -> None:
        """No-op for one-shot source."""
        pass
