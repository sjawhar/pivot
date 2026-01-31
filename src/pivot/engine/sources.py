from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anyio
import watchfiles

from pivot.engine.types import CodeOrConfigChanged, DataArtifactChanged, RunRequested
from pivot.types import OnError

if TYPE_CHECKING:
    from pathlib import Path

    from anyio.streams.memory import MemoryObjectSendStream

    from pivot.engine.types import InputEvent

__all__ = ["FilesystemSource", "OneShotSource"]

_logger = logging.getLogger(__name__)

# watchfiles default debounce delay
_DEFAULT_DEBOUNCE_MS = 1600

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


class OneShotSource:
    """Async source that emits a single RunRequested event then exits.

    Used for 'pivot run' without --watch in the async engine. Emits the
    run request when run() is called, then returns.
    """

    _event: RunRequested

    def __init__(
        self,
        *,
        stages: list[str] | None,
        force: bool,
        reason: str,
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
        self._event = RunRequested(
            type="run_requested",
            stages=stages,
            force=force,
            reason=reason,
            single_stage=single_stage,
            parallel=parallel,
            max_workers=max_workers,
            no_commit=no_commit,
            no_cache=no_cache,
            on_error=on_error,
            cache_dir=cache_dir,
            allow_uncached_incremental=allow_uncached_incremental,
            checkout_missing=checkout_missing,
        )

    async def run(self, send: MemoryObjectSendStream[InputEvent]) -> None:
        """Emit single event then exit."""
        await send.send(self._event)


class FilesystemSource:
    """Async source that watches filesystem for changes using watchfiles.awatch()."""

    _watch_paths: list[Path]
    _debounce_ms: int | None
    _stop_event: anyio.Event | None
    _running: bool

    def __init__(
        self,
        *,
        watch_paths: list[Path],
        debounce_ms: int | None = None,
    ) -> None:
        """Initialize with paths to watch.

        Args:
            watch_paths: Directories/files to watch for changes.
            debounce_ms: Debounce delay in milliseconds. None uses watchfiles default (1600ms).
        """
        self._watch_paths = list(watch_paths)
        self._debounce_ms = debounce_ms
        self._stop_event = None
        self._running = False

    @property
    def watch_paths(self) -> list[Path]:
        """Current watch paths."""
        return list(self._watch_paths)

    def set_watch_paths(self, paths: list[Path]) -> None:
        """Update watched paths.

        If run() is currently active, this will restart the watcher with the new paths.
        """
        self._watch_paths = list(paths)
        # Signal current watcher to stop so it restarts with new paths
        if self._running and self._stop_event is not None:
            self._stop_event.set()

    async def run(self, send: MemoryObjectSendStream[InputEvent]) -> None:
        """Watch filesystem and emit change events.

        Restarts automatically if set_watch_paths() is called while running.
        """
        self._running = True
        try:
            while True:
                if not self._watch_paths:
                    # No paths to watch, wait for paths to be set
                    await anyio.sleep(1.0)
                    continue

                # Create fresh stop event for this watch iteration
                self._stop_event = anyio.Event()
                debounce = (
                    self._debounce_ms if self._debounce_ms is not None else _DEFAULT_DEBOUNCE_MS
                )

                async for changes in watchfiles.awatch(
                    *self._watch_paths,
                    debounce=debounce,
                    stop_event=self._stop_event,
                ):
                    code_paths = list[str]()
                    data_paths = list[str]()

                    for _change_type, path_str in changes:
                        if _is_code_or_config(path_str):
                            code_paths.append(path_str)
                        else:
                            data_paths.append(path_str)

                    if code_paths:
                        await send.send(
                            CodeOrConfigChanged(type="code_or_config_changed", paths=code_paths)
                        )
                    if data_paths:
                        await send.send(
                            DataArtifactChanged(type="data_artifact_changed", paths=data_paths)
                        )

                # awatch() exited - either stop_event was set or paths changed
                # Loop continues to restart with current paths
        finally:
            self._running = False
