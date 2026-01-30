"""Event source implementations for the engine."""

from __future__ import annotations

import logging
import threading
from pathlib import Path as PathClass
from typing import TYPE_CHECKING

import watchfiles

from pivot.engine.types import CodeOrConfigChanged, DataArtifactChanged, RunRequested

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
    p = PathClass(path)
    return p.suffix in _CODE_FILE_SUFFIXES or p.name in _CONFIG_FILE_NAMES


class FilesystemSource:
    """Event source that watches filesystem for changes.

    Wraps watchfiles to detect file changes and emit DataArtifactChanged
    or CodeOrConfigChanged events.
    """

    _watch_paths: list[Path]
    _submit: Callable[[InputEvent], None] | None
    _running: bool
    _shutdown_event: threading.Event
    _watcher_thread: threading.Thread | None

    def __init__(self, watch_paths: list[Path]) -> None:
        """Initialize with paths to watch."""
        self._watch_paths = list(watch_paths)
        self._submit = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._watcher_thread = None

    @property
    def watch_paths(self) -> list[Path]:
        """Current paths being watched."""
        return list(self._watch_paths)

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
            # Only clear _submit if thread actually terminated
            if self._watcher_thread.is_alive():
                _logger.warning("FilesystemSource watcher thread did not terminate within timeout")
            else:
                self._watcher_thread = None
                self._submit = None

    def _watch_loop(self) -> None:
        """Watch for file changes and submit events."""
        for changes in watchfiles.watch(
            *self._watch_paths,
            stop_event=self._shutdown_event,
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
    _emitted: bool

    def __init__(
        self,
        stages: list[str] | None,
        force: bool,
        reason: str,
    ) -> None:
        """Initialize with run parameters.

        Args:
            stages: Stage names to run (None = all stages).
            force: If True, ignore cache and re-run.
            reason: Description of why this run was requested.
        """
        self._stages = stages
        self._force = force
        self._reason = reason
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
        )
        submit(event)
        self._emitted = True

    def stop(self) -> None:
        """No-op for one-shot source."""
        pass
