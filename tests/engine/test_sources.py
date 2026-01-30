"""Tests for event sources."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from pivot.engine import sources, types


def test_filesystem_source_instantiation() -> None:
    """FilesystemSource can be instantiated with watch paths."""
    source = sources.FilesystemSource(watch_paths=[Path("/tmp/test")])
    assert hasattr(source, "start")
    assert hasattr(source, "stop")


def test_filesystem_source_conforms_to_protocol() -> None:
    """FilesystemSource conforms to EventSource protocol."""
    source = sources.FilesystemSource(watch_paths=[])
    # Protocol conformance: has start(submit) and stop()
    _source: types.EventSource = source
    assert _source is source


def test_filesystem_source_set_watch_paths() -> None:
    """FilesystemSource.set_watch_paths() updates watched paths."""
    source = sources.FilesystemSource(watch_paths=[Path("/tmp/a")])

    new_paths = [Path("/tmp/b"), Path("/tmp/c")]
    source.set_watch_paths(new_paths)

    assert source.watch_paths == new_paths


def test_filesystem_source_watch_paths_property() -> None:
    """FilesystemSource.watch_paths returns current paths."""
    paths = [Path("/tmp/test1"), Path("/tmp/test2")]
    source = sources.FilesystemSource(watch_paths=paths)

    assert source.watch_paths == paths


# =============================================================================
# OneShotSource
# =============================================================================


def test_oneshot_source_emits_run_requested() -> None:
    """OneShotSource emits a single RunRequested event then stops."""
    events_received = list[types.InputEvent]()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)

    source = sources.OneShotSource(
        stages=["train", "evaluate"],
        force=True,
        reason="cli",
    )
    source.start(submit)

    assert len(events_received) == 1
    event = events_received[0]
    assert event["type"] == "run_requested"
    assert event["stages"] == ["train", "evaluate"]
    assert event["force"] is True
    assert event["reason"] == "cli"


def test_oneshot_source_with_none_stages() -> None:
    """OneShotSource with stages=None emits event with stages=None."""
    events_received = list[types.InputEvent]()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)

    source = sources.OneShotSource(stages=None, force=False, reason="test")
    source.start(submit)

    assert len(events_received) == 1
    event = events_received[0]
    assert event["type"] == "run_requested"
    assert event["stages"] is None


def test_oneshot_source_stop_is_noop() -> None:
    """OneShotSource.stop() is safe to call multiple times."""
    source = sources.OneShotSource(stages=None, force=False, reason="test")
    source.stop()  # Should not raise
    source.stop()  # Should not raise


def test_oneshot_source_conforms_to_protocol() -> None:
    """OneShotSource conforms to EventSource protocol."""
    source = sources.OneShotSource(stages=None, force=False, reason="test")
    # Protocol conformance: has start(submit) and stop()
    _source: types.EventSource = source
    assert _source is source


# =============================================================================
# FilesystemSource Watcher Tests
# =============================================================================


def test_filesystem_source_starts_watcher_thread(tmp_path: Path) -> None:
    """FilesystemSource.start() spawns a watcher thread."""
    watch_file = tmp_path / "data.csv"
    watch_file.touch()

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    # Give watcher time to start
    time.sleep(0.2)

    # Modify file to trigger event
    watch_file.write_text("new content")

    # Wait for event with timeout
    assert event_received.wait(timeout=2.0), "Timed out waiting for file change event"

    source.stop()

    # Should have received at least one event
    assert len(events_received) >= 1
    assert events_received[0]["type"] == "data_artifact_changed"


def test_filesystem_source_stop_terminates_watcher(tmp_path: Path) -> None:
    """FilesystemSource.stop() terminates the watcher thread."""
    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(lambda _: None)

    # Give watcher time to start
    time.sleep(0.1)

    source.stop()

    # Verify stop() is idempotent (can be called multiple times safely)
    # This implicitly verifies cleanup completed since repeated join() on
    # a non-terminated thread would hang or error
    source.stop()


def test_filesystem_source_emits_code_changed_for_python_files(tmp_path: Path) -> None:
    """FilesystemSource emits code_or_config_changed for .py files."""
    watch_file = tmp_path / "module.py"
    watch_file.touch()

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    # Give watcher time to start
    time.sleep(0.2)

    # Modify Python file to trigger event
    watch_file.write_text("# new content")

    # Wait for event with timeout
    assert event_received.wait(timeout=2.0), "Timed out waiting for file change event"

    source.stop()

    # Should have received code_or_config_changed event
    assert len(events_received) >= 1
    assert events_received[0]["type"] == "code_or_config_changed"


def test_filesystem_source_emits_code_changed_for_config_files(tmp_path: Path) -> None:
    """FilesystemSource emits code_or_config_changed for pivot.yaml."""
    config_file = tmp_path / "pivot.yaml"
    config_file.write_text("# initial")

    events_received = list[types.InputEvent]()
    event_received = threading.Event()

    def submit(event: types.InputEvent) -> None:
        events_received.append(event)
        event_received.set()

    source = sources.FilesystemSource(watch_paths=[tmp_path])
    source.start(submit)

    time.sleep(0.2)
    config_file.write_text("# modified")

    assert event_received.wait(timeout=2.0), "Timed out waiting for config change event"
    source.stop()

    code_events = [e for e in events_received if e["type"] == "code_or_config_changed"]
    assert len(code_events) >= 1
