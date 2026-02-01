"""Tests for event sources."""

from __future__ import annotations

from pathlib import Path

from pivot.engine import sources, types

# =============================================================================
# OneShotSource
# =============================================================================


async def test_async_oneshot_source_emits_run_requested() -> None:
    """OneShotSource emits a single RunRequested event."""
    import anyio

    from pivot.engine.sources import OneShotSource
    from pivot.engine.types import InputEvent

    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    source = OneShotSource(
        stages=["train", "evaluate"],
        force=True,
        reason="cli",
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(source.run, send)

        # Receive the event
        event = await recv.receive()
        events_received.append(event)

        # Source should exit after one event
        tg.cancel_scope.cancel()

    assert len(events_received) == 1
    assert events_received[0]["type"] == "run_requested"
    assert events_received[0]["stages"] == ["train", "evaluate"]
    assert events_received[0]["force"] is True


def test_async_oneshot_source_conforms_to_protocol() -> None:
    """OneShotSource conforms to EventSource protocol."""
    source = sources.OneShotSource(stages=None, force=False, reason="test")
    # Protocol conformance: has async run(send)
    _source: types.EventSource = source
    assert _source is source


# =============================================================================
# FilesystemSource
# =============================================================================


async def test_async_filesystem_source_instantiation() -> None:
    """FilesystemSource can be instantiated with watch paths."""
    from pivot.engine.sources import FilesystemSource

    source = FilesystemSource(watch_paths=[Path("/tmp/test")])
    assert hasattr(source, "run")
    assert source.watch_paths == [Path("/tmp/test")]


async def test_async_filesystem_source_set_watch_paths() -> None:
    """FilesystemSource.set_watch_paths() updates watched paths."""
    from pivot.engine.sources import FilesystemSource

    source = FilesystemSource(watch_paths=[Path("/tmp/a")])
    new_paths = [Path("/tmp/b"), Path("/tmp/c")]
    source.set_watch_paths(new_paths)
    assert source.watch_paths == new_paths


async def test_async_filesystem_source_emits_code_changed_for_py_files(tmp_path: Path) -> None:
    """FilesystemSource emits code_or_config_changed for .py files."""
    import anyio

    from pivot.engine.sources import FilesystemSource
    from pivot.engine.types import InputEvent

    watch_file = tmp_path / "module.py"
    watch_file.write_text("# initial")

    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async def collect_events() -> None:
        async for event in recv:
            events_received.append(event)
            return  # Exit after first event

    async with anyio.create_task_group() as tg:
        source = FilesystemSource(watch_paths=[tmp_path], debounce_ms=100)
        tg.start_soon(source.run, send)
        tg.start_soon(collect_events)

        # Wait for watcher to start
        await anyio.sleep(0.2)

        # Modify Python file
        watch_file.write_text("# modified")

        # Wait for event
        await anyio.sleep(0.3)

        tg.cancel_scope.cancel()

    # Should have received code_or_config_changed event
    code_events = [e for e in events_received if e["type"] == "code_or_config_changed"]
    assert len(code_events) >= 1, "Should emit code_or_config_changed for .py files"


async def test_async_filesystem_source_set_watch_paths_signals_stop_event() -> None:
    """FilesystemSource.set_watch_paths() signals stop event when running."""
    from pivot.engine.sources import FilesystemSource

    source = FilesystemSource(watch_paths=[Path("/tmp/a")])

    # Simulate running state
    import anyio

    source._running = True
    source._stop_event = anyio.Event()

    # set_watch_paths should signal the stop event
    source.set_watch_paths([Path("/tmp/b")])

    assert source._stop_event.is_set(), "stop_event should be set when paths change while running"
    assert source.watch_paths == [Path("/tmp/b")]


async def test_async_filesystem_source_set_watch_paths_no_op_when_not_running() -> None:
    """FilesystemSource.set_watch_paths() doesn't signal stop event when not running."""
    from pivot.engine.sources import FilesystemSource

    source = FilesystemSource(watch_paths=[Path("/tmp/a")])

    # Not running, so stop_event should not be set
    assert source._running is False
    source.set_watch_paths([Path("/tmp/b")])

    # Should just update paths without any stop event issues
    assert source.watch_paths == [Path("/tmp/b")]


async def test_async_filesystem_source_restarts_after_set_watch_paths(tmp_path: Path) -> None:
    """FilesystemSource restarts and watches new paths after set_watch_paths() called."""
    import anyio

    from pivot.engine.sources import FilesystemSource
    from pivot.engine.types import InputEvent

    # Create initial and new directories
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    new_dir = tmp_path / "new"
    new_dir.mkdir()

    events_received = list[InputEvent]()
    send, recv = anyio.create_memory_object_stream[InputEvent](10)

    async def collect_events() -> None:
        async for event in recv:
            events_received.append(event)

    source = FilesystemSource(watch_paths=[initial_dir], debounce_ms=100)

    async with anyio.create_task_group() as tg:
        tg.start_soon(source.run, send)
        tg.start_soon(collect_events)

        # Wait for watcher to start on initial paths
        await anyio.sleep(0.2)

        # Change watch paths to new directory
        source.set_watch_paths([new_dir])

        # Give watcher time to restart with new paths
        await anyio.sleep(0.3)

        # Verify the paths were updated
        assert source.watch_paths == [new_dir], "Watch paths should be updated"

        # Create file in new directory
        new_file = new_dir / "test.txt"
        new_file.write_text("test content")

        # Wait for event
        with anyio.move_on_after(1.0):
            while not events_received:
                await anyio.sleep(0.1)

        tg.cancel_scope.cancel()

    # Should have received an event for the new file
    assert len(events_received) >= 1, "Should receive event for file in new watch path"
    data_events = [e for e in events_received if e["type"] == "data_artifact_changed"]
    assert len(data_events) >= 1, "Should emit data_artifact_changed for .txt file"
