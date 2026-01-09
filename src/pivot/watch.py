import fnmatch
import logging
import pathlib
from typing import TYPE_CHECKING

import watchfiles

from pivot import dag, executor, project, registry
from pivot.tui import console

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change


def run_watch_loop(
    stages: list[str] | None = None,
    single_stage: bool = False,
    cache_dir: pathlib.Path | None = None,
    watch_globs: list[str] | None = None,
    debounce_ms: int = 300,
) -> int:
    """Run pipeline in watch mode, returning number of re-runs."""
    # Determine which stages will actually be executed
    graph = registry.REGISTRY.build_dag(validate=True)
    stages_to_run = dag.get_execution_order(graph, stages, single_stage=single_stage)

    watch_paths = collect_watch_paths(stages_to_run)
    con = console.get_console()
    reloads = 0

    con.watch_start(watch_paths)

    while True:
        try:
            _run_pipeline_safely(con, stages, single_stage, cache_dir)
            con.watch_waiting()

            # Recreate filter each iteration to capture any new outputs from registry
            # Only filter outputs from stages that are actually being run
            watch_filter = create_watch_filter(stages_to_run, watch_globs)
            changes = _wait_for_changes(watch_paths, watch_filter, debounce_ms)
            con.watch_changes_detected(changes)

            # Wait for quiet period to ensure no more rapid changes are incoming
            _wait_for_quiet_period(watch_paths, watch_filter, debounce_ms)
            reloads += 1

        except KeyboardInterrupt:
            con.watch_stopped()
            return reloads


def _run_pipeline_safely(
    con: console.Console,
    stages: list[str] | None,
    single_stage: bool,
    cache_dir: pathlib.Path | None,
) -> None:
    """Run pipeline, logging errors without exiting."""
    try:
        executor.run(stages=stages, single_stage=single_stage, cache_dir=cache_dir)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        con.error(f"Pipeline failed: {e}")


def _wait_for_changes(
    watch_paths: list[pathlib.Path],
    watch_filter: "Callable[[Change, str], bool]",
    debounce_ms: int,
) -> "set[tuple[Change, str]]":
    """Block until file changes detected, return change set."""
    for changes in watchfiles.watch(
        *watch_paths,
        watch_filter=watch_filter,
        raise_interrupt=False,
        debounce=debounce_ms,
    ):
        return changes
    return set()  # Unreachable, but satisfies type checker


def _wait_for_quiet_period(
    watch_paths: list[pathlib.Path],
    watch_filter: "Callable[[Change, str], bool]",
    debounce_ms: int,
) -> None:
    """Wait until no more changes occur within the debounce window."""
    # Use non-blocking check with short timeout to see if more changes are pending
    timeout_seconds = debounce_ms / 1000.0
    while True:
        more_changes = list(
            watchfiles.watch(
                *watch_paths,
                watch_filter=watch_filter,
                raise_interrupt=False,
                debounce=debounce_ms,
                yield_on_timeout=True,
                rust_timeout=int(timeout_seconds * 1000),
            )
        )
        # If we got an empty yield (timeout with no changes), the quiet period is over
        if not more_changes or not more_changes[0]:
            break
        # More changes detected, continue waiting


def collect_watch_paths(stages: list[str]) -> list[pathlib.Path]:
    """Collect paths: project root + dependency directories for specified stages."""
    root = project.get_project_root()
    paths: set[pathlib.Path] = {root}
    for name in stages:
        try:
            info = registry.REGISTRY.get(name)
        except KeyError:
            logger.warning(f"Stage '{name}' not found in registry, skipping")
            continue
        for dep in info["deps"]:
            dep_path = project.try_resolve_path(dep)
            if dep_path is not None and dep_path.exists():
                paths.add(dep_path.parent if dep_path.is_file() else dep_path)
    return list(paths)


def get_output_paths_for_stages(stages: list[str]) -> set[str]:
    """Get output paths for specific stages only."""
    result: set[str] = set()
    for name in stages:
        try:
            info = registry.REGISTRY.get(name)
        except KeyError:
            logger.warning(f"Stage '{name}' not found in registry, skipping")
            continue
        for out_path in info["outs_paths"]:
            result.add(str(out_path))
    return result


def create_watch_filter(
    stages_to_run: list[str],
    watch_globs: list[str] | None = None,
) -> "Callable[[Change, str], bool]":
    """Create filter excluding outputs from stages being run (prevents infinite loops)."""
    # Collect resolved output paths, skipping any that can't be resolved
    outputs_to_filter: set[pathlib.Path] = set()
    for p in get_output_paths_for_stages(stages_to_run):
        resolved = project.try_resolve_path(p)
        if resolved is not None:
            outputs_to_filter.add(resolved)

    def watch_filter(change: "Change", path: str) -> bool:
        _ = change

        # Always filter Python bytecode
        if path.endswith((".pyc", ".pyo")) or "__pycache__" in path:
            return False

        # Resolve incoming path for consistent comparison
        resolved_path = project.try_resolve_path(path)
        if resolved_path is None:
            return True  # Can't resolve, don't filter

        # Check if path is an output of a stage being run, or inside such an output directory
        for out in outputs_to_filter:
            if resolved_path == out or out in resolved_path.parents:
                return False

        # Apply glob filters if specified
        if watch_globs:
            filename = resolved_path.name
            rel_path = path
            return any(
                fnmatch.fnmatch(filename, glob) or fnmatch.fnmatch(rel_path, glob)
                for glob in watch_globs
            )

        return True

    return watch_filter
