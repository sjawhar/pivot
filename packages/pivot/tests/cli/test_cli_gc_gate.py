"""PR535Gate: one red test per finding on sjawhar/pivot#535 at head fa7bd0c1.

Drop in as ``packages/pivot/tests/cli/test_cli_gc_gate.py``. Every test here fails at
fa7bd0c1 and must pass once the finding it names is fixed. The fixtures and helpers repeat
the ones in ``test_cli_gc.py`` so the module stands alone.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, TypedDict

import pytest

from pivot import cli, project, run_history, types
from pivot.storage import cache, gc, lock, state

if TYPE_CHECKING:
    import pathlib

    import click.testing

_STAGE = "demo/train"


class Workspace(TypedDict):
    root: pathlib.Path
    cache_dir: pathlib.Path
    object_root: pathlib.Path
    state_dir: pathlib.Path
    stages_dir: pathlib.Path
    scratch: pathlib.Path


def _helper_workspace(root: pathlib.Path, cache_dir: pathlib.Path) -> Workspace:
    state_dir = root / ".pivot"
    stages_dir = state_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    object_root = cache_dir / gc.OBJECTS_DIR
    object_root.mkdir(parents=True, exist_ok=True)
    scratch = root.parent / f"scratch-{root.name}"
    scratch.mkdir(exist_ok=True)
    return Workspace(
        root=root,
        cache_dir=cache_dir,
        object_root=object_root,
        state_dir=state_dir,
        stages_dir=stages_dir,
        scratch=scratch,
    )


@pytest.fixture
def workspace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A project whose cache lives inside it, so no register is needed."""
    root = tmp_path / "project"
    root.mkdir()
    space = _helper_workspace(root, root / ".pivot" / "cache")
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return space


@pytest.fixture
def shared(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A project whose cache lives outside it, so the register decides."""
    root = tmp_path / "first"
    root.mkdir()
    space = _helper_workspace(root, tmp_path / "shared-cache")
    monkeypatch.setenv("PIVOT_CACHE_DIR", str(space["cache_dir"]))
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", None)
    return space


def _helper_publish(space: Workspace, content: str) -> str:
    source = space["scratch"] / f"source-{content}"
    source.write_text(content)
    key, _ = cache.hash_file(source)
    cache.copy_to_cache(source, cache.get_cache_path(space["object_root"], key))
    source.unlink()
    return key


def _helper_write_lock(
    space: Workspace,
    stage_name: str = _STAGE,
    stages_dir: pathlib.Path | None = None,
    outs: dict[str | None, types.HashInfo] | None = None,
) -> None:
    lock.StageLock(stage_name, stages_dir or space["stages_dir"]).write(
        types.LockData(
            code_manifest={"self:test": "abc123"},
            params={},
            dep_hashes={},
            output_hashes={
                types.ArtifactIdentity(stage_name, key): hash_info
                for key, hash_info in (outs or {}).items()
            },
        )
    )


def _helper_run(runner: click.testing.CliRunner, *args: str) -> click.testing.Result:
    return runner.invoke(cli.cli, ["gc", *args])


def _helper_json(runner: click.testing.CliRunner, *args: str) -> gc.GcResult:
    result = _helper_run(runner, "--json", *args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{") :])


def _helper_keys(space: Workspace) -> set[str]:
    keys = set[str]()
    for prefix in space["object_root"].iterdir():
        if len(prefix.name) != 2:
            continue
        keys.update(
            prefix.name + entry.name
            for entry in prefix.iterdir()
            if cache.is_valid_hash(prefix.name + entry.name)
        )
    return keys


# =============================================================================
# Finding B1: an object a concurrent run republishes and names is removed
# =============================================================================


# Finding B1 (an object a concurrent run re-publishes and names is removed) is tested in
# test_cli_gc_protocol.py, through real pivot processes: a stage commit and a run-cache skip each
# wait while a collection holds the naming lock, and a collection proceeds while a stage runs.
# The gate's original B1 test monkeypatched a lock write inside gc's own process while gc held
# the exclusive lock, an interleaving the protocol rules out, so it was replaced, not kept.


# =============================================================================
# Finding B3: --dry-run under-reports when run history is beyond the retention
# =============================================================================


def _helper_two_runs_one_row(space: Workspace, hash_info: types.HashInfo) -> None:
    """Two runs, the older of which a run-cache row names, so a prune drops that row."""
    with state.StateDB(space["state_dir"]) as db:
        for index in range(2):
            db.write_run(
                run_history.RunManifest(
                    run_id=f"20260101T00000{index}-{index}",
                    started_at="2026-01-01T00:00:00Z",
                    ended_at="2026-01-01T00:00:01Z",
                    targeted_stages=[],
                    execution_order=[],
                    stages={},
                )
            )
        db.apply_deferred_writes(
            _STAGE,
            [],
            types.DeferredWrites(
                run_cache_input_hash="0123456789abcdef",
                run_cache_entry=run_history.RunCacheEntry(
                    run_id="20260101T000000-0",
                    output_hashes=[
                        run_history.output_hash_to_entry("data/demo/train.txt", hash_info)
                    ],
                ),
            ),
        )


def test_gc_dry_run_counts_what_a_real_run_removes_past_the_retention(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """`gc` prunes run history before it reads the records, so a real run removes the
    objects whose only record was a row the prune dropped, while `--dry-run` — which prunes
    nothing — keeps them and reports fewer removals than the run it is a plan for.
    """
    (workspace["state_dir"] / "config.yaml").write_text("core:\n  run_history_retention: 1\n")
    row_only = _helper_publish(workspace, "named only by an old row")
    _helper_two_runs_one_row(workspace, types.FileHash(hash=row_only))

    planned = _helper_json(runner, "--dry-run", "--grace-hours", "0")
    done = _helper_json(runner, "--yes", "--grace-hours", "0")

    assert planned["removed"] == done["removed"], (planned["removed"], done["removed"])


# =============================================================================
# Finding B4: a run that removes nothing has already pruned the run history
# =============================================================================


def test_gc_prunes_no_run_history_when_it_removes_nothing(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """The prune is irreversible and runs before the confirmation, so a refused run — no
    terminal and no --yes, or a user who answers no — still deletes run history and the
    run-cache rows that keep earlier outputs restorable.
    """
    (workspace["state_dir"] / "config.yaml").write_text("core:\n  run_history_retention: 1\n")
    dead = _helper_publish(workspace, "dead")
    _helper_two_runs_one_row(workspace, types.FileHash(hash=dead))

    result = _helper_run(runner, "--grace-hours", "0")

    assert result.exit_code != 0, result.output
    assert "Nothing was removed" in result.output, result.output
    with state.StateDB(workspace["state_dir"], readonly=True) as db:
        assert len(db.list_runs()) == 2, "a refused collection pruned the run history"
        assert len(db.list_run_cache_rows()) == 1, "a refused collection dropped a row"


# =============================================================================
# Finding B2: --projects accepts a path that is not a pivot project
# =============================================================================


def test_gc_refuses_a_projects_path_that_is_not_a_pivot_project(
    shared: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`--projects` only has to name a directory, so a typo — the cache itself, a worktree
    that is gone, a parent directory — satisfies the register rule and the collection
    proceeds on this project's records alone, removing another project's live objects. A
    path with no state directory names no records and must be refused.
    """
    second = tmp_path / "second"
    second_stages = second / ".pivot" / "stages"
    second_stages.mkdir(parents=True)
    live = _helper_publish(shared, "named by the second project")
    _helper_write_lock(
        shared, "demo/other", stages_dir=second_stages, outs={None: types.FileHash(hash=live)}
    )
    not_a_project = tmp_path / "not-a-project"
    not_a_project.mkdir()

    result = _helper_run(runner, "--yes", "--grace-hours", "0", "--projects", str(not_a_project))

    assert result.exit_code != 0, result.output
    assert live in _helper_keys(shared), "a live object of a project gc never read was removed"


# =============================================================================
# Finding N2, as ruled in round 4 (R2): an absent root is never taken as deleted
# =============================================================================


def test_gc_refuses_and_names_the_register_entry_whose_root_is_absent(
    shared: Workspace, runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """At fa7bd0c1 the note said the entry was dropped while it stayed. Round 4's ruling (R2)
    goes further: an absent root may be an unmounted disk or another mount namespace, so the
    collection never drops it on its own. It refuses, names the entry for the user to delete
    if the project is gone for good, and keeps it; a second run says the same."""
    gone = tmp_path / "a-worktree-that-was-deleted"
    gc.register_project(shared["cache_dir"], shared["root"], shared["state_dir"])
    (gone / ".pivot").mkdir(parents=True)
    gc.register_project(shared["cache_dir"], gone, gone / ".pivot")
    entry = gc.project_register_path(shared["cache_dir"], gone, gone / ".pivot")
    assert entry is not None
    shutil.rmtree(gone)

    first = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert first.exit_code != 0, first.output
    assert str(entry) in first.output, first.output
    assert entry.exists(), "a refused collection dropped the entry"

    second = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert second.exit_code != 0, second.output
    assert str(entry) in second.output, second.output


# =============================================================================
# Finding N3: the summary misstates the grace it applied
# =============================================================================


def test_gc_summary_states_the_grace_it_applied(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """A grace under 0.05 h prints as "the last 0.0h", which reads as no grace at all."""
    _helper_publish(workspace, "young")

    result = _helper_run(runner, "--dry-run", "--grace-hours", "0.01")

    assert "published in the last 0.01h" in result.output, result.output


# =============================================================================
# Finding N1: a lock whose YAML does not parse crashes the collection
# =============================================================================


def test_gc_reports_a_lock_it_cannot_parse_as_a_pivot_error(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """Lock files are committed, so a merge leaves conflict markers in one, and a disk error
    truncates one. `yaml.load` then raised out of `_lock_keys` and the user was shown the
    repr of a yaml exception. A record pivot cannot read refuses the run, names the lock and
    says why, and removes nothing.
    """
    dead = _helper_publish(workspace, "dead")
    (workspace["stages_dir"] / "demo").mkdir(parents=True, exist_ok=True)
    conflicted = workspace["stages_dir"] / "demo" / "conflicted.lock"
    conflicted.write_text(
        "<<<<<<< HEAD\nouts:\n- hash: 0123456789abcdef\n=======\n"
        + "outs:\n- hash: fedcba9876543210\n>>>>>>> other\n"
    )

    result = _helper_run(runner, "--yes", "--grace-hours", "0")

    assert "Traceback" not in result.output and "ScannerError" not in result.output, result.output
    assert result.exit_code != 0, result.output
    assert str(conflicted) in result.output and "does not parse" in result.output, result.output
    assert dead in _helper_keys(workspace), result.output


# =============================================================================
# Finding N4: a project that has never cached anything cannot be collected
# =============================================================================


def test_gc_collects_nothing_in_a_project_that_has_never_cached_anything(
    workspace: Workspace, runner: click.testing.CliRunner
) -> None:
    """`pivot init` creates no cache directory, and `gc` opens `<cache>/gc.lock` without
    creating its parent, so the first `pivot gc` of a fresh checkout exits 1 with "cannot
    open …/gc.lock (No such file or directory). Collecting a cache needs write access to
    it" — which blames permissions for an empty cache. A cache with no objects is a
    collection of nothing, not a failure.
    """
    shutil.rmtree(workspace["cache_dir"])

    result = _helper_run(runner, "--dry-run")

    assert result.exit_code == 0, result.output
