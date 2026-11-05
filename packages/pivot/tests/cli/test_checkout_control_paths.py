from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import isolated_pivot_dir
from pivot import cli, config, project, types
from pivot.config import io as config_io
from pivot.remote import storage as remote_storage
from pivot.storage import cache, ownership, track

if TYPE_CHECKING:
    import pathlib

    import click.testing
    from pytest_mock import MockerFixture

    from conftest import GitRepo


def _helper_sidecar(root: pathlib.Path, target: str) -> None:
    source = root / "payload.txt"
    source.write_text("cached replacement\n")
    record = cache.save_to_cache(
        source, config.get_cache_dir() / "files", checkout_mode=cache.CheckoutMode.COPY
    )
    source.unlink()
    path = root / target
    path.parent.mkdir(parents=True, exist_ok=True)
    track.write_pvt_file(
        track.get_pvt_path(path),
        track.PvtData(path=path.name, hash=record["hash"], size=19),
    )


@pytest.mark.parametrize("control", [".git", ".pivot"])
@pytest.mark.parametrize("target_form", ["all", "path", "sidecar"])
def test_forced_checkout_refuses_control_directory_before_restoring_any_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str, target_form: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        marker = root / control / "keep.txt"
        marker.write_text("metadata must survive\n")
        _helper_sidecar(root, "a.txt")
        _helper_sidecar(root, control)
        targets = {"all": [], "path": [control], "sidecar": [f"{control}.pvt"]}

        result = runner.invoke(cli.cli, ["checkout", "--force", *targets[target_form]])

        assert marker.is_file(), result.output
        assert marker.read_text() == "metadata must survive\n", result.output
        assert result.exit_code == 1, result.output
        assert control in result.output and "Refusing" in result.output, result.output
        assert "Restored" not in result.output, result.output
        assert not (root / "a.txt").exists(), "A refused preflight must restore nothing"


@pytest.mark.parametrize("options", [[], ["--force"], ["--only-missing"]])
@pytest.mark.parametrize("target", [".git/HEAD", ".pivot/config.yaml"])
def test_checkout_refuses_files_inside_control_directories(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    options: list[str],
    target: str,
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", *options, target])

        assert result.exit_code == 1, result.output
        assert target in result.output and "Refusing" in result.output, result.output
        assert not (root / target).exists(), result.output


@pytest.mark.parametrize(
    ("configuration", "target", "control"),
    [
        ("core:\n  state_dir: state\n", "state", "state"),
        ("core:\n  state_dir: data/state\n", "data", "data/state"),
        ("cache:\n  dir: objects\n", "objects", "objects"),
    ],
)
def test_forced_checkout_preserves_configured_control_directories(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    configuration: str,
    target: str,
    control: str,
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text(configuration)
        marker = root / control / "keep.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("metadata must survive\n")
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert marker.is_file(), result.output
        assert marker.read_text() == "metadata must survive\n", result.output
        assert result.exit_code == 1, result.output
        assert target in result.output and "Refusing" in result.output, result.output


def test_checkout_refuses_writes_through_an_alias_into_git_metadata(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "alias").symlink_to(root / ".git", target_is_directory=True)
        _helper_sidecar(root, "HEAD")
        sidecar = root / "HEAD.pvt"
        record = track.read_pvt_file(sidecar)
        assert record is not None
        record["path"] = "alias/HEAD"
        track.write_pvt_file(sidecar, record)

        result = runner.invoke(cli.cli, ["checkout", "--force"])

        assert result.exit_code == 1, result.output
        assert "alias/HEAD" in result.output and ".git" in result.output, result.output
        assert not (root / ".git" / "HEAD").exists(), result.output


def test_checkout_still_restores_an_ordinary_dot_directory(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        hidden = root / ".hidden"
        hidden.mkdir()
        (hidden / "file.txt").write_text("cached data\n")
        result = runner.invoke(cli.cli, ["track", ".hidden"])
        assert result.exit_code == 0, result.output
        cache.remove_output(hidden)

        result = runner.invoke(cli.cli, ["checkout", "--force", ".hidden"])

        assert result.exit_code == 0, result.output
        assert (hidden / "file.txt").read_text() == "cached data\n", result.output


def test_checkout_keeps_data_beside_cache_owned_directories(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text("cache:\n  dir: shared\n")
        _helper_sidecar(root, "shared/data/file.txt")

        result = runner.invoke(cli.cli, ["checkout", "--force", "shared/data/file.txt"])

        assert result.exit_code == 0, result.output
        assert (root / "shared/data/file.txt").read_text() == "cached replacement\n"


@pytest.mark.parametrize("state_dir", [".", "state"])
def test_checkout_keeps_data_beside_state_owned_entries(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, state_dir: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text(f"core:\n  state_dir: {state_dir}\n")
        target = "data/file.txt" if state_dir == "." else "state/data/file.txt"
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert result.exit_code == 0, result.output
        assert (root / target).read_text() == "cached replacement\n"


@pytest.mark.parametrize("control", [".git", ".pivot"])
@pytest.mark.parametrize("target_form", ["all", "path", "sidecar"])
def test_checkout_preserves_nested_control_directories(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str, target_form: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        target = f"vendor/lib/{control}"
        marker = root / target / "keep.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("nested metadata\n")
        _helper_sidecar(root, target)
        targets = {"all": [], "path": [target], "sidecar": [f"{target}.pvt"]}

        result = runner.invoke(cli.cli, ["checkout", "--force", *targets[target_form]])

        assert marker.is_file(), result.output
        assert marker.read_text() == "nested metadata\n", result.output
        assert result.exit_code == 1 and "Refusing" in result.output, result.output


_PLOT_PIPELINE = """\
from typing import Annotated
import pandas as pd
from pivot import Pipeline, loaders, plot, stage

@stage
def chart() -> Annotated[pd.DataFrame, plot, loaders.CSV()]:
    return pd.DataFrame({"x": [1]})

with Pipeline("demo") as pipeline:
    chart()
"""


@pytest.mark.parametrize(
    "command", [["checkout", "--force"], ["checkout", "--only-missing"], ["pull"]]
)
def test_checkout_and_pull_restore_plot_outputs_with_state_at_project_root(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
    command: list[str],
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: .\n")
        (root / "pipeline.py").write_text(_PLOT_PIPELINE)
        built = runner.invoke(cli.cli, ["repro"])
        assert built.exit_code == 0, built.output
        output = root / "plots" / "demo" / "chart.csv"
        assert output.read_text() == "x\n1\n"
        output.unlink()
        mocker.patch(
            "pivot.remote.sync.create_remote_from_name",
            autospec=True,
            return_value=(mocker.Mock(spec=remote_storage.S3Remote), "origin"),
        )
        mocker.patch(
            "pivot.remote.sync.pull",
            autospec=True,
            return_value=types.TransferSummary(transferred=0, skipped=0, failed=0, errors=[]),
        )

        result = runner.invoke(cli.cli, [*command, "demo/chart"])

        assert result.exit_code == 0, result.output
        assert output.read_text() == "x\n1\n", result.output


def test_get_refuses_committed_git_sidecar(
    runner: click.testing.CliRunner, git_repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, commit = git_repo
    (root / ".pivot").mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(project, "_project_root_cache", root)
    _helper_sidecar(root, ".git")
    commit("tracked metadata sidecar")
    before = (root / ".git" / "HEAD").read_bytes()

    result = runner.invoke(cli.cli, ["get", ".git", "--force", "--rev", "HEAD"])

    assert (root / ".git").is_dir(), result.output
    assert (root / ".git" / "HEAD").read_bytes() == before, result.output
    assert result.exit_code == 1 and "Refusing" in result.output, result.output


def test_repro_checkout_missing_refuses_creation_inside_control_paths(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / "pipeline.py").write_text(_PLOT_PIPELINE)
        _helper_sidecar(root, "HEAD")
        sidecar = root / "HEAD.pvt"
        record = track.read_pvt_file(sidecar)
        assert record is not None
        record["path"] = ".git/created-metadata"
        track.write_pvt_file(sidecar, record)

        result = runner.invoke(cli.cli, ["repro", "--checkout-missing"])

        assert result.exit_code == 1 and "Refusing" in result.output, result.output
        assert not (root / ".git" / "created-metadata").exists(), result.output


@pytest.mark.parametrize("control", [".git", ".pivot"])
def test_checkout_preserves_a_directory_containing_nested_control_data(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        marker = root / "vendor" / "lib" / control / "keep.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("nested metadata\n")
        _helper_sidecar(root, "vendor")

        result = runner.invoke(cli.cli, ["checkout", "--force", "vendor"])

        assert marker.is_file(), result.output
        assert marker.read_text() == "nested metadata\n", result.output
        assert result.exit_code == 1 and "Refusing" in result.output, result.output


@pytest.mark.parametrize("control", ["projects", "mounts"])
def test_checkout_preserves_linked_cache_physical_register(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        physical = root / "physical"
        logical = root / "logical"
        (physical / "files").mkdir(parents=True)
        logical.mkdir()
        (logical / "files").symlink_to(physical / "files", target_is_directory=True)
        (root / ".pivot" / "config.yaml").write_text("cache:\n  dir: logical\n")
        marker = physical / control / "keep.txt"
        marker.parent.mkdir()
        marker.write_text("shared cache metadata\n")
        target = f"physical/{control}"
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert marker.is_file(), result.output
        assert marker.read_text() == "shared cache metadata\n", result.output
        assert result.exit_code == 1 and "Refusing" in result.output, result.output


_VENDOR_PIPELINE = """\
from typing import Annotated
from pivot import Pipeline, directory, loaders, stage

@stage
def vendor(source: str) -> Annotated[dict[str, str], directory, loaders.Text()]:
    return {"README.txt": source, "__CONTROL__/head.txt": source}

with Pipeline("demo") as pipeline:
    vendor(pipeline.input("input.txt"))
"""


@pytest.mark.parametrize("control", [".git", ".pivot"])
@pytest.mark.parametrize("mode", ["--force", "--only-missing"])
def test_stage_replaces_and_restores_its_own_control_named_members(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str, mode: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        (root / ".pivot" / "config.yaml").write_text("cache:\n  checkout_mode: [copy]\n")
        (root / "pipeline.py").write_text(_VENDOR_PIPELINE.replace("__CONTROL__", control))
        source = root / "data" / "raw" / "input.txt"
        source.parent.mkdir(parents=True)
        source.write_text("first output\n")
        built = runner.invoke(cli.cli, ["repro"])
        assert built.exit_code == 0, built.output
        output = root / "data" / "demo" / "vendor"
        assert (output / control / "head.txt").read_text() == "first output\n"

        source.write_text("changed output bytes\n")
        rebuilt = runner.invoke(cli.cli, ["repro"])

        assert rebuilt.exit_code == 0, rebuilt.output
        assert (output / control / "head.txt").read_text() == "changed output bytes\n"
        (output / "README.txt").unlink()

        restored = runner.invoke(cli.cli, ["checkout", mode, "demo/vendor"])

        assert restored.exit_code == 0, restored.output
        assert (output / "README.txt").read_text() == "changed output bytes\n"
        assert (output / control / "head.txt").read_text() == "changed output bytes\n"


@pytest.mark.parametrize("ancestor", [".git", ".pivot"])
def test_checkout_keeps_data_when_a_control_named_directory_is_above_the_project(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    ancestor: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / ancestor
    parent.mkdir()
    with isolated_pivot_dir(runner, parent) as root:
        monkeypatch.setattr(project, "_project_root_cache", root)
        _helper_sidecar(root, "ordinary.txt")

        result = runner.invoke(cli.cli, ["checkout", "--force", "ordinary.txt"])

        assert result.exit_code == 0, result.output
        assert (root / "ordinary.txt").read_text() == "cached replacement\n"


def test_checkout_refuses_an_unreadable_tracked_tree_before_restoring_a_sibling(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        _helper_sidecar(root, "tree")
        _helper_sidecar(root, "a.txt")
        hidden = root / "tree" / "locked"
        marker = hidden / ".git" / "HEAD"
        marker.parent.mkdir(parents=True)
        marker.write_text("user metadata\n")
        original_mode = hidden.stat().st_mode
        hidden.chmod(0)
        try:
            result = runner.invoke(cli.cli, ["checkout", "--force"])
        finally:
            hidden.chmod(original_mode)

        assert result.exit_code == 1, result.output
        assert "cannot inspect" in result.output, result.output
        assert "tree" in result.output, result.output
        assert marker.read_text() == "user metadata\n"
        assert not (root / "a.txt").exists()


@pytest.mark.parametrize("control", [".git", ".pivot"])
def test_a_stage_still_cannot_replace_actual_project_control_paths(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, control: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        target = ownership.ForcedRestore(
            replacement=ownership.stage_output(root / control, "demo/unsafe"),
            restored=None,
            claims=None,
        )

        refusal = ownership.swap_refusal(root, target, root / control)

        assert refusal is not None
        assert f"overlaps control path '{control}'" in str(refusal)


def test_a_project_root_alias_still_protects_its_nested_repository(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        nested_git = root / "vendor" / "lib" / ".git"
        nested_git.mkdir(parents=True)
        (nested_git / "HEAD").write_text("nested metadata\n")
        (root / "hop").symlink_to(nested_git.parent, target_is_directory=True)
        alias = tmp_path / "project-alias"
        alias.symlink_to(root, target_is_directory=True)

        refusal = ownership.control_path_refusal(
            alias, [ownership.tracked_restore(alias / "hop" / ".git")]
        )

        assert refusal is not None
        assert "overlaps control path" in str(refusal)
        assert (nested_git / "HEAD").read_text() == "nested metadata\n"


def test_metadata_controls_follow_a_changed_cache_configuration(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        configuration = root / ".pivot" / "config.yaml"
        configuration.write_text("cache:\n  dir: first\n")
        replacement = ownership.stage_output(root / "second" / "projects", "p/out")
        assert ownership.control_path_refusal(root, [replacement]) is None

        configuration.write_text("cache:\n  dir: second\n")
        config_io.clear_config_cache()

        refusal = ownership.control_path_refusal(root, [replacement])
        assert refusal is not None
        assert "second/projects" in str(refusal)


def test_metadata_controls_follow_a_repointed_object_store(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        first = root / "first" / "files"
        second = root / "second" / "files"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        logical = root / "logical"
        logical.mkdir()
        files = logical / "files"
        files.symlink_to(first, target_is_directory=True)
        (root / ".pivot" / "config.yaml").write_text("cache:\n  dir: logical\n")
        replacement = ownership.stage_output(second.parent / "projects", "p/out")
        assert ownership.control_path_refusal(root, [replacement]) is None

        files.unlink()
        files.symlink_to(second, target_is_directory=True)

        refusal = ownership.control_path_refusal(root, [replacement])
        assert refusal is not None
        assert "second/projects" in str(refusal)


def test_metadata_preflight_does_not_apply_after_its_target_path_changes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        target = ownership.ForcedRestore(
            replacement=ownership.stage_output(root / "ordinary", "p/out"),
            restored=None,
            claims=None,
        )
        assert ownership.forced_restore_refusal(root, [target]) is None
        target["replacement"] = ownership.stage_output(root / ".git", "p/out")

        refusal = ownership.swap_refusal(root, target, root / ".git")

        assert refusal is not None
        assert "overlaps control path '.git'" in str(refusal)


@pytest.mark.parametrize("target", [".pivot/config.yaml", ".pivot/config.yaml.lock"])
def test_moving_state_keeps_the_marker_configuration_and_lock_protected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, target: str
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        configuration = root / ".pivot" / "config.yaml"
        configuration.write_text("core:\n  state_dir: .state\n")
        before = configuration.read_bytes()
        _helper_sidecar(root, target)

        result = runner.invoke(cli.cli, ["checkout", "--force", target])

        assert result.exit_code == 1 and "Refusing" in result.output, result.output
        assert target in result.output, result.output
        assert configuration.read_bytes() == before


def test_directory_name_refusal_does_not_extend_outside_the_project(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as root:
        outside = tmp_path / "outside" / ".git"
        outside.mkdir(parents=True)

        refusal = ownership.control_path_refusal(root, [ownership.destination(outside)])

        assert refusal is None
