from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from conftest import isolated_pivot_dir
from pivot import cli, compose, discovery, git, project, stage_def
from tests import helpers

if TYPE_CHECKING:
    import pathlib

    import click.testing
    from pytest_mock import MockerFixture


# =============================================================================
# Module-level StageParams classes for annotation-based registration
# =============================================================================


class _TrainParams(stage_def.StageParams):
    lr: float = 0.01
    epochs: int = 10


class _SimpleParams(stage_def.StageParams):
    x: int = 1


class _PrecisionParams(stage_def.StageParams):
    lr: float = 0.123456789


class _ChangedParams(stage_def.StageParams):
    x: int = 2


# =============================================================================
# Module-level stage functions
# =============================================================================


@compose.stage
def train(params: _TrainParams) -> None:
    pass


@compose.stage
def stage_simple(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_a(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_b(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_c(params: _SimpleParams) -> None:
    pass


@compose.stage
def stage_precision(params: _PrecisionParams) -> None:
    pass


@compose.stage
def stage_changed(params: _ChangedParams) -> None:
    pass


def _head_lock(stage_name: str, params: dict[str, object]) -> dict[str, bytes]:
    """HEAD lock-file content for ``git.read_files_from_head`` (path keyed by qualified name)."""
    content = yaml.dump(
        {"schema_version": 3, "code_manifest": {}, "params": params, "deps": [], "outs": []}
    )
    return {f".pivot/stages/{stage_name}.lock": content.encode()}


# =============================================================================
# Params Show Tests
# =============================================================================


def test_params_show_help(runner: click.testing.CliRunner) -> None:
    """params show command shows help."""
    result = runner.invoke(cli.cli, ["params", "show", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--precision" in result.output


def test_params_show_no_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows no params message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_show_with_params(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows params from registered stage."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        train(params=_TrainParams())

    result = runner.invoke(cli.cli, ["params", "show"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "lr" in result.output
    assert "0.01" in result.output


def test_params_show_json_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["test/stage_simple"]["x"] == 1


def test_params_show_md_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "--md"])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


def test_params_show_specific_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show filters to specific stages."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_a(params=_SimpleParams())
        stage_b(params=_SimpleParams())
        stage_c(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "test/stage_a", "test/stage_c"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_c" in result.output
    assert "test/stage_b" not in result.output


def test_params_show_bare_stage_name_resolves(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
) -> None:
    """params show accepts a bare stage name in a single-pipeline project (names.py design)."""
    with mock_discovery:
        stage_a(params=_SimpleParams())
        stage_b(params=_SimpleParams())

    result = runner.invoke(cli.cli, ["params", "show", "stage_a"])

    assert result.exit_code == 0, result.output
    assert "test/stage_a" in result.output
    assert "test/stage_b" not in result.output


def test_params_show_precision(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show respects --precision flag."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_precision(params=_PrecisionParams())

    result = runner.invoke(cli.cli, ["params", "show", "--precision", "2"])

    assert result.exit_code == 0
    assert "0.12" in result.output
    assert "0.123456789" not in result.output


# =============================================================================
# Params Diff Tests
# =============================================================================


def test_params_diff_help(runner: click.testing.CliRunner) -> None:
    """params diff command shows help."""
    result = runner.invoke(cli.cli, ["params", "diff", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output
    assert "--md" in result.output
    assert "--precision" in result.output


def test_params_diff_no_stages(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shows message when no stages registered."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameters found" in result.output


def test_params_diff_no_changes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows no changes when params match HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_simple", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "No parameter changes" in result.output


def test_params_diff_with_changes(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Shows diff when params changed from HEAD."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "modified" in result.output
    assert "stage" in result.output


def test_params_diff_json_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --json outputs valid JSON."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff", "--json"])

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert len(parsed) == 1
    assert parsed[0]["change_type"] == "modified"


def test_params_diff_md_format(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff --md outputs markdown table."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_changed(params=_ChangedParams())

    mocker.patch.object(
        git, "read_files_from_head", return_value=_head_lock("test/stage_changed", {"x": 1})
    )

    result = runner.invoke(cli.cli, ["params", "diff", "--md"])

    assert result.exit_code == 0
    assert "|" in result.output
    assert "---" in result.output


# =============================================================================
# Command Group Tests
# =============================================================================


def test_params_group_help(runner: click.testing.CliRunner) -> None:
    """Params group shows subcommands."""
    result = runner.invoke(cli.cli, ["params", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "diff" in result.output


def test_params_in_main_help(runner: click.testing.CliRunner) -> None:
    """Params command appears in main help."""
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "params" in result.output


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_params_show_unknown_stage_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params show errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "show", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_unknown_stage_error(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """params diff errors on unknown stage names."""
    monkeypatch.chdir(mock_discovery.root)

    result = runner.invoke(cli.cli, ["params", "diff", "nonexistent_stage"])

    assert result.exit_code != 0
    assert "Unknown stage(s): nonexistent_stage" in result.output


def test_params_diff_no_git_warning(
    mock_discovery: compose.Pipeline,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """params diff warns when not in git repo."""
    monkeypatch.chdir(mock_discovery.root)

    with mock_discovery:
        stage_simple(params=_SimpleParams())

    mocker.patch.object(git, "read_files_from_head", return_value={})
    mocker.patch.object(git, "is_git_repo_with_head", return_value=False)

    result = runner.invoke(cli.cli, ["params", "diff"])

    assert result.exit_code == 0
    assert "Warning: Not in a git repository" in result.output


# =============================================================================
# Pipeline Discovery Tests
# =============================================================================


def test_params_show_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    test_pipeline: compose.Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params show discovers pipeline without needing prior command."""
    monkeypatch.chdir(test_pipeline.root)
    (test_pipeline.root / ".pivot").mkdir(exist_ok=True)

    # Mock discover_pipeline to return the test pipeline
    mocker.patch.object(discovery, "discover_pipeline", return_value=test_pipeline)
    mocker.patch.object(project, "_project_root_cache", test_pipeline.root)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    result = runner.invoke(cli.cli, ["params", "show"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output, f"Output: {result.output}"
    assert result.exit_code == 0, f"Output: {result.output}"


def test_params_diff_without_prior_run_discovers_pipeline(
    runner: click.testing.CliRunner,
    test_pipeline: compose.Pipeline,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pivot params diff discovers pipeline without needing prior command."""
    monkeypatch.chdir(test_pipeline.root)
    (test_pipeline.root / ".pivot").mkdir(exist_ok=True)

    # Mock discover_pipeline to return the test pipeline
    mocker.patch.object(discovery, "discover_pipeline", return_value=test_pipeline)
    mocker.patch.object(project, "_project_root_cache", test_pipeline.root)
    # IMPORTANT: Do NOT mock get_pipeline_from_context - we want to test that
    # ensure_stages_registered() is called and populates the context

    result = runner.invoke(cli.cli, ["params", "diff"])

    # Should not raise NoPipelineError - discovery should have been called
    assert "No pipeline found" not in result.output, f"Output: {result.output}"
    assert result.exit_code == 0, f"Output: {result.output}"


# =============================================================================
# params.yaml keys are checked against the loaded stages
# =============================================================================


def _scale_pipeline(name: str) -> str:
    return (
        "import pandas as pd\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
        + "class ScaleParams(StageParams):\n    factor: int = 10\n\n\n"
        + "@stage\ndef scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:\n"
        + "    return numbers * params.factor\n\n\n"
        + f'with Pipeline("{name}") as pipeline:\n'
        + '    scale(pipeline.input("numbers.csv"), params=ScaleParams())\n'
        + '    with pipeline.variant("big"):\n'
        + '        scale(pipeline.input("numbers.csv"), params=ScaleParams(factor=100))\n'
    )


def _write_scale_project(cwd: pathlib.Path, params_yaml: str, *names: str) -> None:
    """One scale pipeline per name: at the root for one name, in <name>/ for several."""
    (cwd / "data" / "raw").mkdir(parents=True)
    (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
    (cwd / "params.yaml").write_text(params_yaml)
    if len(names) == 1:
        (cwd / "pipeline.py").write_text(_scale_pipeline(names[0]))
        return
    for name in names:
        (cwd / name).mkdir()
        (cwd / name / "pipeline.py").write_text(_scale_pipeline(name))


@pytest.mark.parametrize(
    "args",
    [["repro"], ["status"], ["status", "--explain"], ["params", "show"], ["commit"]],
    ids=["repro", "status", "status-explain", "params-show", "commit"],
)
def test_bare_params_key_is_refused_naming_the_qualified_stage(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, args: list[str]
) -> None:
    """Keys used to be bare stage names; a bare 'scale:' must not be dropped silently."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(cwd, "scale:\n  factor: 3\n", "smoke")

        result = runner.invoke(cli.cli, args)

    assert result.exit_code != 0, result.output
    assert (
        "params.yaml key 'scale' names no stage; keys are '<pipeline>/<stage>', "
        + "e.g. 'smoke/scale'"
    ) in result.output, result.output


def test_misspelled_params_key_is_refused_with_close_matches(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A qualified key for a loaded pipeline that matches no stage or variant is a typo."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(
            cwd, "smoke/scael:\n  factor: 3\nsmoke/scale@bgi:\n  factor: 4\n", "smoke"
        )

        result = runner.invoke(cli.cli, ["repro"])

    assert result.exit_code != 0, result.output
    assert (
        "params.yaml key 'smoke/scael' names no stage in pipeline 'smoke'; "
        + "did you mean 'smoke/scale' or 'smoke/scale@big'?"
    ) in result.output, result.output
    assert (
        "params.yaml key 'smoke/scale@bgi' names no stage in pipeline 'smoke'; "
        + "did you mean 'smoke/scale@big'"
    ) in result.output, result.output
    assert not (cwd / "data" / "smoke").exists(), "No stage may run with a refused params.yaml"


@pytest.mark.parametrize(
    "args",
    [["repro"], ["status"], ["params", "show"], ["commit"]],
    ids=["repro", "status", "params-show", "commit"],
)
def test_bad_params_value_is_refused_before_any_stage_runs(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, args: list[str]
) -> None:
    """A value its StageParams field rejects is refused at load, not when the stage runs."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(cwd, "smoke/scale:\n  factor: not-an-int\n", "smoke")

        result = runner.invoke(cli.cli, args)

    assert result.exit_code != 0, result.output
    assert result.output.count("field 'factor' value 'not-an-int'") == 1, result.output
    assert (
        "params.yaml key 'smoke/scale': field 'factor' value 'not-an-int' is invalid for "
        + "ScaleParams: Input should be a valid integer, unable to parse string as an integer"
    ) in result.output, result.output
    assert not (cwd / "data" / "smoke").exists(), "No stage may run with a refused params.yaml"


@pytest.mark.parametrize(
    "args",
    [["repro"], ["status"], ["params", "show"], ["commit"]],
    ids=["repro", "status", "params-show", "commit"],
)
def test_misspelled_params_field_is_refused_with_close_matches(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, args: list[str]
) -> None:
    """A field typo used to be dropped silently, so the stage ran with its code default."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(cwd, "smoke/scale:\n  factr: 20\n", "smoke")

        result = runner.invoke(cli.cli, args)

    assert result.exit_code != 0, result.output
    assert result.output.count("field 'factr'") == 1, result.output
    assert (
        "params.yaml key 'smoke/scale': field 'factr' is not a field of ScaleParams; "
        + "did you mean 'factor'?"
    ) in result.output, result.output
    assert not (cwd / "data" / "smoke").exists(), "No stage may run with a refused params.yaml"


def _write_nested_project(cwd: pathlib.Path, params_yaml: str) -> None:
    """A stage whose params hold a plain BaseModel and one that allows extra fields."""
    (cwd / "data" / "raw").mkdir(parents=True)
    (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
    (cwd / "params.yaml").write_text(params_yaml)
    (cwd / "pipeline.py").write_text(
        "import pandas as pd\nimport pydantic\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
        + "class Opt(pydantic.BaseModel):\n    momentum: float = 0.9\n\n\n"
        + "class Loose(pydantic.BaseModel):\n"
        + '    model_config = pydantic.ConfigDict(extra="allow")\n'
        + "    momentum: float = 0.9\n\n\n"
        + "class TrainParams(StageParams):\n    optimizer: Opt = Opt()\n    loose: Loose = Loose()\n\n\n"
        + "@stage\ndef train(numbers: pd.DataFrame, params: TrainParams) -> dict:\n"
        + '    return {"momentum": params.optimizer.momentum, "loose": params.loose.model_dump()}\n\n\n'
        + 'with Pipeline("smoke") as pipeline:\n'
        + '    train(pipeline.input("numbers.csv"), params=TrainParams())\n'
    )


@pytest.mark.parametrize(
    "args",
    [["repro"], ["status"], ["params", "show"], ["commit"]],
    ids=["repro", "status", "params-show", "commit"],
)
def test_misspelled_field_of_a_plain_nested_model_is_refused(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, args: list[str]
) -> None:
    """Pydantic ignores an unknown key in a plain BaseModel, so the stage ran with the default."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_nested_project(cwd, "smoke/train:\n  optimizer:\n    momentm: 0.5\n")

        result = runner.invoke(cli.cli, args)

    assert result.exit_code != 0, result.output
    assert (
        "params.yaml key 'smoke/train': field 'optimizer.momentm' is not a field of Opt; "
        + "did you mean 'momentum'?"
    ) in result.output, result.output
    assert not (cwd / "data" / "smoke").exists(), "No stage may run with a refused params.yaml"


def test_nested_model_that_allows_extra_fields_takes_one_from_params_yaml(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_nested_project(cwd, "smoke/train:\n  loose:\n    extra: 1\n")

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 0, result.output
        assert yaml.safe_load((cwd / "data" / "smoke" / "train.yaml").read_text()) == {
            "momentum": 0.9,
            "loose": {"momentum": 0.9, "extra": 1},
        }


def test_changing_a_secret_param_reruns_the_stage_without_recording_it(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Effective params showed a secret as '**********', so a new secret never reran."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
        (cwd / "pipeline.py").write_text(
            "import pandas as pd\nimport pydantic\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
            + "class FetchParams(StageParams):\n"
            + '    token: pydantic.SecretStr = pydantic.SecretStr("code-token")\n\n\n'
            + "@stage\ndef fetch(numbers: pd.DataFrame, params: FetchParams) -> dict:\n"
            + '    return {"length": len(params.token.get_secret_value())}\n\n\n'
            + 'with Pipeline("smoke") as pipeline:\n'
            + '    fetch(pipeline.input("numbers.csv"), params=FetchParams())\n'
        )
        lock_file = cwd / ".pivot" / "stages" / "smoke" / "fetch.lock"

        runs = list[str]()
        locks = list[str]()
        for token in ("first-token", "first-token", "second-token-longer"):
            (cwd / "params.yaml").write_text(f"smoke/fetch:\n  token: {token}\n")
            result = runner.invoke(cli.cli, ["repro"])
            assert result.exit_code == 0, result.output
            runs.append(result.output)
            locks.append(lock_file.read_text())
        output = yaml.safe_load((cwd / "data" / "smoke" / "fetch.yaml").read_text())
        key_file = cwd / ".pivot" / "secret.key"
        key_mode = key_file.stat().st_mode & 0o777

    assert "1 ran" in runs[0], runs[0]
    assert "0 ran, 1 cached" in runs[1], runs[1]
    assert "1 ran" in runs[2], runs[2]
    assert output == {"length": len("second-token-longer")}
    assert key_mode == 0o600
    for lock_text, token in zip(
        locks, ("first-token", "first-token", "second-token-longer"), strict=True
    ):
        assert "token: hmac:" in lock_text, lock_text
        assert token not in lock_text
        assert hashlib.sha256(token.encode()).hexdigest()[:16] not in lock_text
    assert locks[0] != locks[2]


_SECRET_PIPELINE = (
    "import pandas as pd\nimport pydantic\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
    + "class FetchParams(StageParams):\n"
    + "    pin: pydantic.Secret[int] = pydantic.Secret[int](424242)\n\n\n"
    + "@stage\ndef fetch(numbers: pd.DataFrame, params: FetchParams) -> dict:\n"
    + '    return {"pin_digits": len(str(params.pin.get_secret_value()))}\n\n\n'
    + 'with Pipeline("smoke") as pipeline:\n'
    + '    fetch(pipeline.input("numbers.csv"), params=FetchParams())\n'
)


def _secret_clone(path: pathlib.Path, home: pathlib.Path) -> None:
    """A git repository with `pivot init`, the secret pipeline and its input."""
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    _pivot(path, home, "init")
    (path / "data" / "raw").mkdir(parents=True)
    (path / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
    (path / "pipeline.py").write_text(_SECRET_PIPELINE)


def _pivot(
    cwd: pathlib.Path, home: pathlib.Path, *args: str, extra_env: dict[str, str] | None = None
) -> str:
    ran = helpers.run_pivot(cwd, home, *args, extra_env=extra_env)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return ran.stdout + ran.stderr


def test_secret_digest_needs_the_key_and_a_shared_key_gives_equal_digests(
    tmp_path: pathlib.Path,
) -> None:
    """A 6-digit secret behind an unkeyed hash falls to a brute-force loop over the lock."""
    home = tmp_path / "home"
    own = tmp_path / "own"
    _secret_clone(own, home)
    _pivot(own, home, "repro")
    own_lock = (own / ".pivot" / "stages" / "smoke" / "fetch.lock").read_text()
    ignored = subprocess.run(
        ["git", "check-ignore", ".pivot/secret.key"], cwd=own, capture_output=True, text=True
    )
    shared_env = {"PIVOT_SECRET_KEY": "5e" * 32}
    shared_locks = list[str]()
    for name in ("first", "second"):
        clone = tmp_path / name
        _secret_clone(clone, home)
        _pivot(clone, home, "repro", extra_env=shared_env)
        shared_locks.append((clone / ".pivot" / "stages" / "smoke" / "fetch.lock").read_text())
        assert not (clone / ".pivot" / "secret.key").exists(), "the environment key is used"

    assert ignored.returncode == 0, "secret.key must be gitignored"
    assert (own / ".pivot" / "secret.key").stat().st_mode & 0o777 == 0o600
    digest = yaml.safe_load(own_lock)["params"]["pin"]
    assert digest.startswith("hmac:")
    unkeyed = {hashlib.sha256(str(n).encode()).hexdigest()[:16] for n in range(424000, 424500)}
    assert digest.removeprefix("hmac:") not in unkeyed
    assert shared_locks[0] == shared_locks[1]
    shared_digest = yaml.safe_load(shared_locks[0])["params"]["pin"]
    assert shared_digest == (
        "hmac:" + hmac.new(bytes.fromhex("5e" * 32), b"424242", hashlib.sha256).hexdigest()[:16]
    )


def test_qualified_params_keys_apply_to_base_and_variant(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(
            cwd, "smoke/scale:\n  factor: 3\nsmoke/scale@big:\n  factor: 300\n", "smoke"
        )

        result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "smoke/scale": {"factor": 3},
        "smoke/scale@big": {"factor": 300},
    }


def test_override_of_a_nested_model_used_twice_reaches_the_worker(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A model used by two fields gives the params schema definitions. The worker rebuilds the
    by-value pickled class from that schema, which pydantic before 2.13 got wrong: the class
    then refused every field, so any override failed in the worker."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
        (cwd / "params.yaml").write_text("smoke/train:\n  optimizer:\n    momentum: 0.5\n")
        (cwd / "pipeline.py").write_text(
            "import pandas as pd\nimport pydantic\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
            + "class Opt(pydantic.BaseModel):\n    momentum: float = 0.9\n\n\n"
            + "class TrainParams(StageParams):\n    optimizer: Opt = Opt()\n"
            + "    layers: list[Opt] = []\n\n\n"
            + "@stage\ndef train(numbers: pd.DataFrame, params: TrainParams) -> dict:\n"
            + '    return {"momentum": params.optimizer.momentum}\n\n\n'
            + 'with Pipeline("smoke") as pipeline:\n'
            + '    train(pipeline.input("numbers.csv"), params=TrainParams())\n'
        )

        result = runner.invoke(cli.cli, ["repro"])

        assert result.exit_code == 0, result.output
        assert yaml.safe_load((cwd / "data" / "smoke" / "train.yaml").read_text()) == {
            "momentum": 0.5
        }


def test_params_key_for_a_pipeline_not_loaded_is_skipped_at_debug(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running one pipeline of several, keys for the others are neither applied nor refused."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_scale_project(cwd, "a/scale:\n  factor: 2\nb/scale:\n  factor: 3\n", "a", "b")
        monkeypatch.chdir(cwd / "a")

        one = runner.invoke(cli.cli, ["--verbose", "params", "show", "--json"])
        every = runner.invoke(cli.cli, ["params", "show", "--all", "--json"])

    assert one.exit_code == 0, one.output
    assert (
        "params.yaml key 'b/scale': pipeline 'b' is not loaded in this run; skipping"
    ) in one.stderr, one.stderr
    # A key without '@' applies to every variant of the stage.
    assert json.loads(one.stdout) == {"a/scale": {"factor": 2}, "a/scale@big": {"factor": 2}}
    assert every.exit_code == 0, every.output
    assert json.loads(every.stdout) == {
        "a/scale": {"factor": 2},
        "a/scale@big": {"factor": 2},
        "b/scale": {"factor": 3},
        "b/scale@big": {"factor": 3},
    }


def test_params_keys_for_an_upstream_pipeline_reached_through_handles(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From report/, prep's reachable stage takes its key; prep's other stage is not loaded."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "numbers.csv").write_text("n\n1\n2\n")
        (cwd / "params.yaml").write_text(
            "prep/scale:\n  factor: 2\nprep/other:\n  factor: 5\nreport/total:\n  factor: 3\n"
        )
        (cwd / "prep").mkdir()
        (cwd / "prep" / "pipeline.py").write_text(
            "import pandas as pd\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
            + "class ScaleParams(StageParams):\n    factor: int = 10\n\n\n"
            + "@stage\ndef scale(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:\n"
            + "    return numbers * params.factor\n\n\n"
            + "@stage\ndef other(numbers: pd.DataFrame, params: ScaleParams) -> pd.DataFrame:\n"
            + "    return numbers\n\n\n"
            + 'with Pipeline("prep") as pipeline:\n'
            + '    numbers = pipeline.input("numbers.csv")\n'
            + "    scaled = scale(numbers, params=ScaleParams())\n"
            + "    other(numbers, params=ScaleParams())\n"
        )
        (cwd / "report").mkdir()
        (cwd / "report" / "pipeline.py").write_text(
            "import pandas as pd\n\nfrom pivot import Pipeline, StageParams, stage\n"
            + "from prep.pipeline import scaled\n\n\n"
            + "class TotalParams(StageParams):\n    factor: int = 1\n\n\n"
            + "@stage\ndef total(rows: pd.DataFrame, params: TotalParams) -> dict:\n"
            + '    return {"sum": int(rows.to_numpy().sum()) * params.factor}\n\n\n'
            + 'with Pipeline("report") as pipeline:\n'
            + "    total(scaled, params=TotalParams())\n"
        )
        monkeypatch.chdir(cwd / "report")

        result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"prep/scale": {"factor": 2}, "report/total": {"factor": 3}}


def _write_excluded_field_project(cwd: pathlib.Path, params_yaml: str) -> None:
    """Create two parameterized stages, one with an excluded required field."""
    (cwd / "params.yaml").write_text(params_yaml)
    (cwd / "pipeline.py").write_text(
        "import pydantic\n\nfrom pivot import Pipeline, StageParams, stage\n\n\n"
        + "class P(StageParams):\n"
        + "    hidden: str = pydantic.Field(exclude=True)\n"
        + "    visible: int = 0\n\n\n"
        + "class Q(StageParams):\n"
        + "    n: int = 0\n\n\n"
        + "@stage\ndef alpha(params: P) -> str:\n"
        + '    return f"{params.hidden}:{params.visible}"\n\n\n'
        + "@stage\ndef beta(params: Q) -> str:\n"
        + "    return str(params.n)\n\n\n"
        + 'with Pipeline("p") as pipeline:\n'
        + '    alpha(params=P(hidden="present"))\n'
        + "    beta(params=Q())\n"
    )


def test_override_for_one_stage_does_not_revalidate_an_unrelated_excluded_field(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An override for beta must not validate alpha's excluded required field."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_excluded_field_project(cwd, "p/beta:\n  n: 1\n")

        shown = runner.invoke(cli.cli, ["params", "show", "--json"])
        repro = runner.invoke(cli.cli, ["repro"])

    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["p/beta"] == {"n": 1}
    assert repro.exit_code == 0, repro.output
    assert (cwd / "data" / "p" / "beta.txt").read_text() == "1"


def test_override_for_excluded_field_stage_keeps_the_registered_value(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An alpha override merges onto its registered P instance instead of an excluded-field dump."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_excluded_field_project(cwd, "p/alpha:\n  visible: 7\n")

        repro = runner.invoke(cli.cli, ["repro"])

    assert repro.exit_code == 0, repro.output
    assert (cwd / "data" / "p" / "alpha.txt").read_text() == "present:7"
