"""Discovery tests for compositional ``pipeline.py`` definitions.

YAML configuration and path-based parent/dependency pipeline discovery were
removed in #452. Cross-pipeline edges now arise from handles and ``include()``.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

from pivot import discovery


def _helper_write_pipeline(
    directory: pathlib.Path,
    pipeline_name: str = "test",
    stage_name: str = "build",
    *,
    input_name: str | None = None,
) -> pathlib.Path:
    """Write a module-level compositional pipeline and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    input_expr = f'pipeline.input("{input_name}", t=str)' if input_name is not None else ""
    parameter = "data: str" if input_name is not None else ""
    body = "return data" if input_name is not None else 'return {"value": 1}'
    pipeline_path = directory / "pipeline.py"
    pipeline_path.write_text(
        f'''\
from pivot.compose import Pipeline, stage


@stage
def {stage_name}({parameter}) -> {"str" if input_name is not None else "dict[str, int]"}:
    {body}


with Pipeline("{pipeline_name}") as pipeline:
    {stage_name}({input_expr})
'''
    )
    return pipeline_path


@pytest.fixture(autouse=True)
def _project_root_is_tmp_path(set_project_root: pathlib.Path) -> pathlib.Path:
    """Every pipeline here belongs to the project at tmp_path."""
    return set_project_root


def _helper_with_cwd(monkeypatch: pytest.MonkeyPatch, cwd: pathlib.Path) -> None:
    monkeypatch.setattr(pathlib.Path, "cwd", lambda: cwd)


# =============================================================================
# discover_pipeline
# =============================================================================


def test_discover_pipeline_returns_none_when_no_files(tmp_path: pathlib.Path) -> None:
    """Returns None when no pipeline.py exists."""
    assert discovery.discover_pipeline(project_root=tmp_path) is None


def test_discover_pipeline_ignores_directories_with_config_names(tmp_path: pathlib.Path) -> None:
    """A directory named pipeline.py is not a pipeline definition."""
    (tmp_path / "pipeline.py").mkdir()
    assert discovery.discover_pipeline(project_root=tmp_path) is None


def test_discover_pipeline_from_pipeline_py(tmp_path: pathlib.Path) -> None:
    """Loads a compositional Pipeline from pipeline.py."""
    _helper_write_pipeline(tmp_path, "test", "build")

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.name == "test"
    assert pipeline.list_stages() == ["test/build"]


def test_discover_pipeline_prefers_cwd_over_project_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline.py in the cwd inside the project wins over the root pipeline."""
    _helper_write_pipeline(tmp_path, "root", "root_stage")
    cwd = tmp_path / "subdir"
    _helper_write_pipeline(cwd, "child", "child_stage")
    _helper_with_cwd(monkeypatch, cwd)

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.name == "child"
    assert pipeline.list_stages() == ["child/child_stage"]


def test_discover_pipeline_cwd_invalid_config_raises_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid cwd pipeline.py is an error rather than falling back to root."""
    _helper_write_pipeline(tmp_path, "root", "root_stage")
    cwd = tmp_path / "subdir"
    cwd.mkdir()
    (cwd / "pipeline.py").write_text("pipeline = object()\n")
    _helper_with_cwd(monkeypatch, cwd)

    with pytest.raises(discovery.DiscoveryError, match="missing required methods"):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_cwd_equals_root_checks_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When cwd is root, discovery loads the root definition once."""
    _helper_write_pipeline(tmp_path, "root", "stage")
    _helper_with_cwd(monkeypatch, tmp_path)

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.name == "root"


def test_discover_pipeline_cwd_outside_project_uses_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cwd outside the project cannot override the root pipeline."""
    _helper_write_pipeline(tmp_path, "root", "stage")
    outside = tmp_path.parent / "outside"
    _helper_write_pipeline(outside, "outside", "stage")
    _helper_with_cwd(monkeypatch, outside)

    pipeline = discovery.discover_pipeline(project_root=tmp_path)

    assert pipeline is not None
    assert pipeline.name == "root"


def test_discover_pipeline_py_no_pipeline_variable(tmp_path: pathlib.Path) -> None:
    """A pipeline.py without a Pipeline object returns None."""
    (tmp_path / "pipeline.py").write_text("x = 1\n")

    assert discovery.discover_pipeline(project_root=tmp_path) is None


def test_discover_pipeline_missing_pipeline_variable(tmp_path: pathlib.Path) -> None:
    """A Pipeline assigned to the wrong module variable produces a useful error."""
    (tmp_path / "pipeline.py").write_text(
        "import pathlib\nfrom pivot.compose import Pipeline\n"
        + "wrong_name = Pipeline('wrong', root=pathlib.Path('.'))\n"
    )

    with pytest.raises(
        discovery.DiscoveryError, match="Found Pipeline instance named 'wrong_name'"
    ):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_wrong_type(tmp_path: pathlib.Path) -> None:
    """A non-Pipeline `pipeline` variable produces a useful error."""
    (tmp_path / "pipeline.py").write_text("pipeline = object()\n")

    with pytest.raises(discovery.DiscoveryError, match="missing required methods"):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_sys_exit_raises(tmp_path: pathlib.Path) -> None:
    """SystemExit from pipeline.py is wrapped as a DiscoveryError."""
    (tmp_path / "pipeline.py").write_text("raise SystemExit(42)\n")

    with pytest.raises(discovery.DiscoveryError, match=r"called sys.exit\(42\)"):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_runtime_error_raises(tmp_path: pathlib.Path) -> None:
    """Runtime errors in pipeline.py are wrapped as a DiscoveryError."""
    (tmp_path / "pipeline.py").write_text("raise RuntimeError('boom')\n")

    with pytest.raises(discovery.DiscoveryError, match="Failed to load"):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_reraises_discovery_error(tmp_path: pathlib.Path) -> None:
    """Existing DiscoveryError messages are preserved rather than double wrapped."""
    (tmp_path / "pipeline.py").write_text(
        "from pivot.discovery import DiscoveryError\nraise DiscoveryError('specific error')\n"
    )

    with pytest.raises(discovery.DiscoveryError, match="specific error"):
        discovery.discover_pipeline(project_root=tmp_path)


def test_discover_pipeline_raises_on_path_resolution_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path resolution errors are converted to DiscoveryError."""
    _helper_write_pipeline(tmp_path)

    def fail_resolve(self: pathlib.Path) -> pathlib.Path:
        raise OSError("resolution failed")

    monkeypatch.setattr(pathlib.Path, "resolve", fail_resolve)

    with pytest.raises(discovery.DiscoveryError, match="Failed to resolve paths"):
        discovery.discover_pipeline(project_root=tmp_path)


# =============================================================================
# load_pipeline_from_path
# =============================================================================


def test_load_pipeline_from_path_loads_pipeline_py(tmp_path: pathlib.Path) -> None:
    """Loads a valid compositional pipeline.py."""
    pipeline_path = _helper_write_pipeline(tmp_path, "loaded", "stage")

    pipeline = discovery.load_pipeline_from_path(pipeline_path)

    assert pipeline is not None
    assert pipeline.name == "loaded"
    assert pipeline.list_stages() == ["loaded/stage"]


def test_load_pipeline_from_path_returns_none_for_no_pipeline(tmp_path: pathlib.Path) -> None:
    """Returns None for a pipeline.py that defines no pipeline variable."""
    path = tmp_path / "pipeline.py"
    path.write_text("x = 1\n")

    assert discovery.load_pipeline_from_path(path) is None


def test_load_pipeline_from_path_returns_none_for_discovery_error(tmp_path: pathlib.Path) -> None:
    """Returns None rather than raising on an invalid pipeline.py."""
    path = tmp_path / "pipeline.py"
    path.write_text("pipeline = object()\n")

    assert discovery.load_pipeline_from_path(path) is None


def test_load_pipeline_from_path_logs_errors(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Logs a warning when a pipeline definition has a discovery issue."""
    path = tmp_path / "pipeline.py"
    path.write_text("pipeline = object()\n")

    with caplog.at_level(logging.WARNING):
        assert discovery.load_pipeline_from_path(path) is None

    assert "Pipeline discovery issue" in caplog.text


def test_load_pipeline_from_path_skips_runtime_error(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Runtime errors are debug-logged and skipped by best-effort path loading."""
    path = tmp_path / "pipeline.py"
    path.write_text("raise RuntimeError('boom')\n")

    with caplog.at_level(logging.DEBUG):
        assert discovery.load_pipeline_from_path(path) is None

    assert "Failed to load pipeline from" in caplog.text


def test_load_pipeline_from_path_unknown_file_type(tmp_path: pathlib.Path) -> None:
    """Returns None for a non-pipeline.py file."""
    path = tmp_path / "other.py"
    path.write_text("x = 1\n")

    assert discovery.load_pipeline_from_path(path) is None


# =============================================================================
# glob_all_pipelines
# =============================================================================


def test_glob_all_pipelines_finds_pipeline_py(tmp_path: pathlib.Path) -> None:
    """Finds pipeline.py recursively."""
    expected = _helper_write_pipeline(tmp_path / "sub")

    assert discovery.glob_all_pipelines(tmp_path) == [expected]


@pytest.mark.parametrize(
    "excluded",
    [
        ".pivot",
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "dist-packages",
        ".eggs",
    ],
)
def test_glob_all_pipelines_skips_excluded_dirs(tmp_path: pathlib.Path, excluded: str) -> None:
    """Does not scan generated, VCS, dependency or pivot state directories."""
    _helper_write_pipeline(tmp_path / excluded / "nested")
    expected = _helper_write_pipeline(tmp_path / "included")

    assert discovery.glob_all_pipelines(tmp_path) == [expected]


def test_glob_all_pipelines_finds_multiple_pipelines(tmp_path: pathlib.Path) -> None:
    """Finds and orders multiple pipeline.py definitions."""
    alpha = _helper_write_pipeline(tmp_path / "alpha", "alpha", "stage")
    beta = _helper_write_pipeline(tmp_path / "beta", "beta", "stage")

    assert discovery.glob_all_pipelines(tmp_path) == [alpha, beta]


def test_glob_all_pipelines_project_root_named_venv_is_not_excluded(tmp_path: pathlib.Path) -> None:
    """Exclusions are relative to project root, not its absolute parents."""
    root = tmp_path / "venv"
    pipeline_path = _helper_write_pipeline(root, "inside", "stage")

    assert discovery.glob_all_pipelines(root) == [pipeline_path]


# =============================================================================
# --all combining
# =============================================================================


def test_discover_all_pipelines_combines_stages(tmp_path: pathlib.Path) -> None:
    """--all combines every discovered pipeline's stages."""
    _helper_write_pipeline(tmp_path / "alpha", "alpha", "stage_a")
    _helper_write_pipeline(tmp_path / "beta", "beta", "stage_b")

    combined = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert combined is not None
    assert combined.name == "all"
    assert set(combined.list_stages()) == {"alpha/stage_a", "beta/stage_b"}


def test_discover_all_pipelines_keeps_every_stage_in_the_project_state_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Stages from every discovered pipeline keep their state in the project's .pivot."""
    _helper_write_pipeline(tmp_path / "alpha", "alpha", "stage")
    _helper_write_pipeline(tmp_path / "beta", "beta", "stage")

    combined = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert combined is not None
    assert combined.get_stage("alpha/stage")["state_dir"] == tmp_path.resolve() / ".pivot"
    assert combined.get_stage("beta/stage")["state_dir"] == tmp_path.resolve() / ".pivot"


def test_discover_all_pipelines_returns_none_when_empty(tmp_path: pathlib.Path) -> None:
    """--all returns None when no pipeline files exist."""
    assert discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True) is None


def test_discover_all_pipelines_skips_failed_loads(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """--all skips invalid modules while retaining valid sibling pipelines."""
    _helper_write_pipeline(tmp_path / "valid", "valid", "stage")
    bad = tmp_path / "invalid"
    bad.mkdir()
    (bad / "pipeline.py").write_text("pipeline = object()\n")

    with caplog.at_level(logging.WARNING):
        combined = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert combined is not None
    assert combined.list_stages() == ["valid/stage"]
    assert "failed to load pipeline" in caplog.text


def test_discover_all_pipelines_debugs_unresolved_identity_deps(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """--all logs unresolved identity dependencies at debug level."""
    _helper_write_pipeline(tmp_path / "alpha", "alpha", "consume", input_name="missing.txt")

    with caplog.at_level(logging.DEBUG):
        combined = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert combined is not None
    assert combined.list_stages() == ["alpha/consume"]
    assert "dependency path(s) not produced" in caplog.text


def test_discover_all_logs_per_pipeline_details(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """--all logs a pipeline name, path and stage count for each loaded pipeline."""
    alpha = _helper_write_pipeline(tmp_path / "alpha", "alpha", "stage_a")
    beta = _helper_write_pipeline(tmp_path / "beta", "beta", "stage_b")

    with caplog.at_level(logging.INFO):
        combined = discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True)

    assert combined is not None
    assert f"alpha: {alpha} (1 stages)" in caplog.text
    assert f"beta: {beta} (1 stages)" in caplog.text


def test_discover_all_pipelines_returns_none_when_all_fail(tmp_path: pathlib.Path) -> None:
    """--all returns None when every candidate fails discovery."""
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    (alpha / "pipeline.py").write_text("pipeline = object()\n")

    assert discovery.discover_pipeline(project_root=tmp_path, all_pipelines=True) is None


# Existing pipeline discovery no longer reads YAML or walks parent/sibling
# definitions to resolve path-based deps. Those tests were deleted: compositional
# cross-pipeline dependencies use ArtifactHandles and Pipeline.include().
