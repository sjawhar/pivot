# pyright: reportUnusedFunction=false

import pathlib

import pytest
import yaml

from pivot import dvc_compat, outputs, registry
from pivot.exceptions import DVCImportError, ExportError


# Module-level functions for export tests (can't export from __main__)
def exportable_stage() -> None:
    """A stage that can be exported."""
    pass


def exportable_with_params(learning_rate: float = 0.01, epochs: int = 100) -> None:
    """A stage with parameter defaults."""
    pass


# === _to_relative_path Helper Tests ===


def test_relative_path_unchanged(tmp_path: pathlib.Path) -> None:
    """Relative paths should pass through unchanged."""
    result = dvc_compat._to_relative_path("data/file.csv", tmp_path)
    assert result == "data/file.csv"


def test_absolute_inside_root_becomes_relative(tmp_path: pathlib.Path) -> None:
    """Absolute paths inside root become relative."""
    absolute = str(tmp_path / "data" / "file.csv")
    result = dvc_compat._to_relative_path(absolute, tmp_path)
    assert result == "data/file.csv"


def test_absolute_outside_root_stays_absolute(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Paths outside project root stay absolute with warning."""
    outside_path = "/some/other/path/file.csv"
    result = dvc_compat._to_relative_path(outside_path, tmp_path)
    assert result == outside_path
    assert "outside project root" in caplog.text


# === _generate_cmd Helper Tests ===


def test_generates_module_import_cmd() -> None:
    """Should generate python -c import command."""
    cmd = dvc_compat._generate_cmd(exportable_stage)
    # Module name may include 'tests.' prefix when run from pytest
    assert cmd.startswith("python -c 'from ")
    assert "import exportable_stage; exportable_stage()'" in cmd


def test_generate_cmd_raises_on_main_module() -> None:
    """Should raise ExportError for functions in __main__."""

    def local_func() -> None:
        pass

    # Simulate __main__ module
    local_func.__module__ = "__main__"

    with pytest.raises(ExportError, match="__main__"):
        dvc_compat._generate_cmd(local_func)


def test_generate_cmd_raises_on_lambda() -> None:
    """Should raise ExportError for lambda functions."""
    lamb = lambda: None  # noqa: E731
    with pytest.raises(ExportError, match="lambda"):
        dvc_compat._generate_cmd(lamb)


# === _extract_param_defaults Helper Tests ===


def test_extract_param_defaults() -> None:
    """Should extract parameter defaults from signature."""
    import inspect

    sig = inspect.signature(exportable_with_params)
    defaults = dvc_compat._extract_param_defaults(sig)
    assert defaults == {"learning_rate": 0.01, "epochs": 100}


def test_extract_param_defaults_no_defaults_returns_empty() -> None:
    """Should return empty dict if no defaults."""
    import inspect

    def no_defaults(x: int, y: str) -> None:
        pass

    sig = inspect.signature(no_defaults)
    defaults = dvc_compat._extract_param_defaults(sig)
    assert defaults == {}


def test_extract_param_defaults_partial() -> None:
    """Should only include parameters with defaults."""
    import inspect

    def partial(x: int, y: str = "default") -> None:
        pass

    sig = inspect.signature(partial)
    defaults = dvc_compat._extract_param_defaults(sig)
    assert defaults == {"y": "default"}


# === _build_out_entry Helper Tests ===


def test_build_out_entry_out_default_returns_string() -> None:
    """Out with default cache=True returns just the path string."""
    out = outputs.Out(path="model.pkl")
    result = dvc_compat._build_out_entry(out, "model.pkl")
    assert result == "model.pkl"


def test_build_out_entry_out_cache_false_returns_dict() -> None:
    """Out with cache=False returns dict with cache option."""
    out = outputs.Out(path="model.pkl", cache=False)
    result = dvc_compat._build_out_entry(out, "model.pkl")
    assert result == {"model.pkl": {"cache": False}}


def test_build_out_entry_metric_default_returns_string() -> None:
    """Metric with default cache=False returns just the path string."""
    metric = outputs.Metric(path="metrics.json")
    result = dvc_compat._build_out_entry(metric, "metrics.json")
    assert result == "metrics.json"


def test_build_out_entry_metric_cache_true_returns_dict() -> None:
    """Metric with cache=True returns dict with cache option."""
    metric = outputs.Metric(path="metrics.json", cache=True)
    result = dvc_compat._build_out_entry(metric, "metrics.json")
    assert result == {"metrics.json": {"cache": True}}


def test_build_out_entry_plot_with_options() -> None:
    """Plot with x/y/template returns dict with those options."""
    plot = outputs.Plot(path="loss.csv", x="epoch", y="loss", template="linear")
    result = dvc_compat._build_out_entry(plot, "loss.csv")
    assert result == {"loss.csv": {"x": "epoch", "y": "loss", "template": "linear"}}


def test_build_out_entry_persist_option() -> None:
    """Persist option should be included when True."""
    out = outputs.Out(path="model.pkl", persist=True)
    result = dvc_compat._build_out_entry(out, "model.pkl")
    assert result == {"model.pkl": {"persist": True}}


# === export_dvc_yaml Tests ===


def test_export_simple_stage(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should export basic stage to dvc.yaml."""
    # Mock project root
    monkeypatch.setattr("pivot.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(
        exportable_stage,
        name="my_stage",
        deps=[str(tmp_path / "input.txt")],
        outs=[str(tmp_path / "output.txt")],
    )

    dvc_yaml_path = tmp_path / "dvc.yaml"
    result = dvc_compat.export_dvc_yaml(dvc_yaml_path)

    assert "stages" in result
    assert "my_stage" in result["stages"]
    stage = result["stages"]["my_stage"]
    assert "cmd" in stage
    assert "exportable_stage" in stage["cmd"]
    assert stage["deps"] == ["input.txt"]
    assert stage["outs"] == ["output.txt"]

    # Verify file was written
    assert dvc_yaml_path.exists()
    with open(dvc_yaml_path) as f:
        written = yaml.safe_load(f)
    assert written == result


def test_export_with_rich_outputs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should separate Out/Metric/Plot into correct sections."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(
        exportable_stage,
        name="train",
        deps=[],
        outs=[
            outputs.Out(path=str(tmp_path / "model.pkl")),
            outputs.Metric(path=str(tmp_path / "metrics.json")),
            outputs.Plot(path=str(tmp_path / "loss.csv"), x="epoch", y="loss"),
        ],
    )

    result = dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml")
    stage = result["stages"]["train"]

    assert stage["outs"] == ["model.pkl"]
    assert stage["metrics"] == ["metrics.json"]
    assert stage["plots"] == [{"loss.csv": {"x": "epoch", "y": "loss"}}]


def test_export_generates_params_yaml(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should generate params.yaml with function defaults."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(
        exportable_with_params,
        name="train",
        deps=[],
        outs=[],
    )

    dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml")

    params_path = tmp_path / "params.yaml"
    assert params_path.exists()

    with open(params_path) as f:
        params = yaml.safe_load(f)
    assert params == {"train": {"learning_rate": 0.01, "epochs": 100}}


def test_export_references_params(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage should reference params from params.yaml."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(
        exportable_with_params,
        name="train",
        deps=[],
        outs=[],
    )

    result = dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml")
    stage = result["stages"]["train"]

    assert "params" in stage
    assert set(stage["params"]) == {"train.learning_rate", "train.epochs"}


def test_export_empty_registry_raises_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Should raise error when no stages registered."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    with pytest.raises(ExportError, match="No stages registered"):
        dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml")


def test_export_missing_stages_raises_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise error when requested stages don't exist."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(exportable_stage, name="stage1", deps=[], outs=[])

    with pytest.raises(ExportError, match="Stages not found.*nonexistent"):
        dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml", stages=["stage1", "nonexistent"])


def test_export_subset_of_stages(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should export only specified stages."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(exportable_stage, name="stage1", deps=[], outs=[])
    registry.REGISTRY.register(exportable_with_params, name="stage2", deps=[], outs=[])

    result = dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml", stages=["stage1"])

    assert "stage1" in result["stages"]
    assert "stage2" not in result["stages"]


def test_export_out_cache_false(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Out with cache=False should have cache: false in yaml."""
    monkeypatch.setattr("pivot.dvc_compat.project.get_project_root", lambda: tmp_path)

    registry.REGISTRY.register(
        exportable_stage,
        name="stage",
        deps=[],
        outs=[outputs.Out(path=str(tmp_path / "file.txt"), cache=False)],
    )

    result = dvc_compat.export_dvc_yaml(tmp_path / "dvc.yaml")
    assert result["stages"]["stage"]["outs"] == [{"file.txt": {"cache": False}}]


# === import_dvc_yaml Tests ===


def test_import_raises_without_dvc(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise DVCImportError if DVC not installed."""
    # Make DVC import fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "dvc.repo":
            raise ImportError("No module named 'dvc'")
        # Forward to original import - type checker can't verify dynamic import signature
        return original_import(name, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(builtins, "__import__", mock_import)

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("stages: {}")

    with pytest.raises(DVCImportError, match="DVC is required|No module named 'dvc'"):
        dvc_compat.import_dvc_yaml(dvc_yaml)


def test_import_raises_file_not_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Should raise DVCImportError if file doesn't exist."""
    # Need to make DVC importable first
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "dvc.repo":

            class MockRepo:
                def __init__(self, path: str) -> None:
                    pass

            class MockModule:
                Repo: type[MockRepo] = MockRepo

            return MockModule()
        # Forward to original import - type checker can't verify dynamic import signature
        return original_import(name, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(DVCImportError, match="not found"):
        dvc_compat.import_dvc_yaml(tmp_path / "nonexistent.yaml")


# === StageSpec Tests ===


def test_stage_spec_creation() -> None:
    """Should create StageSpec with all fields."""
    spec = dvc_compat.StageSpec(
        name="train",
        cmd="python train.py",
        deps=["data.csv"],
        outs=[outputs.Out(path="model.pkl")],
        params={"lr": 0.01},
        frozen=True,
        desc="Training stage",
    )

    assert spec.name == "train"
    assert spec.cmd == "python train.py"
    assert spec.deps == ["data.csv"]
    assert len(spec.outs) == 1
    assert spec.params == {"lr": 0.01}
    assert spec.frozen is True
    assert spec.desc == "Training stage"


def test_stage_spec_defaults() -> None:
    """StageSpec should have sensible defaults."""
    spec = dvc_compat.StageSpec(
        name="stage",
        cmd="echo hi",
        deps=[],
        outs=[],
        params={},
    )

    assert spec.frozen is False
    assert spec.desc is None
