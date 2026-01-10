# pyright: reportUnusedFunction=false, reportPrivateUsage=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportIncompatibleVariableOverride=false, reportUnknownArgumentType=false
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import pandas
import pydantic
import pytest

from pivot import loaders, stage_def

if TYPE_CHECKING:
    import pathlib

# ==============================================================================
# StageDef base class tests
# ==============================================================================


def test_stage_def_extends_base_model() -> None:
    """StageDef should extend pydantic.BaseModel."""
    assert issubclass(stage_def.StageDef, pydantic.BaseModel)


def test_stage_def_with_params_only() -> None:
    """StageDef with only params (no deps/outs) should work."""

    class SimpleParams(stage_def.StageDef):
        learning_rate: float = 0.01
        epochs: int = 10

    params = SimpleParams()
    assert params.learning_rate == 0.01
    assert params.epochs == 10


def test_stage_def_discovers_deps_class() -> None:
    """StageDef should discover nested deps class via __init_subclass__."""

    class TrainParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "data/train.csv"

        learning_rate: float = 0.01

    # Should have deps_specs class variable
    assert hasattr(TrainParams, "_deps_specs")
    assert "data" in TrainParams._deps_specs


def test_stage_def_discovers_outs_class() -> None:
    """StageDef should discover nested outs class via __init_subclass__."""

    class TrainParams(stage_def.StageDef):
        class outs:
            model: loaders.Pickle[dict[str, Any]] = "models/model.pkl"

        learning_rate: float = 0.01

    # Should have outs_specs class variable
    assert hasattr(TrainParams, "_outs_specs")
    assert "model" in TrainParams._outs_specs


def test_stage_def_extracts_loader_type() -> None:
    """StageDef should extract loader type from annotation."""

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "input.csv"

    spec = ProcessParams._deps_specs["data"]
    assert isinstance(spec.loader, loaders.CSV)


def test_stage_def_extracts_path_from_default() -> None:
    """StageDef should extract file path from default value."""

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "data/input.csv"

    spec = ProcessParams._deps_specs["data"]
    assert spec.path == "data/input.csv"


def test_stage_def_class_access_returns_descriptor() -> None:
    """Class attribute access should return descriptor (for framework)."""

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "input.csv"

    # Class access returns descriptor with path attribute
    descriptor = ProcessParams.deps.data
    assert hasattr(descriptor, "path")
    assert descriptor.path == "input.csv"


def test_stage_def_instance_access_before_load_raises() -> None:
    """Instance attribute access before _load_deps() should raise RuntimeError."""

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "input.csv"

    params = ProcessParams()

    with pytest.raises(RuntimeError, match="not loaded"):
        _ = params.deps.data


def test_stage_def_instance_access_after_load_returns_data(
    tmp_path: pathlib.Path,
) -> None:
    """Instance attribute access after _load_deps() should return loaded data."""
    # Create test CSV file
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "input.csv"

    params = ProcessParams()
    params._load_deps(tmp_path)

    # Should return loaded DataFrame
    df = params.deps.data
    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == ["a", "b"]


@pytest.mark.xfail(
    reason=(
        "Classes defined in test functions cannot be pickled by Python. "
        "In real usage, StageDef subclasses are module-level and can be pickled."
    )
)
def test_stage_def_is_picklable() -> None:
    """StageDef should be picklable (paths only, not loaded data)."""

    class SimpleParams(stage_def.StageDef):
        threshold: float = 0.5

    params = SimpleParams(threshold=0.7)
    pickled = pickle.dumps(params)
    restored = pickle.loads(pickled)

    assert restored.threshold == 0.7


def test_stage_def_output_assignment_tracking() -> None:
    """StageDef should track output assignments."""

    class ProcessParams(stage_def.StageDef):
        class outs:
            result: loaders.JSON[dict[str, int]] = "output.json"

    params = ProcessParams()
    params.outs.result = {"value": 42}

    # Should be marked as assigned
    assert "result" in params._assigned_outs


def test_stage_def_save_outs_raises_if_not_assigned(tmp_path: pathlib.Path) -> None:
    """_save_outs() should raise if output was declared but never assigned."""

    class ProcessParams(stage_def.StageDef):
        class outs:
            result: loaders.JSON[dict[str, int]] = "output.json"

    params = ProcessParams()
    # Never assign params.outs.result

    with pytest.raises(RuntimeError, match="never assigned"):
        params._save_outs(tmp_path)


def test_stage_def_save_outs_writes_file(tmp_path: pathlib.Path) -> None:
    """_save_outs() should write assigned outputs to files."""

    class ProcessParams(stage_def.StageDef):
        class outs:
            result: loaders.JSON[dict[str, int]] = "output.json"

    params = ProcessParams()
    params.outs.result = {"value": 42}
    params._save_outs(tmp_path)

    # Should have written the file
    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert "42" in output_file.read_text()


def test_stage_def_get_deps_paths() -> None:
    """get_deps_paths() should return dict of name -> path."""

    class ProcessParams(stage_def.StageDef):
        class deps:
            train: loaders.CSV[pandas.DataFrame] = "data/train.csv"
            test: loaders.CSV[pandas.DataFrame] = "data/test.csv"

    paths = ProcessParams.get_deps_paths()

    assert paths == {"train": "data/train.csv", "test": "data/test.csv"}


def test_stage_def_get_outs_paths() -> None:
    """get_outs_paths() should return dict of name -> path."""

    class ProcessParams(stage_def.StageDef):
        class outs:
            model: loaders.Pickle[dict[str, Any]] = "models/model.pkl"
            metrics: loaders.JSON[dict[str, float]] = "metrics.json"

    paths = ProcessParams.get_outs_paths()

    assert paths == {"model": "models/model.pkl", "metrics": "metrics.json"}


def test_stage_def_multiple_deps_loaded(tmp_path: pathlib.Path) -> None:
    """Multiple deps should all be loadable."""
    # Create test files
    (tmp_path / "train.csv").write_text("x,y\n1,2\n")
    (tmp_path / "test.csv").write_text("x,y\n3,4\n")

    class ProcessParams(stage_def.StageDef):
        class deps:
            train: loaders.CSV[pandas.DataFrame] = "train.csv"
            test: loaders.CSV[pandas.DataFrame] = "test.csv"

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert isinstance(params.deps.train, pandas.DataFrame)
    assert isinstance(params.deps.test, pandas.DataFrame)


def test_stage_def_multiple_outs_saved(tmp_path: pathlib.Path) -> None:
    """Multiple outs should all be saveable."""

    class ProcessParams(stage_def.StageDef):
        class outs:
            model: loaders.JSON[dict[str, int]] = "model.json"
            metrics: loaders.JSON[dict[str, float]] = "metrics.json"

    params = ProcessParams()
    params.outs.model = {"weights": 1}
    params.outs.metrics = {"accuracy": 0.95}
    params._save_outs(tmp_path)

    assert (tmp_path / "model.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_stage_def_json_loader_roundtrip(tmp_path: pathlib.Path) -> None:
    """JSON loader should work for deps and outs."""
    # Create input file
    input_file = tmp_path / "config.json"
    input_file.write_text('{"key": "value"}')

    class ProcessParams(stage_def.StageDef):
        class deps:
            config: loaders.JSON[dict[str, str]] = "config.json"

        class outs:
            result: loaders.JSON[dict[str, str]] = "result.json"

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.deps.config == {"key": "value"}

    params.outs.result = {"processed": "data"}
    params._save_outs(tmp_path)

    assert (tmp_path / "result.json").exists()


def test_stage_def_yaml_loader(tmp_path: pathlib.Path) -> None:
    """YAML loader should work for deps."""
    # Create YAML file
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("key: value\nlist:\n  - a\n  - b\n")

    class ProcessParams(stage_def.StageDef):
        class deps:
            config: loaders.YAML[dict[str, Any]] = "config.yaml"

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.deps.config == {"key": "value", "list": ["a", "b"]}


def test_stage_def_pickle_loader(tmp_path: pathlib.Path) -> None:
    """Pickle loader should work for deps and outs."""
    # Create pickle file
    pkl_file = tmp_path / "data.pkl"
    pkl_file.write_bytes(pickle.dumps({"complex": [1, 2, 3]}))

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.Pickle[dict[str, list[int]]] = "data.pkl"

        class outs:
            result: loaders.Pickle[dict[str, int]] = "result.pkl"

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.deps.data == {"complex": [1, 2, 3]}

    params.outs.result = {"value": 42}
    params._save_outs(tmp_path)

    loaded = pickle.loads((tmp_path / "result.pkl").read_bytes())
    assert loaded == {"value": 42}


def test_stage_def_pathonly_loader(tmp_path: pathlib.Path) -> None:
    """PathOnly loader should return path for manual handling."""
    # Create a file
    data_file = tmp_path / "data.bin"
    data_file.write_bytes(b"binary data")

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.PathOnly = "data.bin"

    params = ProcessParams()
    params._load_deps(tmp_path)

    # Should return the path itself
    path = params.deps.data
    assert path == tmp_path / "data.bin"
    assert path.exists()


def test_stage_def_csv_with_default_config(tmp_path: pathlib.Path) -> None:
    """CSV loader uses default config from annotation."""
    # Create CSV file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    class ProcessParams(stage_def.StageDef):
        class deps:
            data: loaders.CSV[pandas.DataFrame] = "data.csv"

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert list(params.deps.data.columns) == ["a", "b"]


def test_stage_def_load_failure_clears_state(tmp_path: pathlib.Path) -> None:
    """If loading fails partway, state should be cleared for GC."""
    # Create only first file
    (tmp_path / "first.csv").write_text("a,b\n1,2\n")
    # Don't create second.csv

    class ProcessParams(stage_def.StageDef):
        class deps:
            first: loaders.CSV[pandas.DataFrame] = "first.csv"
            second: loaders.CSV[pandas.DataFrame] = "second.csv"

    params = ProcessParams()

    with pytest.raises(FileNotFoundError):
        params._load_deps(tmp_path)

    # Should have cleared any partial state
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = params.deps.first


def test_stage_def_params_validation() -> None:
    """StageDef should validate params via pydantic."""

    class ProcessParams(stage_def.StageDef):
        threshold: float = pydantic.Field(gt=0, le=1.0)

    # Valid
    params = ProcessParams(threshold=0.5)
    assert params.threshold == 0.5

    # Invalid
    with pytest.raises(pydantic.ValidationError):
        ProcessParams(threshold=2.0)
