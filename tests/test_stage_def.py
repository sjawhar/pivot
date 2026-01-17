# pyright: reportUnusedFunction=false, reportPrivateUsage=false
from __future__ import annotations

import pathlib  # noqa: TC003 - needed at runtime for stage_def.Dep[pathlib.Path]
import pickle
from typing import Any

import pandas
import pydantic
import pytest

from pivot import loaders, stage_def

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


def test_stage_def_discovers_deps() -> None:
    """StageDef should discover Dep descriptors via __set_name__."""

    class TrainParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data/train.csv", loaders.CSV())
        learning_rate: float = 0.01

    # Should have deps_specs class variable
    assert hasattr(TrainParams, "_deps_specs")
    assert "data" in TrainParams._deps_specs


def test_stage_def_discovers_outs() -> None:
    """StageDef should discover Out descriptors via __set_name__."""

    class TrainParams(stage_def.StageDef):
        model: stage_def.Out[dict[str, Any]] = stage_def.Out("models/model.pkl", loaders.Pickle())
        learning_rate: float = 0.01

    # Should have outs_specs class variable
    assert hasattr(TrainParams, "_outs_specs")
    assert "model" in TrainParams._outs_specs


def test_stage_def_extracts_loader_type() -> None:
    """StageDef should extract loader from Dep descriptor."""

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())

    spec = ProcessParams._deps_specs["data"]
    assert isinstance(spec.loader, loaders.CSV)


def test_stage_def_extracts_path_from_dep() -> None:
    """StageDef should extract file path from Dep descriptor."""

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data/input.csv", loaders.CSV())

    spec = ProcessParams._deps_specs["data"]
    assert spec.path == "data/input.csv"


def test_stage_def_class_access_returns_descriptor() -> None:
    """Class-level access to deps/outs should be via _deps_specs/_outs_specs."""

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())

    # Access descriptor via specs (Pydantic intercepts direct class attribute access)
    spec = ProcessParams._deps_specs["data"]
    assert isinstance(spec.descriptor, stage_def.Dep)  # pyright: ignore[reportAttributeAccessIssue] - descriptor stored in dataclass
    assert spec.descriptor.path == "input.csv"  # pyright: ignore[reportAttributeAccessIssue] - descriptor stored in dataclass
    assert spec.path == "input.csv"


def test_stage_def_get_deps_paths() -> None:
    """get_deps_paths() should return dict of name -> path."""

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())

    paths = ProcessParams.get_deps_paths()
    assert "data" in paths
    assert paths["data"] == "input.csv"


def test_stage_def_instance_access_before_load_raises() -> None:
    """Instance attribute access before _load_deps() should raise RuntimeError."""

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())

    params = ProcessParams()

    with pytest.raises(RuntimeError, match="not loaded"):
        _ = params.data


def test_stage_def_instance_access_after_load_returns_data(
    tmp_path: pathlib.Path,
) -> None:
    """Instance attribute access after _load_deps() should return loaded data."""
    # Create test CSV file
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("input.csv", loaders.CSV())

    params = ProcessParams()
    params._load_deps(tmp_path)

    # Should return loaded DataFrame
    df = params.data
    assert isinstance(df, pandas.DataFrame)
    assert list(df.columns) == ["a", "b"]


def test_stage_def_is_picklable() -> None:
    """StageDef pickling is tested in tests/test_stage_def_pickle.py.

    This test documents that inline (function-local) classes cannot be pickled
    by Python - this is a Python limitation, not a StageDef limitation.
    For real usage, StageDef subclasses must be module-level.
    """

    # Inline classes cannot be pickled - this is expected Python behavior
    class SimpleParams(stage_def.StageDef):
        threshold: float = 0.5

    params = SimpleParams(threshold=0.7)

    # Verify the limitation exists
    with pytest.raises(AttributeError, match="Can't get local object"):
        pickle.dumps(params)

    # Real pickle tests are in tests/test_stage_def_pickle.py with module-level classes


def test_stage_def_output_assignment() -> None:
    """StageDef should track output assignments via Out descriptor."""

    class ProcessParams(stage_def.StageDef):
        result: stage_def.Out[dict[str, int]] = stage_def.Out("output.json", loaders.JSON())

    params = ProcessParams()
    params.result = {"value": 42}

    # Should be stored in _outs_data
    assert "result" in params._outs_data
    assert params._outs_data["result"] == {"value": 42}


def test_stage_def_output_access_before_assignment_raises() -> None:
    """Accessing an Out before assignment should raise RuntimeError."""

    class ProcessParams(stage_def.StageDef):
        result: stage_def.Out[dict[str, int]] = stage_def.Out("output.json", loaders.JSON())

    params = ProcessParams()

    with pytest.raises(RuntimeError, match="not yet assigned"):
        _ = params.result


def test_stage_def_save_outs_raises_if_not_assigned(tmp_path: pathlib.Path) -> None:
    """_save_outs() should raise if output was declared but never assigned."""

    class ProcessParams(stage_def.StageDef):
        result: stage_def.Out[dict[str, int]] = stage_def.Out("output.json", loaders.JSON())

    params = ProcessParams()
    # Never assign params.result

    with pytest.raises(RuntimeError, match="never assigned"):
        params._save_outs(tmp_path)


def test_stage_def_save_outs_writes_file(tmp_path: pathlib.Path) -> None:
    """_save_outs() should write assigned outputs to files."""

    class ProcessParams(stage_def.StageDef):
        result: stage_def.Out[dict[str, int]] = stage_def.Out("output.json", loaders.JSON())

    params = ProcessParams()
    params.result = {"value": 42}
    params._save_outs(tmp_path)

    # Should have written the file
    output_file = tmp_path / "output.json"
    assert output_file.exists()
    assert "42" in output_file.read_text()


def test_stage_def_get_deps_paths_multiple() -> None:
    """get_deps_paths() should return dict of name -> path."""

    class ProcessParams(stage_def.StageDef):
        train: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data/train.csv", loaders.CSV())
        test: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data/test.csv", loaders.CSV())

    paths = ProcessParams.get_deps_paths()

    assert paths == {"train": "data/train.csv", "test": "data/test.csv"}


def test_stage_def_get_outs_paths() -> None:
    """get_outs_paths() should return dict of name -> path."""

    class ProcessParams(stage_def.StageDef):
        model: stage_def.Out[dict[str, Any]] = stage_def.Out("models/model.pkl", loaders.Pickle())
        metrics: stage_def.Out[dict[str, float]] = stage_def.Out("metrics.json", loaders.JSON())

    paths = ProcessParams.get_outs_paths()

    assert paths == {"model": "models/model.pkl", "metrics": "metrics.json"}


def test_stage_def_multiple_deps_loaded(tmp_path: pathlib.Path) -> None:
    """Multiple deps should all be loadable."""
    # Create test files
    (tmp_path / "train.csv").write_text("x,y\n1,2\n")
    (tmp_path / "test.csv").write_text("x,y\n3,4\n")

    class ProcessParams(stage_def.StageDef):
        train: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("train.csv", loaders.CSV())
        test: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("test.csv", loaders.CSV())

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert isinstance(params.train, pandas.DataFrame)
    assert isinstance(params.test, pandas.DataFrame)


def test_stage_def_multiple_outs_saved(tmp_path: pathlib.Path) -> None:
    """Multiple outs should all be saveable."""

    class ProcessParams(stage_def.StageDef):
        model: stage_def.Out[dict[str, int]] = stage_def.Out("model.json", loaders.JSON())
        metrics: stage_def.Out[dict[str, float]] = stage_def.Out("metrics.json", loaders.JSON())

    params = ProcessParams()
    params.model = {"weights": 1}
    params.metrics = {"accuracy": 0.95}
    params._save_outs(tmp_path)

    assert (tmp_path / "model.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_stage_def_json_loader_roundtrip(tmp_path: pathlib.Path) -> None:
    """JSON loader should work for deps and outs."""
    # Create input file
    input_file = tmp_path / "config.json"
    input_file.write_text('{"key": "value"}')

    class ProcessParams(stage_def.StageDef):
        config: stage_def.Dep[dict[str, str]] = stage_def.Dep("config.json", loaders.JSON())
        result: stage_def.Out[dict[str, str]] = stage_def.Out("result.json", loaders.JSON())

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.config == {"key": "value"}

    params.result = {"processed": "data"}
    params._save_outs(tmp_path)

    assert (tmp_path / "result.json").exists()


def test_stage_def_yaml_loader(tmp_path: pathlib.Path) -> None:
    """YAML loader should work for deps."""
    # Create YAML file
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("key: value\nlist:\n  - a\n  - b\n")

    class ProcessParams(stage_def.StageDef):
        config: stage_def.Dep[dict[str, Any]] = stage_def.Dep("config.yaml", loaders.YAML())

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.config == {"key": "value", "list": ["a", "b"]}


def test_stage_def_pickle_loader(tmp_path: pathlib.Path) -> None:
    """Pickle loader should work for deps and outs."""
    # Create pickle file
    pkl_file = tmp_path / "data.pkl"
    pkl_file.write_bytes(pickle.dumps({"complex": [1, 2, 3]}))

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[dict[str, list[int]]] = stage_def.Dep("data.pkl", loaders.Pickle())
        result: stage_def.Out[dict[str, int]] = stage_def.Out("result.pkl", loaders.Pickle())

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert params.data == {"complex": [1, 2, 3]}

    params.result = {"value": 42}
    params._save_outs(tmp_path)

    loaded = pickle.loads((tmp_path / "result.pkl").read_bytes())
    assert loaded == {"value": 42}


def test_stage_def_pathonly_loader(tmp_path: pathlib.Path) -> None:
    """PathOnly loader should return path for manual handling."""
    # Create a file
    data_file = tmp_path / "data.bin"
    data_file.write_bytes(b"binary data")

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pathlib.Path] = stage_def.Dep("data.bin", loaders.PathOnly())

    params = ProcessParams()
    params._load_deps(tmp_path)

    # Should return the path itself
    path = params.data
    assert path == tmp_path / "data.bin"
    assert path.exists()


def test_stage_def_csv_with_default_config(tmp_path: pathlib.Path) -> None:
    """CSV loader uses default config from annotation."""
    # Create CSV file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    class ProcessParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data.csv", loaders.CSV())

    params = ProcessParams()
    params._load_deps(tmp_path)

    assert list(params.data.columns) == ["a", "b"]


def test_stage_def_load_failure_clears_state(tmp_path: pathlib.Path) -> None:
    """If loading fails partway, state should be cleared for GC."""
    # Create only first file
    (tmp_path / "first.csv").write_text("a,b\n1,2\n")
    # Don't create second.csv

    class ProcessParams(stage_def.StageDef):
        first: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("first.csv", loaders.CSV())
        second: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("second.csv", loaders.CSV())

    params = ProcessParams()

    with pytest.raises(FileNotFoundError):
        params._load_deps(tmp_path)

    # Should have cleared any partial state
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = params.first


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


# ==============================================================================
# Inheritance tests
# ==============================================================================


def test_stage_def_child_inherits_parent_deps(tmp_path: pathlib.Path) -> None:
    """Child StageDef should inherit parent's deps."""
    # Create test files
    (tmp_path / "parent.csv").write_text("a,b\n1,2\n")
    (tmp_path / "child.csv").write_text("x,y\n3,4\n")

    class ParentParams(stage_def.StageDef):
        parent_data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("parent.csv", loaders.CSV())

    class ChildParams(ParentParams):
        child_data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("child.csv", loaders.CSV())

    # Child should have both deps
    assert "parent_data" in ChildParams.get_deps_paths()
    assert "child_data" in ChildParams.get_deps_paths()

    # Both should be loadable
    params = ChildParams()
    params._load_deps(tmp_path)

    assert isinstance(params.parent_data, pandas.DataFrame)
    assert isinstance(params.child_data, pandas.DataFrame)


def test_stage_def_child_inherits_parent_outs(tmp_path: pathlib.Path) -> None:
    """Child StageDef should inherit parent's outs."""

    class ParentParams(stage_def.StageDef):
        parent_out: stage_def.Out[dict[str, int]] = stage_def.Out("parent.json", loaders.JSON())

    class ChildParams(ParentParams):
        child_out: stage_def.Out[dict[str, int]] = stage_def.Out("child.json", loaders.JSON())

    # Child should have both outs
    assert "parent_out" in ChildParams.get_outs_paths()
    assert "child_out" in ChildParams.get_outs_paths()

    # Both should be saveable
    params = ChildParams()
    params.parent_out = {"parent": 1}
    params.child_out = {"child": 2}
    params._save_outs(tmp_path)

    assert (tmp_path / "parent.json").exists()
    assert (tmp_path / "child.json").exists()


def test_stage_def_child_without_new_deps_inherits_all(tmp_path: pathlib.Path) -> None:
    """Child StageDef without new deps should still inherit parent's deps."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")

    class ParentParams(stage_def.StageDef):
        data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data.csv", loaders.CSV())

    class ChildParams(ParentParams):
        # No new deps, just a new param
        threshold: float = 0.5

    # Child should inherit parent's deps
    assert "data" in ChildParams.get_deps_paths()

    # Should be loadable
    params = ChildParams()
    params._load_deps(tmp_path)
    assert isinstance(params.data, pandas.DataFrame)
