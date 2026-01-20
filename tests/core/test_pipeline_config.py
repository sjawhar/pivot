"""Tests for pivot.yaml configuration loading and stage registration."""

from __future__ import annotations

import pathlib
import shutil
import sys
from typing import TYPE_CHECKING

import pytest

from pivot import registry
from pivot.pipeline import yaml as pipeline_config

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "pipeline_config"


# =============================================================================
# Fixture Helpers
# =============================================================================


@pytest.fixture
def simple_pipeline(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Copy simple pipeline fixture to tmp_path and set up imports."""
    fixture_dir = FIXTURES_DIR / "simple"
    shutil.copytree(fixture_dir, tmp_path, dirs_exist_ok=True)

    # Create data directory structure
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "raw.csv").write_text("id,value\n1,10\n2,20\n")
    (tmp_path / "models").mkdir(exist_ok=True)

    # Set project root so paths resolve correctly
    mocker.patch("pivot.project._project_root_cache", tmp_path)

    # Clear any cached stages module from previous tests
    if "stages" in sys.modules:
        del sys.modules["stages"]

    # Add fixture dir to sys.path so stages module can be imported
    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    if "stages" in sys.modules:
        del sys.modules["stages"]


@pytest.fixture
def params_pipeline(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Copy params pipeline fixture to tmp_path and set up imports."""
    fixture_dir = FIXTURES_DIR / "with_params"
    shutil.copytree(fixture_dir, tmp_path, dirs_exist_ok=True)

    # Create data directory structure
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "raw.csv").write_text("id,value\n1,10\n2,20\n")
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "metrics").mkdir(exist_ok=True)

    # Set project root
    mocker.patch("pivot.project._project_root_cache", tmp_path)

    # Clear any cached stages module from previous tests
    if "stages" in sys.modules:
        del sys.modules["stages"]

    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    if "stages" in sys.modules:
        del sys.modules["stages"]


@pytest.fixture
def matrix_pipeline(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Copy matrix pipeline fixture to tmp_path and set up imports."""
    fixture_dir = FIXTURES_DIR / "with_matrix"
    shutil.copytree(fixture_dir, tmp_path, dirs_exist_ok=True)

    # Create data directory structure
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "raw.csv").write_text("id,value\n1,10\n2,20\n")
    (tmp_path / "data" / "gpt_tokenizer.json").write_text("{}")
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "bert.yaml").write_text("model: bert")
    (tmp_path / "configs" / "gpt.yaml").write_text("model: gpt")
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "metrics").mkdir(exist_ok=True)

    # Set project root
    mocker.patch("pivot.project._project_root_cache", tmp_path)

    # Clear any cached stages module from previous tests
    if "stages" in sys.modules:
        del sys.modules["stages"]

    sys.path.insert(0, str(tmp_path))
    yield tmp_path
    sys.path.remove(str(tmp_path))
    if "stages" in sys.modules:
        del sys.modules["stages"]


# =============================================================================
# Basic Loading Tests
# =============================================================================


def test_load_simple_config(simple_pipeline: pathlib.Path) -> None:
    """Load a simple pivot.yaml with two stages."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    config = pipeline_config.load_pipeline_file(pipeline_file)

    assert len(config.stages) == 2
    assert "preprocess" in config.stages
    assert "train" in config.stages


def test_load_pipeline_file_parses_stage_fields(simple_pipeline: pathlib.Path) -> None:
    """Stage config contains python, deps, and outs fields."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    config = pipeline_config.load_pipeline_file(pipeline_file)

    preprocess = config.stages["preprocess"]
    assert preprocess.python == "stages.preprocess"
    assert preprocess.deps == {"raw": "data/raw.csv"}
    assert preprocess.outs == {"clean": "data/clean.csv"}


def test_load_pipeline_file_with_metrics(params_pipeline: pathlib.Path) -> None:
    """Stage config with metrics field is parsed correctly."""
    pipeline_file = params_pipeline / "pivot.yaml"
    config = pipeline_config.load_pipeline_file(pipeline_file)

    train = config.stages["train"]
    assert train.metrics == {"train": "metrics/train.json"}


def test_load_pipeline_file_with_params(params_pipeline: pathlib.Path) -> None:
    """Stage config with params overrides is parsed correctly."""
    pipeline_file = params_pipeline / "pivot.yaml"
    config = pipeline_config.load_pipeline_file(pipeline_file)

    train = config.stages["train"]
    assert train.params["learning_rate"] == 0.05
    assert train.params["epochs"] == 50


# =============================================================================
# Stage Registration Tests
# =============================================================================


def test_register_simple_stages(simple_pipeline: pathlib.Path) -> None:
    """Register stages from simple pivot.yaml into registry."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()
    assert "preprocess" in stages
    assert "train" in stages


def test_registered_stage_has_correct_deps(simple_pipeline: pathlib.Path) -> None:
    """Registered stage has correct dependencies."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    # Deps should be normalized to absolute paths
    assert any("data/raw.csv" in dep for dep in info["deps_paths"])


def test_registered_stage_has_correct_outs(simple_pipeline: pathlib.Path) -> None:
    """Registered stage has correct outputs."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    assert any("data/clean.csv" in out for out in info["outs_paths"])


def test_registered_stage_function_is_callable(simple_pipeline: pathlib.Path) -> None:
    """Registered stage function is the actual imported function."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    assert callable(info["func"])
    assert info["func"].__name__ == "preprocess"


# =============================================================================
# Params Introspection Tests
# =============================================================================


def test_params_introspected_from_signature(params_pipeline: pathlib.Path) -> None:
    """Params class is introspected from function signature."""
    pipeline_file = params_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train")
    params = info["params"]
    assert params is not None
    assert params.__class__.__name__ == "TrainParams"


def test_params_values_from_yaml_override_defaults(params_pipeline: pathlib.Path) -> None:
    """Params values from pivot.yaml override class defaults."""
    pipeline_file = params_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train")
    params = info["params"]
    assert params is not None, "Expected params to be set"

    # These should be overridden by pivot.yaml
    assert params.model_dump()["learning_rate"] == 0.05
    assert params.model_dump()["epochs"] == 50
    # This should be the class default (not in pivot.yaml)
    assert params.model_dump()["batch_size"] == 32


def test_stage_without_params_has_none(simple_pipeline: pathlib.Path) -> None:
    """Stage without params parameter has params=None."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    assert info["params"] is None


def test_error_if_params_in_yaml_but_no_signature(simple_pipeline: pathlib.Path) -> None:
    """Error if pivot.yaml has params but function has no params parameter."""
    # Modify the config to add params to preprocess (which has no params parameter)
    pipeline_file = simple_pipeline / "pivot.yaml"
    config_text = pipeline_file.read_text()
    config_text = config_text.replace(
        "python: stages.preprocess",
        "python: stages.preprocess\n    params:\n      foo: bar",
    )
    pipeline_file.write_text(config_text)

    with pytest.raises(pipeline_config.PipelineConfigError, match="has no StageParams parameter"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_error_if_params_parameter_has_no_type_hint(
    params_pipeline: pathlib.Path,
) -> None:
    """Error if function has params parameter without type hint."""
    # Create a stage with untyped params
    stages_file = params_pipeline / "stages.py"
    content = stages_file.read_text()
    # Remove type hint from params parameter (multi-line signature)
    content = content.replace("params: TrainParams,", "params,")
    stages_file.write_text(content)

    # Need to reload the module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_file = params_pipeline / "pivot.yaml"
    # With type-based detection, untyped params won't be detected as StageParams
    with pytest.raises(pipeline_config.PipelineConfigError, match="has no StageParams parameter"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


# =============================================================================
# Matrix Expansion Tests
# =============================================================================


def test_matrix_expands_to_variants(matrix_pipeline: pathlib.Path) -> None:
    """Matrix config expands to multiple variant stages."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()

    # Should have preprocess + 4 train variants (2 models x 2 datasets)
    assert "preprocess" in stages
    assert "train@bert_swe" in stages
    assert "train@bert_human" in stages
    assert "train@gpt_swe" in stages
    assert "train@gpt_human" in stages
    assert len(stages) == 5


def test_matrix_variant_has_interpolated_deps(matrix_pipeline: pathlib.Path) -> None:
    """Matrix variant has ${dim} interpolated in deps."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@bert_swe")
    deps_str = " ".join(info["deps_paths"])

    assert "configs/bert.yaml" in deps_str
    # Should NOT have ${model} or ${dataset} - should be interpolated
    assert "${model}" not in deps_str
    assert "${dataset}" not in deps_str


def test_matrix_variant_has_interpolated_outs(matrix_pipeline: pathlib.Path) -> None:
    """Matrix variant has ${dim} interpolated in outs."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@bert_swe")
    outs_str = " ".join(info["outs_paths"])

    assert "models/bert_swe.pkl" in outs_str
    assert "${model}" not in outs_str


def test_matrix_variant_has_interpolated_params(matrix_pipeline: pathlib.Path) -> None:
    """Matrix variant has ${dim} interpolated in params values."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@bert_swe")
    params = info["params"]
    assert params is not None

    assert params.model_dump()["model_type"] == "bert"


def test_matrix_dict_dimension_applies_overrides(matrix_pipeline: pathlib.Path) -> None:
    """Dict dimension applies overrides to specific variants."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert_swe")
    gpt_info = registry.REGISTRY.get("train@gpt_swe")
    bert_params = bert_info["params"]
    gpt_params = gpt_info["params"]
    assert bert_params is not None
    assert gpt_params is not None

    # bert has hidden_size=768, gpt has hidden_size=1024
    assert bert_params.model_dump()["hidden_size"] == 768
    assert gpt_params.model_dump()["hidden_size"] == 1024


def test_matrix_list_dimension_uses_value_as_key(matrix_pipeline: pathlib.Path) -> None:
    """List dimension uses primitive value as key (no overrides)."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    # Both swe and human variants should exist
    assert "train@bert_swe" in registry.REGISTRY.list_stages()
    assert "train@bert_human" in registry.REGISTRY.list_stages()


# =============================================================================
# DAG Building Tests
# =============================================================================


def test_dag_built_from_registered_stages(simple_pipeline: pathlib.Path) -> None:
    """DAG can be built from registered stages."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    dag = registry.REGISTRY.build_dag()

    # train depends on preprocess (via data/clean.csv)
    assert dag.has_edge("train", "preprocess")


def test_matrix_dag_has_correct_dependencies(matrix_pipeline: pathlib.Path) -> None:
    """Matrix variants have correct dependencies in DAG."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    dag = registry.REGISTRY.build_dag()

    # All train variants depend on preprocess (via data/clean.csv)
    assert dag.has_edge("train@bert_swe", "preprocess")
    assert dag.has_edge("train@gpt_human", "preprocess")


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_load_nonexistent_file_raises(tmp_path: pathlib.Path) -> None:
    """Loading a non-existent file raises PipelineConfigError."""
    with pytest.raises(pipeline_config.PipelineConfigError, match="not found"):
        pipeline_config.load_pipeline_file(tmp_path / "nonexistent.yaml")


def test_load_empty_file_raises(tmp_path: pathlib.Path) -> None:
    """Loading an empty file raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text("")

    with pytest.raises(pipeline_config.PipelineConfigError, match="empty"):
        pipeline_config.load_pipeline_file(pipeline_file)


def test_load_invalid_yaml_raises(tmp_path: pathlib.Path) -> None:
    """Loading invalid YAML structure raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text("stages: not_a_dict")

    with pytest.raises(pipeline_config.PipelineConfigError, match="Invalid"):
        pipeline_config.load_pipeline_file(pipeline_file)


def test_import_function_invalid_path_raises(tmp_path: pathlib.Path) -> None:
    """Import path without dot raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: no_module_path
    outs:
      out: out.txt
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="module.function"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_import_nonexistent_module_raises(tmp_path: pathlib.Path) -> None:
    """Importing from non-existent module raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: nonexistent_module_xyz.func
    outs:
      out: out.txt
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="import module"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_import_nonexistent_function_raises(tmp_path: pathlib.Path) -> None:
    """Importing non-existent function raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: os.nonexistent_function_xyz
    outs:
      out: out.txt
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="no function"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_import_non_callable_raises(tmp_path: pathlib.Path) -> None:
    """Importing a non-callable raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: os.name
    outs:
      out: out.txt
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="not callable"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


# =============================================================================
# Output Options Tests
# =============================================================================


def test_output_with_cache_false(simple_pipeline: pathlib.Path) -> None:
    """Output with cache: false option is parsed correctly."""
    from pivot import outputs

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    deps:
      raw: data/raw.csv
    outs:
      clean: {path: data/clean.csv, cache: false}
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    out = info["outs"][0]
    assert isinstance(out, outputs.Out)
    assert out.cache is False


def test_plot_with_options(simple_pipeline: pathlib.Path) -> None:
    """Multiple outputs defined via annotations are registered correctly."""
    # Create stages file with multiple outputs in annotations
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class PreprocessOutputs(TypedDict):
    clean: Annotated[pathlib.Path, outputs.Out("data/clean.csv", loaders.PathOnly())]
    curve: Annotated[pathlib.Path, outputs.Out("plots/curve.json", loaders.PathOnly())]

def preprocess(
    raw: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> PreprocessOutputs:
    return {"clean": pathlib.Path("data/clean.csv"), "curve": pathlib.Path("plots/curve.json")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
"""
    )

    # Clear cached module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    # Should have 2 outputs registered
    assert len(info["outs"]) == 2
    paths = " ".join(info["outs_paths"])
    assert "data/clean.csv" in paths
    assert "plots/curve.json" in paths


# =============================================================================
# Matrix Error Tests
# =============================================================================


def test_matrix_empty_dimension_list_raises(simple_pipeline: pathlib.Path) -> None:
    """Empty matrix dimension list raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      clean: data/clean.csv
    outs:
      out: models/out.pkl
    matrix:
      model: []
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="empty"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_empty_dimension_dict_raises(simple_pipeline: pathlib.Path) -> None:
    """Empty matrix dimension dict raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      clean: data/clean.csv
    outs:
      out: models/out.pkl
    matrix:
      model: {}
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="empty"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_unresolved_variable_raises(simple_pipeline: pathlib.Path) -> None:
    """Unresolved ${var} in deps/outs raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      data: "data/${unknown}.csv"
    outs:
      out: models/out.pkl
    matrix:
      model:
        - bert
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="unresolved"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_interpolates_plots(simple_pipeline: pathlib.Path) -> None:
    """Matrix ${var} interpolation works in plots section."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    plot: Annotated[pathlib.Path, outputs.Plot("plots/default.png", loaders.PathOnly())]

def train(
    data: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"plot": pathlib.Path("plots/default.png")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    plots:
      plot: "plots/${model}_curve.png"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Plot paths should be interpolated
    assert any("plots/bert_curve.png" in p for p in bert_info["outs_paths"])
    assert any("plots/gpt_curve.png" in p for p in gpt_info["outs_paths"])
    # Should NOT have unresolved variables
    assert "${model}" not in " ".join(bert_info["outs_paths"])


def test_matrix_interpolates_metrics(simple_pipeline: pathlib.Path) -> None:
    """Matrix ${var} interpolation works in metrics section."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    metrics: Annotated[dict, outputs.Metric("metrics/default.json")]

def train(
    data: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"metrics": {"loss": 0.5}}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    metrics:
      metrics: "metrics/${model}_results.json"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Metric paths should be interpolated
    assert any("metrics/bert_results.json" in p for p in bert_info["outs_paths"])
    assert any("metrics/gpt_results.json" in p for p in gpt_info["outs_paths"])
    # Should NOT have unresolved variables
    assert "${model}" not in " ".join(bert_info["outs_paths"])


def test_matrix_unresolved_variable_in_params_raises(simple_pipeline: pathlib.Path) -> None:
    """Unresolved ${var} in params raises PipelineConfigError."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
import pydantic
from pivot import loaders, outputs, stage_def

class TrainParams(stage_def.StageParams):
    config_path: str

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/out.pkl", loaders.PathOnly())]

def train(
    params: TrainParams,
    data: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/out.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    params:
      config_path: "configs/${unknown}/model.yaml"
    matrix:
      model:
        - bert
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="unresolved"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_name_template_missing_dimensions_raises(
    simple_pipeline: pathlib.Path,
) -> None:
    """Name template missing matrix dimensions raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  "train@{model}":
    python: stages.train
    deps:
      clean: data/clean.csv
    outs:
      model: "models/${model}_${dataset}.pkl"
    matrix:
      model:
        - bert
        - gpt
      dataset:
        - swe
        - human
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="missing dimensions"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_name_template_unknown_variables_raises(
    simple_pipeline: pathlib.Path,
) -> None:
    """Name template with unknown variables raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  "train@{model}_{unknown}":
    python: stages.train
    deps:
      clean: data/clean.csv
    outs:
      model: "models/${model}.pkl"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="unknown variables"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_matrix_name_with_at_but_no_template_raises(
    simple_pipeline: pathlib.Path,
) -> None:
    """Name with @ but no template variables raises PipelineConfigError."""
    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train@variant:
    python: stages.train
    deps:
      clean: data/clean.csv
    outs:
      model: "models/${model}.pkl"
    matrix:
      model:
        - bert
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="no template"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


# =============================================================================
# Variants (Python Escape Hatch) Tests
# =============================================================================


def test_variants_function_registers_stages(
    simple_pipeline: pathlib.Path,
) -> None:
    """variants function returns list of dicts with params to register."""
    # Create a variants generator function with annotation-based train function
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs, stage_def

class TrainParams(stage_def.StageParams):
    learning_rate: float = 0.01

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: TrainParams,
    data: Annotated[pathlib.Path, outputs.Dep("data/input.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}

def get_variants():
    return [
        {"name": "v1", "params": {"learning_rate": 0.01}, "outs": {"model": "models/model_v1.pkl"}},
        {"name": "v2", "params": {"learning_rate": 0.001}, "outs": {"model": "models/model_v2.pkl"}},
    ]
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    variants: stages.get_variants
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()
    assert "train@v1" in stages
    assert "train@v2" in stages


def test_variants_function_not_list_raises(
    simple_pipeline: pathlib.Path,
) -> None:
    """variants function returning non-list raises PipelineConfigError."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass

def get_variants():
    return "not a list"
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    variants: stages.get_variants
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="must return a list"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_variants_function_unknown_keys_raises(
    simple_pipeline: pathlib.Path,
) -> None:
    """variants function with unknown keys in dict raises PipelineConfigError."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def train(
    data: Annotated[pathlib.Path, outputs.Dep("data/input.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}

def get_variants():
    return [
        {"name": "v1", "parmas": {"learning_rate": 0.01}},  # typo: "parmas" not "params"
    ]
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    variants: stages.get_variants
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="unknown keys.*parmas"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


# =============================================================================
# Matrix Override Tests (Additional Coverage)
# =============================================================================


def test_matrix_metrics_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix path interpolation works for multiple outputs."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]
    metrics: Annotated[pathlib.Path, outputs.Out("metrics/base.json", loaders.PathOnly())]

def preprocess():
    pass

def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl"), "metrics": pathlib.Path("metrics/base.json")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/${model}.pkl"
      metrics: "metrics/${model}_base.json"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    # Clear cached module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Each variant has 2 outputs with interpolated paths
    assert len(bert_info["outs"]) == 2
    bert_paths = " ".join(bert_info["outs_paths"])
    assert "models/bert.pkl" in bert_paths
    assert "bert_base.json" in bert_paths

    assert len(gpt_info["outs"]) == 2
    gpt_paths = " ".join(gpt_info["outs_paths"])
    assert "models/gpt.pkl" in gpt_paths
    assert "gpt_base.json" in gpt_paths


def test_matrix_plots_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension overrides can add additional outputs (plots use case)."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]
    plot: Annotated[pathlib.Path, outputs.Out("plots/base.json", loaders.PathOnly())]

def preprocess():
    pass

def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl"), "plot": pathlib.Path("plots/base.json")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/${model}.pkl"
      plot: "plots/${model}_base.json"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    # Clear cached module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Each variant has 2 outputs with interpolated paths
    assert len(bert_info["outs"]) == 2
    bert_paths = " ".join(bert_info["outs_paths"])
    assert "models/bert.pkl" in bert_paths
    assert "bert_base.json" in bert_paths

    assert len(gpt_info["outs"]) == 2
    gpt_paths = " ".join(gpt_info["outs_paths"])
    assert "models/gpt.pkl" in gpt_paths
    assert "gpt_base.json" in gpt_paths


def test_matrix_mutex_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override mutex list (replacement only)."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/${model}.pkl"
    mutex:
      - gpu
    matrix:
      model:
        bert:
          mutex:
            - cpu
        gpt:
          mutex:
            - memory
            - disk
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Both variants have their mutex list replaced (not merged with base)
    assert bert_info["mutex"] == ["cpu"]
    assert "memory" in gpt_info["mutex"]
    assert "disk" in gpt_info["mutex"]
    assert "gpu" not in gpt_info["mutex"]  # Base was replaced, not merged


def test_matrix_outs_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override out paths with interpolation."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    base: Annotated[pathlib.Path, outputs.Out("models/base.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"base": pathlib.Path("models/base.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      base: "models/${model}_base.pkl"
    matrix:
      model:
        - bert
        - gpt
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # Each variant has its out path interpolated
    assert len(bert_info["outs_paths"]) == 1
    assert "bert_base.pkl" in bert_info["outs_paths"][0]

    assert len(gpt_info["outs_paths"]) == 1
    assert "gpt_base.pkl" in gpt_info["outs_paths"][0]


def test_matrix_interpolates_nested_params(simple_pipeline: pathlib.Path) -> None:
    """Matrix interpolates ${dim} in nested params structures."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, Any, TypedDict
from pivot import loaders, outputs, stage_def

class TrainParams(stage_def.StageParams):
    config: dict[str, Any]

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: TrainParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/${model}.pkl"
    params:
      config:
        model_name: "${model}"
        paths:
          - "path/${model}/a"
          - "path/${model}/b"
    matrix:
      model:
        - bert
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@bert")
    params = info["params"]
    assert params is not None

    config = params.model_dump()["config"]
    assert config["model_name"] == "bert"
    assert config["paths"] == ["path/bert/a", "path/bert/b"]


def test_matrix_interpolates_output_dict_keys(simple_pipeline: pathlib.Path) -> None:
    """Matrix interpolates ${dim} in output dict with options."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: {path: "models/${model}.pkl", cache: false}
    matrix:
      model:
        - bert
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@bert")
    assert "models/bert.pkl" in info["outs_paths"][0]


# =============================================================================
# Non-String Matrix Dimension Tests
# =============================================================================


def test_matrix_boolean_values(simple_pipeline: pathlib.Path) -> None:
    """Boolean matrix values generate correct variants and preserve type in params."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs, stage_def

class FlagParams(stage_def.StageParams):
    enabled: bool

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: FlagParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/${flag}.pkl"
    params:
      enabled: "${flag}"
    matrix:
      flag:
        - true
        - false
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()
    assert "train@True" in stages
    assert "train@False" in stages

    true_info = registry.REGISTRY.get("train@True")
    false_info = registry.REGISTRY.get("train@False")

    # Params should preserve boolean type
    assert true_info["params"] is not None
    assert false_info["params"] is not None
    true_params = true_info["params"].model_dump()
    false_params = false_info["params"].model_dump()
    assert true_params["enabled"] is True
    assert false_params["enabled"] is False

    # Paths should use string conversion
    assert "models/True.pkl" in true_info["outs_paths"][0]
    assert "models/False.pkl" in false_info["outs_paths"][0]


def test_matrix_integer_values(simple_pipeline: pathlib.Path) -> None:
    """Integer matrix values generate correct variants and preserve type in params."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs, stage_def

class SizeParams(stage_def.StageParams):
    batch_size: int

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: SizeParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/batch_${size}.pkl"
    params:
      batch_size: "${size}"
    matrix:
      size:
        - 16
        - 32
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()
    assert "train@16" in stages
    assert "train@32" in stages

    info_16 = registry.REGISTRY.get("train@16")
    info_32 = registry.REGISTRY.get("train@32")

    # Params should preserve int type
    assert info_16["params"] is not None
    params_16 = info_16["params"].model_dump()
    assert params_16["batch_size"] == 16
    assert isinstance(params_16["batch_size"], int)

    assert info_32["params"] is not None
    params_32 = info_32["params"].model_dump()
    assert params_32["batch_size"] == 32

    # Paths should use string conversion
    assert "models/batch_16.pkl" in info_16["outs_paths"][0]
    assert "models/batch_32.pkl" in info_32["outs_paths"][0]


def test_matrix_float_values(simple_pipeline: pathlib.Path) -> None:
    """Float matrix values generate correct variants and preserve type in params."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs, stage_def

class LRParams(stage_def.StageParams):
    learning_rate: float

class TrainOutputs(TypedDict):
    model: Annotated[pathlib.Path, outputs.Out("models/model.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: LRParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"model": pathlib.Path("models/model.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    outs:
      model: "models/lr_${lr}.pkl"
    params:
      learning_rate: "${lr}"
    matrix:
      lr:
        - 0.001
        - 0.01
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    stages = registry.REGISTRY.list_stages()
    assert "train@0.001" in stages
    assert "train@0.01" in stages

    info_001 = registry.REGISTRY.get("train@0.001")
    info_01 = registry.REGISTRY.get("train@0.01")

    # Params should preserve float type
    assert info_001["params"] is not None
    params_001 = info_001["params"].model_dump()
    assert params_001["learning_rate"] == 0.001
    assert isinstance(params_001["learning_rate"], float)

    assert info_01["params"] is not None
    params_01 = info_01["params"].model_dump()
    assert params_01["learning_rate"] == 0.01

    # Paths should use string conversion
    assert "models/lr_0.001.pkl" in info_001["outs_paths"][0]
    assert "models/lr_0.01.pkl" in info_01["outs_paths"][0]


def test_matrix_mixed_interpolation(simple_pipeline: pathlib.Path) -> None:
    """String containing ${var} uses string conversion, exact ${var} preserves type."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs, stage_def

class MixedParams(stage_def.StageParams):
    rate: float
    path: str

class TrainOutputs(TypedDict):
    out: Annotated[pathlib.Path, outputs.Out("models/out.pkl", loaders.PathOnly())]

def preprocess():
    pass

def train(
    params: MixedParams,
    clean: Annotated[pathlib.Path, outputs.Dep("data/clean.csv", loaders.PathOnly())],
) -> TrainOutputs:
    return {"out": pathlib.Path("models/out.pkl")}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    params:
      rate: "${lr}"
      path: "models/${lr}.pkl"
    matrix:
      lr:
        - 0.001
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("train@0.001")

    # Params validation
    assert info["params"] is not None
    params = info["params"].model_dump()

    # Exact match preserves type
    assert params["rate"] == 0.001
    assert isinstance(params["rate"], float)

    # String with interpolation becomes string
    assert params["path"] == "models/0.001.pkl"
    assert isinstance(params["path"], str)


# =============================================================================
# Dict-Form Output with List Paths Tests
# =============================================================================


def test_output_dict_form_with_list_path(simple_pipeline: pathlib.Path) -> None:
    """Dict-form output with list path is parsed correctly."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ProcessOutputs(TypedDict):
    files: Annotated[list[pathlib.Path], outputs.Out(["a.csv", "b.csv"], loaders.PathOnly())]

def preprocess(
    raw: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> ProcessOutputs:
    return {"files": [pathlib.Path("a.csv"), pathlib.Path("b.csv")]}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    outs:
      files: {path: ["out/a.csv", "out/b.csv"], cache: false}
"""
    )

    # Clear cached module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    paths = info["outs_paths"]
    assert len(paths) == 2
    assert any("out/a.csv" in p for p in paths)
    assert any("out/b.csv" in p for p in paths)


def test_output_dict_form_list_path_interpolation(simple_pipeline: pathlib.Path) -> None:
    """Dict-form output with list path supports ${var} interpolation."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pathlib
from typing import Annotated, TypedDict
from pivot import loaders, outputs

class ProcessOutputs(TypedDict):
    files: Annotated[list[pathlib.Path], outputs.Out(["a.csv", "b.csv"], loaders.PathOnly())]

def preprocess(
    raw: Annotated[pathlib.Path, outputs.Dep("data/raw.csv", loaders.PathOnly())],
) -> ProcessOutputs:
    return {"files": [pathlib.Path("a.csv"), pathlib.Path("b.csv")]}
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    outs:
      files: {path: ["out/${model}/a.csv", "out/${model}/b.csv"], cache: false}
    matrix:
      model:
        - bert
"""
    )

    # Clear cached module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess@bert")
    paths = info["outs_paths"]
    assert len(paths) == 2
    assert any("out/bert/a.csv" in p for p in paths)
    assert any("out/bert/b.csv" in p for p in paths)
