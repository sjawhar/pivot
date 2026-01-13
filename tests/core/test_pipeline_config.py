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
    assert preprocess.deps == ["data/raw.csv"]
    assert preprocess.outs == ["data/clean.csv"]


def test_load_pipeline_file_with_metrics(params_pipeline: pathlib.Path) -> None:
    """Stage config with metrics field is parsed correctly."""
    pipeline_file = params_pipeline / "pivot.yaml"
    config = pipeline_config.load_pipeline_file(pipeline_file)

    train = config.stages["train"]
    assert train.metrics == ["metrics/train.json"]


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
    assert any("data/raw.csv" in dep for dep in info["deps"])


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

    with pytest.raises(pipeline_config.PipelineConfigError, match="no 'params' parameter"):
        pipeline_config.register_from_pipeline_file(pipeline_file)


def test_error_if_params_parameter_has_no_type_hint(
    params_pipeline: pathlib.Path,
) -> None:
    """Error if function has params parameter without type hint."""
    # Create a stage with untyped params
    stages_file = params_pipeline / "stages.py"
    content = stages_file.read_text()
    content = content.replace("def train(params: TrainParams)", "def train(params)")
    stages_file.write_text(content)

    # Need to reload the module
    if "stages" in sys.modules:
        del sys.modules["stages"]

    pipeline_file = params_pipeline / "pivot.yaml"
    with pytest.raises(pipeline_config.PipelineConfigError, match="type hint"):
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
    deps_str = " ".join(info["deps"])

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


def test_matrix_append_override_adds_deps(matrix_pipeline: pathlib.Path) -> None:
    """deps+ override appends to deps list."""
    pipeline_file = matrix_pipeline / "pivot.yaml"
    pipeline_config.register_from_pipeline_file(pipeline_file)

    gpt_info = registry.REGISTRY.get("train@gpt_swe")
    deps_str = " ".join(gpt_info["deps"])

    # gpt variants should have the extra dep
    assert "data/gpt_tokenizer.json" in deps_str

    # bert variants should NOT have it
    bert_info = registry.REGISTRY.get("train@bert_swe")
    bert_deps_str = " ".join(bert_info["deps"])
    assert "data/gpt_tokenizer.json" not in bert_deps_str


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


def test_cwd_with_path_traversal_raises(tmp_path: pathlib.Path) -> None:
    """cwd with path traversal (..) raises validation error."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: os.getcwd
    outs: [out.txt]
    cwd: "../escape"
"""
    )

    with pytest.raises(pipeline_config.PipelineConfigError, match="\\.\\."):
        pipeline_config.load_pipeline_file(pipeline_file)


def test_import_function_invalid_path_raises(tmp_path: pathlib.Path) -> None:
    """Import path without dot raises PipelineConfigError."""
    pipeline_file = tmp_path / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  bad_stage:
    python: no_module_path
    outs: [out.txt]
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
    outs: [out.txt]
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
    outs: [out.txt]
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
    outs: [out.txt]
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
    deps: [data/raw.csv]
    outs:
      - data/clean.csv: {cache: false}
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    out = info["outs"][0]
    assert isinstance(out, outputs.Out)
    assert out.cache is False


def test_plot_with_options(simple_pipeline: pathlib.Path) -> None:
    """Plot with x, y, template options is parsed correctly."""
    from pivot import outputs

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  preprocess:
    python: stages.preprocess
    deps: [data/raw.csv]
    outs: [data/clean.csv]
    plots:
      - plots/curve.json: {x: epoch, y: loss, template: linear}
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("preprocess")
    plot_outs = [o for o in info["outs"] if isinstance(o, outputs.Plot)]
    assert len(plot_outs) == 1
    plot = plot_outs[0]
    assert plot.x == "epoch"
    assert plot.y == "loss"
    assert plot.template == "linear"


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
      - data/clean.csv
    outs:
      - models/out.pkl
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
      - data/clean.csv
    outs:
      - models/out.pkl
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
      - "data/${unknown}.csv"
    outs:
      - models/out.pkl
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
      - data/clean.csv
    outs:
      - "models/${model}_${dataset}.pkl"
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
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
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
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
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
    """variants function returns list of dicts to register."""
    # Create a variants generator function
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass

def get_variants():
    return [
        {"name": "v1", "deps": ["a.txt"], "outs": ["b.txt"]},
        {"name": "v2", "deps": ["c.txt"], "outs": ["d.txt"]},
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


def test_variants_with_cwd(
    simple_pipeline: pathlib.Path,
) -> None:
    """variants with cwd override is applied correctly."""
    (simple_pipeline / "subdir").mkdir()
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass

def get_variants():
    return [
        {"name": "v1", "deps": ["a.txt"], "outs": ["b.txt"], "cwd": "subdir"},
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

    info = registry.REGISTRY.get("train@v1")
    assert "subdir" in str(info["cwd"])


# =============================================================================
# Matrix Override Tests (Additional Coverage)
# =============================================================================


def test_matrix_metrics_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override metrics list."""
    from pivot import outputs

    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
    metrics:
      - metrics/base.json
    matrix:
      model:
        bert:
          metrics:
            - metrics/bert.json
        gpt:
          metrics+:
            - metrics/gpt_extra.json
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    bert_metrics = [o for o in bert_info["outs"] if isinstance(o, outputs.Metric)]
    gpt_metrics = [o for o in gpt_info["outs"] if isinstance(o, outputs.Metric)]

    # bert has override (replaces)
    assert len(bert_metrics) == 1
    assert "bert.json" in bert_metrics[0].path

    # gpt has append
    assert len(gpt_metrics) == 2
    metric_paths = " ".join(m.path for m in gpt_metrics)
    assert "base.json" in metric_paths
    assert "gpt_extra.json" in metric_paths


def test_matrix_plots_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override plots list."""
    from pivot import outputs

    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
    plots:
      - plots/base.json
    matrix:
      model:
        bert:
          plots:
            - plots/bert.json
        gpt:
          plots+:
            - plots/gpt_extra.json
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    bert_plots = [o for o in bert_info["outs"] if isinstance(o, outputs.Plot)]
    gpt_plots = [o for o in gpt_info["outs"] if isinstance(o, outputs.Plot)]

    assert len(bert_plots) == 1
    assert "bert.json" in bert_plots[0].path

    assert len(gpt_plots) == 2


def test_matrix_mutex_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override mutex list."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
    mutex:
      - gpu
    matrix:
      model:
        bert:
          mutex:
            - cpu
        gpt:
          mutex+:
            - memory
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # bert has override (replaces)
    assert bert_info["mutex"] == ["cpu"]

    # gpt has append
    assert "gpu" in gpt_info["mutex"]
    assert "memory" in gpt_info["mutex"]


def test_matrix_cwd_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override cwd."""
    (simple_pipeline / "bert_dir").mkdir()
    (simple_pipeline / "gpt_dir").mkdir()
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
    matrix:
      model:
        bert:
          cwd: bert_dir
        gpt:
          cwd: gpt_dir
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    assert "bert_dir" in str(bert_info["cwd"])
    assert "gpt_dir" in str(gpt_info["cwd"])


def test_matrix_outs_override(simple_pipeline: pathlib.Path) -> None:
    """Matrix dimension can override outs list."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - models/base.pkl
    matrix:
      model:
        bert:
          outs:
            - models/bert_specific.pkl
        gpt:
          outs+:
            - models/gpt_extra.pkl
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    bert_info = registry.REGISTRY.get("train@bert")
    gpt_info = registry.REGISTRY.get("train@gpt")

    # bert has override (replaces)
    assert len(bert_info["outs_paths"]) == 1
    assert "bert_specific.pkl" in bert_info["outs_paths"][0]

    # gpt has append
    assert len(gpt_info["outs_paths"]) == 2


def test_matrix_interpolates_nested_params(simple_pipeline: pathlib.Path) -> None:
    """Matrix interpolates ${dim} in nested params structures."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
import pydantic

class TrainParams(pydantic.BaseModel):
    config: dict

def preprocess():
    pass

def train(params: TrainParams):
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl"
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
    """Matrix interpolates ${dim} in output dict keys."""
    stages_file = simple_pipeline / "stages.py"
    stages_file.write_text(
        """\
def preprocess():
    pass

def train():
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${model}.pkl":
          cache: false
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
import pydantic

class FlagParams(pydantic.BaseModel):
    enabled: bool

def preprocess():
    pass

def train(params: FlagParams):
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/${flag}.pkl"
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
import pydantic

class SizeParams(pydantic.BaseModel):
    batch_size: int

def preprocess():
    pass

def train(params: SizeParams):
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/batch_${size}.pkl"
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
import pydantic

class LRParams(pydantic.BaseModel):
    learning_rate: float

def preprocess():
    pass

def train(params: LRParams):
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - "models/lr_${lr}.pkl"
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
import pydantic

class MixedParams(pydantic.BaseModel):
    rate: float
    path: str

def preprocess():
    pass

def train(params: MixedParams):
    pass
"""
    )

    pipeline_file = simple_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
  train:
    python: stages.train
    deps:
      - data/clean.csv
    outs:
      - models/out.pkl
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
# StageDef Integration Tests
# =============================================================================


@pytest.fixture
def stage_def_pipeline(tmp_path: pathlib.Path, mocker: MockerFixture) -> Generator[pathlib.Path]:
    """Set up a pipeline with StageDef-based stages."""
    # Create data directory structure
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "input.csv").write_text("a,b\n1,2\n3,4\n")
    (tmp_path / "output").mkdir(exist_ok=True)

    # Create stages.py with StageDef
    stages_file = tmp_path / "stages.py"
    stages_file.write_text(
        """\
import pandas
from pivot import loaders, stage_def


class ProcessParams(stage_def.StageDef):
    data: stage_def.Dep[pandas.DataFrame] = stage_def.Dep("data/input.csv", loaders.CSV())
    result: stage_def.Out[dict[str, int]] = stage_def.Out("output/result.json", loaders.JSON())

    threshold: float = 0.5


def process(params: ProcessParams) -> None:
    df = params.data
    result = {"count": len(df[df["a"] > params.threshold])}
    params.result = result
"""
    )

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


def test_stage_def_extracts_deps_from_class(stage_def_pipeline: pathlib.Path) -> None:
    """StageDef deps are auto-extracted when not specified in YAML."""
    pipeline_file = stage_def_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
 process:
   python: stages.process
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("process")
    # Deps should be extracted from StageDef
    assert any("data/input.csv" in dep for dep in info["deps"])


def test_stage_def_extracts_outs_from_class(stage_def_pipeline: pathlib.Path) -> None:
    """StageDef outs are auto-extracted when not specified in YAML."""
    pipeline_file = stage_def_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
 process:
   python: stages.process
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("process")
    # Outs should be extracted from StageDef
    assert any("output/result.json" in out for out in info["outs_paths"])


def test_stage_def_yaml_deps_override(stage_def_pipeline: pathlib.Path) -> None:
    """Explicit YAML deps completely replace StageDef deps."""
    pipeline_file = stage_def_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
 process:
   python: stages.process
   deps:
     - data/override.csv
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("process")
    deps_str = " ".join(info["deps"])
    # Should use YAML deps, not StageDef deps
    assert "data/override.csv" in deps_str
    # Should NOT have StageDef deps (complete replacement)
    assert "data/input.csv" not in deps_str


def test_stage_def_yaml_outs_override(stage_def_pipeline: pathlib.Path) -> None:
    """Explicit YAML outs completely replace StageDef outs."""
    pipeline_file = stage_def_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
 process:
   python: stages.process
   outs:
     - output/override.json
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("process")
    outs_str = " ".join(info["outs_paths"])
    # Should use YAML outs, not StageDef outs
    assert "output/override.json" in outs_str
    # Should NOT have StageDef outs (complete replacement)
    assert "output/result.json" not in outs_str


def test_stage_def_params_available(stage_def_pipeline: pathlib.Path) -> None:
    """StageDef params are available in registered stage."""
    pipeline_file = stage_def_pipeline / "pivot.yaml"
    pipeline_file.write_text(
        """\
stages:
 process:
   python: stages.process
   params:
     threshold: 0.8
"""
    )

    pipeline_config.register_from_pipeline_file(pipeline_file)

    info = registry.REGISTRY.get("process")
    params = info["params"]
    assert params is not None
    assert params.model_dump()["threshold"] == 0.8
