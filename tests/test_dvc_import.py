from __future__ import annotations

import pathlib

import pytest

from pivot import dvc_import, exceptions

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "dvc_import"


# =============================================================================
# parse_dvc_yaml tests
# =============================================================================


def test_parse_simple_dvc_yaml() -> None:
    """Parse basic dvc.yaml with stages."""
    result = dvc_import.parse_dvc_yaml(FIXTURES_DIR / "simple" / "dvc.yaml")
    assert "stages" in result
    assert "preprocess" in result["stages"]
    assert "train" in result["stages"]


def test_parse_dvc_yaml_extracts_cmd() -> None:
    """Parses cmd field from stages."""
    result = dvc_import.parse_dvc_yaml(FIXTURES_DIR / "simple" / "dvc.yaml")
    assert result["stages"]["preprocess"].get("cmd") == "python src/preprocess.py data/raw.csv"


def test_parse_dvc_yaml_extracts_deps() -> None:
    """Parses deps list from stages."""
    result = dvc_import.parse_dvc_yaml(FIXTURES_DIR / "simple" / "dvc.yaml")
    deps = result["stages"]["preprocess"].get("deps", [])
    assert "data/raw.csv" in deps
    assert "src/preprocess.py" in deps


def test_parse_dvc_yaml_extracts_outs() -> None:
    """Parses outs list from stages."""
    result = dvc_import.parse_dvc_yaml(FIXTURES_DIR / "simple" / "dvc.yaml")
    outs = result["stages"]["preprocess"].get("outs", [])
    assert "data/clean.csv" in outs


def test_parse_dvc_yaml_missing_file(tmp_path: pathlib.Path) -> None:
    """Raises error for missing file."""
    with pytest.raises(exceptions.DVCImportError, match="Cannot read"):
        dvc_import.parse_dvc_yaml(tmp_path / "nonexistent.yaml")


def test_parse_dvc_yaml_empty_file(tmp_path: pathlib.Path) -> None:
    """Raises error for empty file."""
    empty_file = tmp_path / "dvc.yaml"
    empty_file.write_text("")
    with pytest.raises(exceptions.DVCImportError, match="empty"):
        dvc_import.parse_dvc_yaml(empty_file)


def test_parse_dvc_yaml_missing_stages(tmp_path: pathlib.Path) -> None:
    """Raises error when stages key is missing."""
    bad_file = tmp_path / "dvc.yaml"
    bad_file.write_text("vars:\n  - foo.yaml\n")
    with pytest.raises(exceptions.DVCImportError, match="missing 'stages'"):
        dvc_import.parse_dvc_yaml(bad_file)


def test_parse_dvc_yaml_size_limit(tmp_path: pathlib.Path) -> None:
    """Raises error for oversized file."""
    large_file = tmp_path / "dvc.yaml"
    # Create file larger than limit
    large_file.write_text("x" * (dvc_import.MAX_DVC_FILE_SIZE + 1))
    with pytest.raises(exceptions.DVCImportError, match="exceeds maximum size"):
        dvc_import.parse_dvc_yaml(large_file)


# =============================================================================
# parse_dvc_lock tests
# =============================================================================


def test_parse_dvc_lock_returns_none_for_missing() -> None:
    """Returns None when dvc.lock doesn't exist."""
    result = dvc_import.parse_dvc_lock(pathlib.Path("/nonexistent/dvc.lock"))
    assert result is None


def test_parse_dvc_lock_extracts_stages() -> None:
    """Extracts stages from dvc.lock."""
    result = dvc_import.parse_dvc_lock(FIXTURES_DIR / "simple" / "dvc.lock")
    assert result is not None
    assert "stages" in result
    assert "preprocess" in result["stages"]
    assert "train" in result["stages"]


# =============================================================================
# parse_params_yaml tests
# =============================================================================


def test_parse_params_yaml() -> None:
    """Parses params.yaml correctly."""
    result, notes = dvc_import.parse_params_yaml(FIXTURES_DIR / "with_params" / "params.yaml")
    assert result["train"]["learning_rate"] == 0.01
    assert result["train"]["epochs"] == 100
    assert result["model"]["hidden_size"] == 256
    assert len(notes) == 0


def test_parse_params_yaml_missing_returns_empty() -> None:
    """Returns empty dict when file doesn't exist."""
    result, notes = dvc_import.parse_params_yaml(pathlib.Path("/nonexistent/params.yaml"))
    assert result == {}
    assert len(notes) == 0


def test_parse_params_yaml_warns_on_non_string_keys(tmp_path: pathlib.Path) -> None:
    """Warns when params.yaml has non-string keys."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("123: numeric_key\ntrue: bool_key\nvalid: string_key\n")

    result, notes = dvc_import.parse_params_yaml(params_file)

    assert "valid" in result
    assert 123 not in result
    assert len(notes) == 1
    assert notes[0]["severity"] == "warning"
    assert "non-string keys" in notes[0]["message"]


# =============================================================================
# convert_pipeline tests
# =============================================================================


def test_convert_simple_stage(tmp_path: pathlib.Path) -> None:
    """Converts basic DVC stage to Pivot format."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "simple" / "dvc.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    assert "preprocess" in stages
    assert stages["preprocess"].get("python") == "PLACEHOLDER.preprocess"
    assert "data/raw.csv" in stages["preprocess"].get("deps", [])
    assert "data/clean.csv" in stages["preprocess"].get("outs", [])


def test_convert_stage_generates_shell_command_note(tmp_path: pathlib.Path) -> None:
    """Shell commands generate migration notes."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "simple" / "dvc.yaml",
        project_root=tmp_path,
    )
    notes = result["notes"]

    # Should have notes for shell commands
    shell_notes = [n for n in notes if n["original_cmd"]]
    assert len(shell_notes) >= 2  # preprocess and train both have cmd


def test_convert_stage_with_params_inlines_values(tmp_path: pathlib.Path) -> None:
    """Params from params.yaml are inlined."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "with_params" / "dvc.yaml",
        params_yaml_path=FIXTURES_DIR / "with_params" / "params.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    # train stage should have inlined params
    train_params = stages["train"].get("params", {})
    assert train_params.get("learning_rate") == 0.01
    assert train_params.get("epochs") == 100


def test_convert_foreach_to_matrix(tmp_path: pathlib.Path) -> None:
    """DVC foreach converts to Pivot matrix."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "foreach" / "dvc.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    assert "process" in stages
    process_stage = stages["process"]
    assert "matrix" in process_stage
    assert process_stage["matrix"]["item"] == ["train", "test", "validation"]


def test_convert_outs_cache_false(tmp_path: pathlib.Path) -> None:
    """Outputs with cache: false are preserved."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "complex" / "dvc.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    preprocess_outs = stages["preprocess"].get("outs", [])
    # Should have dict format with cache: false
    cache_false_out = [o for o in preprocess_outs if isinstance(o, dict)]
    assert len(cache_false_out) == 1


def test_convert_wdir_generates_info_note(tmp_path: pathlib.Path) -> None:
    """DVC wdir generates an info note (paths are auto-prefixed for resolution)."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "complex" / "dvc.yaml",
        project_root=tmp_path,
    )
    notes = result["notes"]

    wdir_notes = [n for n in notes if "wdir" in n["message"].lower()]
    assert len(wdir_notes) == 1
    assert wdir_notes[0]["stage"] == "preprocess"
    assert wdir_notes[0]["severity"] == "info"


def test_frozen_stage_generates_warning(tmp_path: pathlib.Path) -> None:
    """Frozen stages generate warning notes."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "complex" / "dvc.yaml",
        project_root=tmp_path,
    )
    notes = result["notes"]

    frozen_notes = [n for n in notes if "frozen" in n["message"].lower()]
    assert len(frozen_notes) == 1
    assert frozen_notes[0]["stage"] == "frozen_stage"


def test_convert_metrics_preserved(tmp_path: pathlib.Path) -> None:
    """Metrics are preserved in conversion."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "complex" / "dvc.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    assert "metrics/train_metrics.json" in stages["train"].get("metrics", [])


def test_convert_plots_preserved(tmp_path: pathlib.Path) -> None:
    """Plots are preserved in conversion."""
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=FIXTURES_DIR / "complex" / "dvc.yaml",
        project_root=tmp_path,
    )
    stages = result["stages"]

    assert "plots/loss_curve.csv" in stages["train"].get("plots", [])


# =============================================================================
# path validation tests
# =============================================================================


def test_path_traversal_rejected(tmp_path: pathlib.Path) -> None:
    """Paths with traversal outside project are rejected."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  evil:
    cmd: echo
    deps:
      - ../../../etc/passwd
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )
    notes = result["notes"]

    error_notes = [n for n in notes if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "etc/passwd" in error_notes[0]["message"]


def test_interpolated_paths_allowed(tmp_path: pathlib.Path) -> None:
    """Paths with ${} interpolation are allowed without validation."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  templated:
    cmd: echo
    deps:
      - data/${item}.csv
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    # Should not have error notes for the interpolated path
    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 0


def test_non_string_dict_key_in_deps_generates_error(tmp_path: pathlib.Path) -> None:
    """Non-string keys in deps dict generate error notes."""
    dvc_yaml = tmp_path / "dvc.yaml"
    # YAML allows numeric keys - this creates {123: {cache: false}}
    dvc_yaml.write_text("""
stages:
  test:
    cmd: echo
    outs:
      - 123:
          cache: false
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "expected string path" in error_notes[0]["message"]
    assert "int" in error_notes[0]["message"]


def test_interpolation_path_with_traversal_rejected(tmp_path: pathlib.Path) -> None:
    """Interpolation paths with traversal in literal parts are rejected."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  evil:
    cmd: echo
    deps:
      - ${item}/../../../etc/passwd
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "Path traversal not allowed" in error_notes[0]["message"]


# =============================================================================
# param format tests
# =============================================================================


def test_param_formats_dotted_key(tmp_path: pathlib.Path) -> None:
    """Dotted key format resolves params."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.001\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - train.lr
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    train_params = result["stages"]["train"].get("params", {})
    assert train_params.get("lr") == 0.001


def test_param_formats_explicit_file(tmp_path: pathlib.Path) -> None:
    """File:key format resolves params."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("model:\n  hidden: 256\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - params.yaml:model.hidden
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    train_params = result["stages"]["train"].get("params", {})
    assert train_params.get("hidden") == 256


def test_param_formats_nested(tmp_path: pathlib.Path) -> None:
    """Nested key format resolves params."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("model:\n  config:\n    layers: 12\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - model.config.layers
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    train_params = result["stages"]["train"].get("params", {})
    assert train_params.get("layers") == 12


# =============================================================================
# param collision and multi-file tests
# =============================================================================


def test_param_key_collision_warns(tmp_path: pathlib.Path) -> None:
    """Warns when multiple params map to same leaf key."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\nmodel:\n  lr: 0.001\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - train.lr
      - model.lr
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )

    collision_notes = [n for n in result["notes"] if "collision" in n["message"].lower()]
    assert len(collision_notes) == 1
    assert "lr" in collision_notes[0]["message"]


def test_non_default_param_file_warns(tmp_path: pathlib.Path) -> None:
    """Warns when params reference non-default file."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - custom.yaml:train.lr
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )

    file_notes = [n for n in result["notes"] if "custom.yaml" in n["message"]]
    assert len(file_notes) == 1
    assert "not loaded" in file_notes[0]["message"]


def test_params_inlined_stat_updated(tmp_path: pathlib.Path) -> None:
    """Stats correctly count inlined params."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n  epochs: 100\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - train.lr
      - train.epochs
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )

    assert result["stats"]["params_inlined"] == 2


# =============================================================================
# wdir validation tests
# =============================================================================


def test_absolute_wdir_generates_info(tmp_path: pathlib.Path) -> None:
    """Absolute wdir paths generate info note (paths auto-prefixed)."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  test:
    cmd: echo
    wdir: /tmp/work
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    # wdir generates info note since paths are auto-prefixed
    wdir_notes = [n for n in result["notes"] if "wdir" in n["message"].lower()]
    assert len(wdir_notes) == 1
    assert wdir_notes[0]["severity"] == "info"


# =============================================================================
# foreach edge case tests
# =============================================================================


def test_empty_foreach_list_error(tmp_path: pathlib.Path) -> None:
    """Empty foreach list generates error."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  process:
    foreach: []
    do:
      cmd: echo
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "empty" in error_notes[0]["message"].lower()


def test_empty_foreach_dict_error(tmp_path: pathlib.Path) -> None:
    """Empty foreach dict generates error."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  process:
    foreach: {}
    do:
      cmd: echo
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "empty" in error_notes[0]["message"].lower()


def test_dict_foreach_warns_about_lost_values(tmp_path: pathlib.Path) -> None:
    """Dict foreach warns that values are not preserved."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  process:
    foreach:
      lr_001: {val: 0.001}
      lr_01: {val: 0.01}
    do:
      cmd: echo ${item}
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    value_notes = [n for n in result["notes"] if "values are not preserved" in n["message"]]
    assert len(value_notes) == 1


# =============================================================================
# write_pivot_yaml tests
# =============================================================================


def test_write_pivot_yaml_creates_file(tmp_path: pathlib.Path) -> None:
    """write_pivot_yaml creates output file."""
    output_path = tmp_path / "pivot.yaml"
    stages = {"test": dvc_import.PivotStageConfig(python="test.main", deps=["data.csv"])}

    dvc_import.write_pivot_yaml(stages, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "test:" in content
    assert "python: test.main" in content


def test_write_pivot_yaml_refuses_overwrite_without_force(tmp_path: pathlib.Path) -> None:
    """write_pivot_yaml errors on existing file without force."""
    output_path = tmp_path / "pivot.yaml"
    output_path.write_text("existing content")

    with pytest.raises(exceptions.DVCImportError, match="already exists"):
        dvc_import.write_pivot_yaml({}, output_path)


def test_write_pivot_yaml_overwrites_with_force(tmp_path: pathlib.Path) -> None:
    """write_pivot_yaml overwrites with force=True."""
    output_path = tmp_path / "pivot.yaml"
    output_path.write_text("old content")

    dvc_import.write_pivot_yaml(
        {"new": dvc_import.PivotStageConfig(python="new.main")}, output_path, force=True
    )

    content = output_path.read_text()
    assert "new:" in content


# =============================================================================
# write_migration_notes tests
# =============================================================================


def test_write_migration_notes_creates_file(tmp_path: pathlib.Path) -> None:
    """write_migration_notes creates output file."""
    notes_path = tmp_path / ".pivot" / "migration-notes.md"
    notes: list[dvc_import.MigrationNote] = [
        dvc_import.MigrationNote(
            stage="train",
            severity="warning",
            message="Shell command needs conversion",
            original_cmd="python train.py",
        )
    ]
    stats = dvc_import.ConversionStats(
        stages_converted=1,
        stages_with_shell_commands=1,
        stages_with_warnings=1,
        params_inlined=0,
    )

    dvc_import.write_migration_notes(notes, stats, notes_path)

    assert notes_path.exists()
    content = notes_path.read_text()
    assert "DVC to Pivot Migration Notes" in content
    assert "train" in content


def test_write_migration_notes_includes_security_warning(tmp_path: pathlib.Path) -> None:
    """Migration notes include security warning."""
    notes_path = tmp_path / "notes.md"
    notes: list[dvc_import.MigrationNote] = []
    stats = dvc_import.ConversionStats(
        stages_converted=0,
        stages_with_shell_commands=0,
        stages_with_warnings=0,
        params_inlined=0,
    )

    dvc_import.write_migration_notes(notes, stats, notes_path)

    content = notes_path.read_text()
    assert "Security Review Required" in content


def test_write_migration_notes_escapes_commands(tmp_path: pathlib.Path) -> None:
    """Commands with backticks are escaped in notes."""
    notes_path = tmp_path / "notes.md"
    notes: list[dvc_import.MigrationNote] = [
        dvc_import.MigrationNote(
            stage="test",
            severity="warning",
            message="test",
            original_cmd="echo `whoami`",
        )
    ]
    stats = dvc_import.ConversionStats(
        stages_converted=1,
        stages_with_shell_commands=1,
        stages_with_warnings=0,
        params_inlined=0,
    )

    dvc_import.write_migration_notes(notes, stats, notes_path)

    content = notes_path.read_text()
    assert "\\`whoami\\`" in content


# =============================================================================
# parse_dvc_yaml error handling tests (additional coverage)
# =============================================================================


def test_parse_dvc_yaml_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """Raises error for invalid YAML syntax."""
    bad_file = tmp_path / "dvc.yaml"
    bad_file.write_text("stages:\n  - invalid: [unbalanced")
    with pytest.raises(exceptions.DVCImportError, match="Invalid YAML"):
        dvc_import.parse_dvc_yaml(bad_file)


def test_parse_dvc_yaml_non_dict_root(tmp_path: pathlib.Path) -> None:
    """Raises error when root is not a dict."""
    bad_file = tmp_path / "dvc.yaml"
    bad_file.write_text("- item1\n- item2\n")
    with pytest.raises(exceptions.DVCImportError, match="must be a mapping"):
        dvc_import.parse_dvc_yaml(bad_file)


def test_parse_dvc_yaml_non_dict_stages(tmp_path: pathlib.Path) -> None:
    """Raises error when stages is not a dict."""
    bad_file = tmp_path / "dvc.yaml"
    bad_file.write_text("stages:\n  - stage1\n  - stage2\n")
    with pytest.raises(exceptions.DVCImportError, match="'stages' must be a mapping"):
        dvc_import.parse_dvc_yaml(bad_file)


# =============================================================================
# parse_dvc_lock error handling tests
# =============================================================================


def test_parse_dvc_lock_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """Returns None for invalid YAML in dvc.lock."""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text("stages:\n  - invalid: [unbalanced")
    result = dvc_import.parse_dvc_lock(lock_file)
    assert result is None


def test_parse_dvc_lock_non_dict_returns_none(tmp_path: pathlib.Path) -> None:
    """Returns None when dvc.lock is not a dict."""
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text("- item1\n- item2\n")
    result = dvc_import.parse_dvc_lock(lock_file)
    assert result is None


# =============================================================================
# foreach error handling tests
# =============================================================================


def test_foreach_missing_do_block(tmp_path: pathlib.Path) -> None:
    """foreach without 'do' block generates error."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  process:
    foreach:
      - a
      - b
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "missing 'do' block" in error_notes[0]["message"]


def test_foreach_unsupported_type(tmp_path: pathlib.Path) -> None:
    """foreach with unsupported type generates error."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  process:
    foreach: "not_a_list_or_dict"
    do:
      cmd: echo ${item}
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )

    error_notes = [n for n in result["notes"] if n["severity"] == "error"]
    assert len(error_notes) == 1
    assert "Unsupported foreach type" in error_notes[0]["message"]


# =============================================================================
# dict-form params handling tests
# =============================================================================


def test_dict_params_with_non_list_keys_skipped(tmp_path: pathlib.Path) -> None:
    """Dict-form params with non-list keys are skipped."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - params.yaml:
          nested: not_a_list
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    # Should not crash, just skip the invalid format
    assert "train" in result["stages"]


def test_dict_params_with_non_string_key_skipped(tmp_path: pathlib.Path) -> None:
    """Dict-form params with non-string keys are skipped."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("123: numeric_value\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - params.yaml:
          - 123
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    # Should not crash, just skip non-string keys
    train_params = result["stages"]["train"].get("params", {})
    assert 123 not in train_params


# =============================================================================
# migration notes section helper tests
# =============================================================================


def test_write_migration_notes_includes_error_section(tmp_path: pathlib.Path) -> None:
    """Migration notes include error section when errors present."""
    notes_path = tmp_path / "notes.md"
    notes: list[dvc_import.MigrationNote] = [
        dvc_import.MigrationNote(
            stage="broken",
            severity="error",
            message="Critical error occurred",
            original_cmd=None,
        )
    ]
    stats = dvc_import.ConversionStats(
        stages_converted=0,
        stages_with_shell_commands=0,
        stages_with_warnings=0,
        params_inlined=0,
    )

    dvc_import.write_migration_notes(notes, stats, notes_path)

    content = notes_path.read_text()
    assert "## Errors" in content
    assert "Critical error occurred" in content
    assert "Stage 'broken'" in content


def test_write_migration_notes_empty_section_omitted(tmp_path: pathlib.Path) -> None:
    """Empty note sections are not written."""
    notes_path = tmp_path / "notes.md"
    notes: list[dvc_import.MigrationNote] = []
    stats = dvc_import.ConversionStats(
        stages_converted=1,
        stages_with_shell_commands=0,
        stages_with_warnings=0,
        params_inlined=0,
    )

    dvc_import.write_migration_notes(notes, stats, notes_path)

    content = notes_path.read_text()
    assert "## Errors" not in content
    assert "## Warnings" not in content


def test_write_migration_notes_refuses_overwrite_without_force(tmp_path: pathlib.Path) -> None:
    """write_migration_notes refuses to overwrite without force."""
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("existing content")

    stats = dvc_import.ConversionStats(
        stages_converted=0,
        stages_with_shell_commands=0,
        stages_with_warnings=0,
        params_inlined=0,
    )
    with pytest.raises(exceptions.DVCImportError, match="already exists"):
        dvc_import.write_migration_notes([], stats, notes_path)


# =============================================================================
# params.yaml error handling tests
# =============================================================================


def test_parse_params_yaml_invalid_yaml(tmp_path: pathlib.Path) -> None:
    """Returns empty dict and no error for invalid YAML."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("invalid: [unbalanced")
    with pytest.raises(exceptions.DVCImportError, match="Invalid YAML"):
        dvc_import.parse_params_yaml(params_file)


def test_parse_params_yaml_empty_returns_empty(tmp_path: pathlib.Path) -> None:
    """Returns empty dict for empty params.yaml."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("")
    result, _ = dvc_import.parse_params_yaml(params_file)
    assert result == {}


def test_parse_params_yaml_non_dict_raises(tmp_path: pathlib.Path) -> None:
    """Raises error when params.yaml is not a dict."""
    params_file = tmp_path / "params.yaml"
    params_file.write_text("- item1\n- item2\n")
    with pytest.raises(exceptions.DVCImportError, match="must be a mapping"):
        dvc_import.parse_params_yaml(params_file)


# =============================================================================
# dict-form deps/outs tests
# =============================================================================


def test_dict_form_deps_extracted(tmp_path: pathlib.Path) -> None:
    """Dict-form deps are extracted correctly."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    deps:
      - data/input.csv: {}
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )
    assert "data/input.csv" in result["stages"]["train"].get("deps", [])


def test_dict_form_outs_without_cache_false(tmp_path: pathlib.Path) -> None:
    """Dict-form outs without cache:false are converted to simple paths."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    outs:
      - output.csv: {persist: true}
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        project_root=tmp_path,
    )
    outs = result["stages"]["train"].get("outs", [])
    # Should be a simple string path, not a dict
    assert "output.csv" in outs


# =============================================================================
# dvc.lock params extraction tests
# =============================================================================


def test_params_from_dvc_lock_used(tmp_path: pathlib.Path) -> None:
    """Parameters from dvc.lock are used when available (nested structure)."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - train.lr
""")

    # DVC lock file stores params in nested structure matching params.yaml
    dvc_lock = tmp_path / "dvc.lock"
    dvc_lock.write_text("""
stages:
  train:
    params:
      params.yaml:
        train:
          lr: 0.001
""")

    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        dvc_lock_path=dvc_lock,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    # Should use lock value (0.001) over params.yaml value (0.01)
    assert result["stages"]["train"].get("params", {}).get("lr") == 0.001


# =============================================================================
# params string form with colon tests
# =============================================================================


def test_params_whole_file_dep_skipped(tmp_path: pathlib.Path) -> None:
    """Whole file params deps (file:) are skipped."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - params.yaml:
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )
    # Should not have any params (whole file dep is skipped)
    train_params = result["stages"]["train"].get("params", {})
    assert len(train_params) == 0


# =============================================================================
# dict-form params with non-default file tests
# =============================================================================


def test_dict_form_params_non_default_file_warns(tmp_path: pathlib.Path) -> None:
    """Dict-form params referencing non-default file warns."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - custom.yaml:
          - train.lr
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )

    file_notes = [n for n in result["notes"] if "custom.yaml" in n["message"]]
    assert len(file_notes) == 1
    assert "not loaded" in file_notes[0]["message"]


def test_dict_form_params_with_params_yaml_inlined(tmp_path: pathlib.Path) -> None:
    """Dict-form params from params.yaml are inlined."""
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text("train:\n  lr: 0.01\n")

    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text("""
stages:
  train:
    cmd: echo
    params:
      - params.yaml:
          - train.lr
""")
    result = dvc_import.convert_pipeline(
        dvc_yaml_path=dvc_yaml,
        params_yaml_path=params_yaml,
        project_root=tmp_path,
    )

    train_params = result["stages"]["train"].get("params", {})
    assert train_params.get("lr") == 0.01
