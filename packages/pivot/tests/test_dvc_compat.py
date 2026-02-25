from __future__ import annotations

import inspect
import pathlib  # noqa: TC003

import pytest

from pivot import dvc_compat, exceptions, loaders, stage_def, types


def _helper_export_stage() -> dict[str, object]:
    return {}


def _helper_stage_info(name: str) -> dict[str, object]:
    return {
        "func": _helper_export_stage,
        "name": name,
        "deps": {
            "raw": types.ArtifactRef(
                identity=types.ArtifactIdentity(producer="data/raw/input.csv", key=None),
                format=loaders.CSV(),
                python_type=dict,
                tag=types.ArtifactTag.DATA,
            )
        },
        "outs": [
            types.ArtifactRef(
                identity=types.ArtifactIdentity(producer=name, key="metrics"),
                format=loaders.YAML(),
                python_type=dict,
                tag=types.ArtifactTag.METRIC,
            )
        ],
        "params": stage_def.StageParams(),
        "mutex": [],
        "variant": None,
        "signature": inspect.signature(_helper_export_stage),
        "fingerprint": None,
        "params_arg_name": None,
        "state_dir": None,
        "collection_params": {},
    }


def test_to_relative_path_keeps_non_absolute_path(tmp_path: pathlib.Path) -> None:
    assert dvc_compat._to_relative_path("data/file.csv", tmp_path) == "data/file.csv"


def test_generate_cmd_rejects_main_and_lambda() -> None:
    def _helper_main() -> None:
        return None

    _helper_main.__module__ = "__main__"

    with pytest.raises(exceptions.ExportError, match="__main__"):
        dvc_compat._generate_cmd(_helper_main)

    with pytest.raises(exceptions.ExportError, match="lambda"):
        dvc_compat._generate_cmd(lambda: None)


def test_extract_param_defaults_only_includes_defaults() -> None:
    def _helper(a: int, b: str = "x", c: int = 2) -> None:
        return None

    defaults = dvc_compat._extract_param_defaults(inspect.signature(_helper))

    assert defaults == {"b": "x", "c": 2}


def test_build_out_entry_marks_metrics_as_nocache() -> None:
    metric_out = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="train", key="metrics"),
        format=loaders.YAML(),
        python_type=dict,
        tag=types.ArtifactTag.METRIC,
    )
    data_out = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="train", key="model"),
        format=loaders.Pickle(),
        python_type=dict,
        tag=types.ArtifactTag.DATA,
    )

    metric_entry = dvc_compat._build_out_entry(metric_out, "metrics.yaml")
    data_entry = dvc_compat._build_out_entry(data_out, "model.pkl")

    assert metric_entry == {"metrics.yaml": {"cache": False}}
    assert data_entry == "model.pkl"


def test_export_dvc_yaml_writes_expected_structure(
    tmp_path: pathlib.Path,
    mocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dvc.yaml"
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    mocker.patch("pivot.cli.helpers.list_stages", autospec=True, return_value=["train"])
    mocker.patch(
        "pivot.cli.helpers.get_stage",
        autospec=True,
        side_effect=lambda name: _helper_stage_info(name),
    )

    result = dvc_compat.export_dvc_yaml(output)

    assert "stages" in result
    assert "train" in result["stages"]
    assert output.exists()


def test_export_dvc_yaml_rejects_missing_requested_stage(
    tmp_path: pathlib.Path,
    mocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dvc.yaml"
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    mocker.patch("pivot.cli.helpers.list_stages", autospec=True, return_value=["train"])

    with pytest.raises(exceptions.ExportError, match="Stages not found"):
        dvc_compat.export_dvc_yaml(output, stages=["missing"])
