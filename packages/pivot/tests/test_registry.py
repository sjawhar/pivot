from __future__ import annotations

import inspect
import pathlib  # noqa: TC003

import pytest

from pivot import exceptions, loaders, project, registry, types
from pivot import stage_def as stage_def_mod


def _helper_stage() -> dict[str, object]:
    return {}


def _helper_stage_info(name: str) -> registry.RegistryStageInfo:
    return registry.RegistryStageInfo(
        func=_helper_stage,
        name=name,
        deps={},
        outs=[
            types.ArtifactRef(
                identity=types.ArtifactIdentity(producer=name, key=None),
                format=loaders.YAML(),
                python_type=dict,
                tag=types.ArtifactTag.DATA,
            )
        ],
        params=stage_def_mod.StageParams(),
        mutex=[],
        variant=None,
        signature=inspect.signature(_helper_stage),
        fingerprint=None,
        params_arg_name=None,
        state_dir=None,
        collection_params={},
    )


def test_stage_registry_add_get_and_list() -> None:
    stage_registry = registry.StageRegistry()
    info = _helper_stage_info("build")

    stage_registry.add_existing(info)

    assert stage_registry.get("build") == info
    assert stage_registry.list_stages() == ["build"]


def test_stage_registry_rejects_duplicate_stage_names() -> None:
    stage_registry = registry.StageRegistry()
    info = _helper_stage_info("build")

    stage_registry.add_existing(info)
    with pytest.raises(exceptions.ValidationError, match="already registered"):
        stage_registry.add_existing(info)


def test_stage_registry_snapshot_restore_and_clear() -> None:
    stage_registry = registry.StageRegistry()
    stage_registry.add_existing(_helper_stage_info("a"))
    snapshot = stage_registry.snapshot()

    stage_registry.clear()
    assert stage_registry.list_stages() == []

    stage_registry.restore(snapshot)
    assert stage_registry.list_stages() == ["a"]


def test_ensure_fingerprint_caches_computation(mocker) -> None:
    stage_registry = registry.StageRegistry()
    stage_registry.add_existing(_helper_stage_info("cached"))

    fp_mock = mocker.patch(
        "pivot.fingerprint.get_stage_fingerprint_cached",
        autospec=True,
        return_value={"code": "abc"},
    )
    mocker.patch("pivot.fingerprint.get_loader_fingerprint", autospec=True, return_value={})

    first = stage_registry.ensure_fingerprint("cached")
    second = stage_registry.ensure_fingerprint("cached")

    assert first == second == {"code": "abc"}
    assert fp_mock.call_count == 1


def test_compute_fingerprint_wraps_errors(mocker) -> None:
    info = _helper_stage_info("broken")
    mocker.patch(
        "pivot.fingerprint.get_stage_fingerprint_cached",
        autospec=True,
        side_effect=RuntimeError("boom"),
    )

    with pytest.raises(exceptions.PivotError, match="fingerprinting failed"):
        registry._compute_fingerprint("broken", info)


def test_get_stage_state_dir_uses_default_when_unset(tmp_path: pathlib.Path) -> None:
    info = _helper_stage_info("x")
    default_dir = tmp_path / ".pivot"

    assert registry.get_stage_state_dir(info, default_dir) == default_dir


def test_build_dag_uses_cache_until_invalidated(mocker) -> None:
    stage_registry = registry.StageRegistry()
    stage_registry.add_existing(_helper_stage_info("train"))

    build_graph_mock = mocker.patch(
        "pivot.engine.graph.build_graph",
        autospec=True,
        return_value="bipartite",
    )
    get_stage_dag_mock = mocker.patch(
        "pivot.engine.graph.get_stage_dag",
        autospec=True,
        return_value="dag",
    )

    first = stage_registry.build_dag()
    second = stage_registry.build_dag()
    stage_registry.invalidate_dag_cache()
    third = stage_registry.build_dag()

    assert first == second == third == "dag"
    assert build_graph_mock.call_count == 2
    assert get_stage_dag_mock.call_count == 2


def test_compute_fingerprint_no_fingerprint_mode_includes_code_deps(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dep_file = tmp_path / "deps.py"
    dep_file.write_text("x = 1\n")
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def _helper_no_fp() -> dict[str, object]:
        return {}

    setattr(_helper_no_fp, "__pivot_no_fingerprint__", True)  # noqa: B010
    setattr(_helper_no_fp, "__pivot_code_deps__", ["deps.py"])  # noqa: B010

    info = _helper_stage_info("nf")
    info["func"] = _helper_no_fp

    fp = registry._compute_fingerprint("nf", info)

    assert any(key.startswith("file:") for key in fp)
    assert any("deps.py" in key for key in fp)


def test_compute_file_fingerprint_raises_for_missing_code_dep(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)

    def _helper_no_fp_missing() -> dict[str, object]:
        return {}

    setattr(_helper_no_fp_missing, "__pivot_code_deps__", ["missing_dep.py"])  # noqa: B010

    with pytest.raises(FileNotFoundError, match="code_deps file not found"):
        registry._compute_file_fingerprint(_helper_no_fp_missing)
