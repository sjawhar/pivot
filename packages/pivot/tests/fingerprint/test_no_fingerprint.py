"""Integration tests for @no_fingerprint skip detection behavior."""

from __future__ import annotations

import importlib.util
import math
import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import compose, decorators, executor, fingerprint, loaders, registry
from pivot.storage import cache, state
from pivot.storage import store as store_mod
from pivot.types import ArtifactIdentity, ArtifactRef, ArtifactTag, identity_key

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage, StageResult


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@decorators.no_fingerprint()
def _helper_no_fingerprint_stage() -> str:
    return "output"


@decorators.no_fingerprint(code_deps=["helper.py"])
def _helper_no_fingerprint_with_code_dep() -> str:
    return "output"


@decorators.no_fingerprint(code_deps=["nonexistent.py"])
def _helper_no_fingerprint_missing_dep() -> str:
    return "output"


def _helper_ast_fingerprint_stage() -> str:
    return str(math.pi)


@compose.stage
@decorators.no_fingerprint(code_deps=["helper.py"])
def _helper_code_deps_inside_stage() -> str:
    return "output"


@decorators.no_fingerprint(code_deps=["helper.py"])
@compose.stage
def _helper_code_deps_outside_stage() -> str:
    return "output"


def _output_ref(stage_name: str) -> ArtifactRef:
    return ArtifactRef(
        identity=ArtifactIdentity(stage_name, None),
        format=loaders.Text(),
        python_type=str,
        tag=ArtifactTag.DATA,
    )


def _make_stage_info(
    func: Callable[..., object],
    tmp_path: pathlib.Path,
    *,
    code_fingerprint: dict[str, str],
    outs: list[ArtifactRef],
) -> WorkerStageInfo:
    return {
        "func": func,
        "fingerprint": code_fingerprint,
        "deps": {},
        "signature": None,
        "outs": outs,
        "store_spec": {
            "kind": "workspace",
            "cache_dir": str(tmp_path / ".pivot" / "cache"),
            "project_root": str(tmp_path),
            "pipeline_name": "test",
            "input_bindings": {},
        },
        "params": None,
        "variant": None,
        "overrides": {},
        "checkout_modes": [
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
        "run_id": "test_run",
        "force": False,
        "no_commit": False,
        "params_arg_name": None,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
        "collection_params": {},
    }


def _apply_deferred_writes(
    stage_name: str,
    stage_info: WorkerStageInfo,
    result: StageResult,
) -> None:
    if "deferred_writes" not in result:
        return
    deferred = result["deferred_writes"]
    out_keys = [identity_key(out.identity) for out in stage_info["outs"]]

    with state.StateDB(stage_info["state_dir"]) as state_db:
        state_db.apply_deferred_writes(stage_name, out_keys, deferred)


def _read_output(stage_info: WorkerStageInfo, ref: ArtifactRef) -> str:
    store = store_mod.store_from_spec(stage_info["store_spec"])
    return store.checkout(ref).read_text()


def test_no_fingerprint_stage_skips_when_unchanged(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", _REPO_ROOT)
    fingerprint_map = registry._compute_file_fingerprint(_helper_no_fingerprint_stage, [])
    out = _output_ref("no_fp")
    stage_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        code_fingerprint=fingerprint_map,
        outs=[out],
    )

    result1 = executor.execute_stage("no_fp", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("no_fp", stage_info, result1)

    result2 = executor.execute_stage("no_fp", stage_info, worker_env, output_queue)
    assert result2["status"] == "cached", f"Expected skip, got: {result2}"


def test_no_fingerprint_reruns_on_source_file_change(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)

    stage_file = tmp_path / "stages.py"
    stage_file.write_text("def stage_func():\n    return 'v1'\n")

    spec = importlib.util.spec_from_file_location("stages", stage_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decorators.no_fingerprint()(mod.stage_func)

    fp1 = registry._compute_file_fingerprint(mod.stage_func, [])
    assert any(k.startswith("file:") for k in fp1), "Fingerprint should include file: keys"

    out = _output_ref("src_fp")
    stage_info = _make_stage_info(mod.stage_func, tmp_path, code_fingerprint=fp1, outs=[out])

    result1 = executor.execute_stage("src_fp", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("src_fp", stage_info, result1)
    assert _read_output(stage_info, out) == "v1", "First run should write v1"

    # Different length from v1: a same-size rewrite within one second reuses the stale .pyc.
    stage_file.write_text("def stage_func():\n    return 'v2-changed'\n")
    spec2 = importlib.util.spec_from_file_location("stages", stage_file)
    assert spec2 is not None and spec2.loader is not None
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    decorators.no_fingerprint()(mod2.stage_func)

    fp2 = registry._compute_file_fingerprint(mod2.stage_func, [])
    assert fp1 != fp2, "Fingerprint should change when source file is modified"

    stage_info_changed = _make_stage_info(
        mod2.stage_func, tmp_path, code_fingerprint=fp2, outs=[out]
    )

    result2 = executor.execute_stage("src_fp", stage_info_changed, worker_env, output_queue)
    assert result2["status"] == "ran", f"Expected re-run, got: {result2}"
    assert result2["reason"] == "Code changed", "Expected Code changed reason"
    assert _read_output(stage_info_changed, out) == "v2-changed", "Re-run should write v2"


def test_no_fingerprint_reruns_on_code_deps_change(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    (tmp_path / "helper.py").write_text("value = 1\n")

    fingerprint_map = registry._compute_file_fingerprint(
        _helper_no_fingerprint_with_code_dep, ["helper.py"]
    )
    assert "file:helper.py" in fingerprint_map, "Fingerprint should include helper.py"

    out = _output_ref("no_fp_dep")
    stage_info = _make_stage_info(
        _helper_no_fingerprint_with_code_dep,
        tmp_path,
        code_fingerprint=fingerprint_map,
        outs=[out],
    )

    result1 = executor.execute_stage("no_fp_dep", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("no_fp_dep", stage_info, result1)

    (tmp_path / "helper.py").write_text("value = 2\n")
    fingerprint_changed = registry._compute_file_fingerprint(
        _helper_no_fingerprint_with_code_dep, ["helper.py"]
    )
    stage_info_changed = _make_stage_info(
        _helper_no_fingerprint_with_code_dep,
        tmp_path,
        code_fingerprint=fingerprint_changed,
        outs=[out],
    )

    result2 = executor.execute_stage("no_fp_dep", stage_info_changed, worker_env, output_queue)
    assert result2["status"] == "ran", f"Expected re-run, got: {result2}"
    assert result2["reason"] == "Code changed", "Expected Code changed reason"


def test_no_fingerprint_skip_unrelated_change(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", _REPO_ROOT)
    fingerprint_map = registry._compute_file_fingerprint(_helper_no_fingerprint_stage, [])

    out = _output_ref("no_fp")
    stage_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        code_fingerprint=fingerprint_map,
        outs=[out],
    )

    result1 = executor.execute_stage("no_fp", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("no_fp", stage_info, result1)

    (tmp_path / "unrelated.txt").write_text("unrelated")
    result2 = executor.execute_stage("no_fp", stage_info, worker_env, output_queue)
    assert result2["status"] == "cached", f"Expected skip, got: {result2}"


def test_no_fingerprint_mixed_pipeline(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", _REPO_ROOT)
    ast_fingerprint = fingerprint.get_stage_fingerprint_cached(
        "ast_stage", _helper_ast_fingerprint_stage
    )
    no_fp_fingerprint = registry._compute_file_fingerprint(_helper_no_fingerprint_stage, [])

    ast_out = _output_ref("ast_stage")
    no_fp_out = _output_ref("no_fp")

    ast_info = _make_stage_info(
        _helper_ast_fingerprint_stage,
        tmp_path,
        code_fingerprint=ast_fingerprint,
        outs=[ast_out],
    )
    no_fp_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        code_fingerprint=no_fp_fingerprint,
        outs=[no_fp_out],
    )

    result1_ast = executor.execute_stage("ast_stage", ast_info, worker_env, output_queue)
    assert result1_ast["status"] == "ran", f"Expected ran, got: {result1_ast}"
    _apply_deferred_writes("ast_stage", ast_info, result1_ast)

    result1_no_fp = executor.execute_stage("no_fp", no_fp_info, worker_env, output_queue)
    assert result1_no_fp["status"] == "ran", f"Expected ran, got: {result1_no_fp}"
    _apply_deferred_writes("no_fp", no_fp_info, result1_no_fp)

    result2_ast = executor.execute_stage("ast_stage", ast_info, worker_env, output_queue)
    assert result2_ast["status"] == "cached", f"Expected skip, got: {result2_ast}"

    result2_no_fp = executor.execute_stage("no_fp", no_fp_info, worker_env, output_queue)
    assert result2_no_fp["status"] == "cached", f"Expected skip, got: {result2_no_fp}"


def test_no_fingerprint_missing_code_deps_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    with pytest.raises(FileNotFoundError, match="nonexistent.py"):
        registry._compute_file_fingerprint(_helper_no_fingerprint_missing_dep, ["nonexistent.py"])


@pytest.mark.parametrize(
    "stage_func",
    [
        pytest.param(_helper_code_deps_inside_stage, id="no_fingerprint-inside-stage"),
        pytest.param(_helper_code_deps_outside_stage, id="no_fingerprint-outside-stage"),
    ],
)
def test_no_fingerprint_code_deps_tracked_through_pipeline(
    stage_func: Callable[[], str], set_project_root: pathlib.Path
) -> None:
    (set_project_root / "helper.py").write_text("value = 1\n")
    pipeline = compose.Pipeline("test", root=set_project_root)
    with pipeline:
        stage_func()

    fp = pipeline.ensure_fingerprint(f"test/{stage_func.__name__}")

    assert any(k.startswith("file:") for k in fp), f"Expected file-hash fingerprint, got: {fp}"
    assert "file:helper.py" in fp, f"code_deps file must be part of the fingerprint, got: {fp}"
