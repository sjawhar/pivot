"""Integration tests for @no_fingerprint skip detection behavior."""

from __future__ import annotations

import math
import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import decorators, executor, fingerprint, loaders, outputs, registry, stage_def
from pivot.storage import cache, state

if TYPE_CHECKING:
    import inspect
    import multiprocessing as mp
    from collections.abc import Callable

    from pivot.executor import WorkerStageInfo
    from pivot.types import OutputMessage, StageResult


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@decorators.no_fingerprint()
def _helper_no_fingerprint_stage() -> None:
    pathlib.Path("no_fingerprint_output.txt").write_text("output")


@decorators.no_fingerprint(code_deps=["helper.py"])
def _helper_no_fingerprint_with_code_dep() -> None:
    pathlib.Path("code_dep_output.txt").write_text("output")


@decorators.no_fingerprint(code_deps=["nonexistent.py"])
def _helper_no_fingerprint_missing_dep() -> None:
    pathlib.Path("missing_dep_output.txt").write_text("output")


def _helper_ast_fingerprint_stage() -> None:
    value = math.pi
    pathlib.Path("ast_output.txt").write_text(str(value))


def _make_stage_info(
    func: Callable[..., object],
    tmp_path: pathlib.Path,
    *,
    fingerprint: dict[str, str] | None = None,
    deps: list[str] | None = None,
    outs: list[outputs.BaseOut] | None = None,
    params: stage_def.StageParams | None = None,
    signature: inspect.Signature | None = None,
    checkout_modes: list[cache.CheckoutMode] | None = None,
    run_id: str = "test_run",
    force: bool = False,
    no_commit: bool = False,
    dep_specs: dict[str, stage_def.FuncDepSpec] | None = None,
    out_specs: dict[str, outputs.BaseOut] | None = None,
    params_arg_name: str | None = None,
) -> WorkerStageInfo:
    expanded_outs = [outputs.require_expanded(out) for out in outs] if outs else []
    expanded_out_specs = out_specs or {}
    return {
        "func": func,
        "fingerprint": fingerprint or {"self:test": "abc123"},
        "deps": deps or [],
        "signature": signature,
        "outs": expanded_outs,
        "params": params,
        "variant": None,
        "overrides": {},
        "checkout_modes": checkout_modes
        or [cache.CheckoutMode.HARDLINK, cache.CheckoutMode.SYMLINK, cache.CheckoutMode.COPY],
        "run_id": run_id,
        "force": force,
        "no_commit": no_commit,
        "dep_specs": dep_specs or {},
        "out_specs": expanded_out_specs,
        "params_arg_name": params_arg_name,
        "project_root": tmp_path,
        "state_dir": tmp_path / ".pivot",
    }


def _apply_deferred_writes(
    stage_name: str,
    stage_info: WorkerStageInfo,
    result: StageResult,
) -> None:
    if "deferred_writes" not in result:
        return
    deferred = result["deferred_writes"]
    state_dir = stage_info["state_dir"]
    out_paths = [str(out.path) for out in stage_info["outs"]]

    with state.StateDB(state_dir / "state.db") as state_db:
        state_db.apply_deferred_writes(stage_name, out_paths, deferred)


def test_no_fingerprint_stage_skips_when_unchanged(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", _REPO_ROOT)
    fingerprint_map = registry._compute_file_fingerprint(_helper_no_fingerprint_stage)
    out = outputs.Out("no_fingerprint_output.txt", loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        fingerprint=fingerprint_map,
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
    stage_file.write_text(
        "import pathlib\ndef stage_func():\n    pathlib.Path('source_output.txt').write_text('v1')\n"
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("stages", stage_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decorators.no_fingerprint()(mod.stage_func)

    fp1 = registry._compute_file_fingerprint(mod.stage_func)
    assert any(k.startswith("file:") for k in fp1), "Fingerprint should include file: keys"

    out = outputs.Out("source_output.txt", loader=loaders.PathOnly())
    stage_info = _make_stage_info(mod.stage_func, tmp_path, fingerprint=fp1, outs=[out])

    result1 = executor.execute_stage("src_fp", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("src_fp", stage_info, result1)

    stage_file.write_text(
        "import pathlib\ndef stage_func():\n    pathlib.Path('source_output.txt').write_text('v2')\n"
    )
    spec2 = importlib.util.spec_from_file_location("stages", stage_file)
    assert spec2 is not None and spec2.loader is not None
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    decorators.no_fingerprint()(mod2.stage_func)

    fp2 = registry._compute_file_fingerprint(mod2.stage_func)
    assert fp1 != fp2, "Fingerprint should change when source file is modified"

    stage_info_changed = _make_stage_info(mod2.stage_func, tmp_path, fingerprint=fp2, outs=[out])

    result2 = executor.execute_stage("src_fp", stage_info_changed, worker_env, output_queue)
    assert result2["status"] == "ran", f"Expected re-run, got: {result2}"
    assert result2["reason"] == "Code changed", "Expected Code changed reason"


def test_no_fingerprint_reruns_on_code_deps_change(
    worker_env: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pivot.project._project_root_cache", tmp_path)
    (tmp_path / "helper.py").write_text("value = 1\n")

    fingerprint_map = registry._compute_file_fingerprint(_helper_no_fingerprint_with_code_dep)
    assert "file:helper.py" in fingerprint_map, "Fingerprint should include helper.py"

    out = outputs.Out("code_dep_output.txt", loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        _helper_no_fingerprint_with_code_dep,
        tmp_path,
        fingerprint=fingerprint_map,
        outs=[out],
    )

    result1 = executor.execute_stage("no_fp_dep", stage_info, worker_env, output_queue)
    assert result1["status"] == "ran", f"Expected ran, got: {result1}"
    _apply_deferred_writes("no_fp_dep", stage_info, result1)

    (tmp_path / "helper.py").write_text("value = 2\n")
    fingerprint_changed = registry._compute_file_fingerprint(_helper_no_fingerprint_with_code_dep)
    stage_info_changed = _make_stage_info(
        _helper_no_fingerprint_with_code_dep,
        tmp_path,
        fingerprint=fingerprint_changed,
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
    fingerprint_map = registry._compute_file_fingerprint(_helper_no_fingerprint_stage)

    out = outputs.Out("no_fingerprint_output.txt", loader=loaders.PathOnly())
    stage_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        fingerprint=fingerprint_map,
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
    no_fp_fingerprint = registry._compute_file_fingerprint(_helper_no_fingerprint_stage)

    ast_out = outputs.Out("ast_output.txt", loader=loaders.PathOnly())
    no_fp_out = outputs.Out("no_fingerprint_output.txt", loader=loaders.PathOnly())

    ast_info = _make_stage_info(
        _helper_ast_fingerprint_stage,
        tmp_path,
        fingerprint=ast_fingerprint,
        outs=[ast_out],
    )
    no_fp_info = _make_stage_info(
        _helper_no_fingerprint_stage,
        tmp_path,
        fingerprint=no_fp_fingerprint,
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
        registry._compute_file_fingerprint(_helper_no_fingerprint_missing_dep)
