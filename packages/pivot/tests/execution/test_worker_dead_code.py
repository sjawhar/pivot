from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from pivot.executor import worker
from pivot.storage import cache

if TYPE_CHECKING:
    import multiprocessing as mp
    import pathlib
    from collections.abc import Callable

    from pivot import types
    from pivot.types import OutputMessage


def _stage_noop() -> None:
    return None


def _make_stage_info(
    func: Callable[..., object],
    tmp_path: pathlib.Path,
    deps: object,
) -> worker.WorkerStageInfo:
    return worker.WorkerStageInfo(
        func=func,
        fingerprint={"self:test": "abc123"},
        deps=cast("dict[str, types.ArtifactRef]", deps),
        signature=None,
        outs=[],
        store_spec={
            "kind": "workspace",
            "cache_dir": str(tmp_path / ".pivot" / "cache"),
            "project_root": str(tmp_path),
            "pipeline_name": "test",
            "input_bindings": {},
        },
        params=None,
        variant=None,
        overrides={},
        checkout_modes=[
            cache.CheckoutMode.HARDLINK,
            cache.CheckoutMode.SYMLINK,
            cache.CheckoutMode.COPY,
        ],
        run_id="test_run",
        force=False,
        no_commit=True,
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=tmp_path / ".pivot",
        collection_params={},
    )


def test_execute_stage_rejects_legacy_deps_list(
    worker_env: pathlib.Path, tmp_path: pathlib.Path, output_queue: mp.Queue[OutputMessage]
) -> None:
    stage_info = _make_stage_info(_stage_noop, tmp_path, deps=[])

    with pytest.raises(AttributeError, match="values"):
        _ = worker.execute_stage("test_stage", stage_info, worker_env, output_queue)
