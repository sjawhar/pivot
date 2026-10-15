from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import exceptions, loaders, types
from pivot.executor import worker
from pivot.storage import cache
from pivot.storage import store as store_mod

if TYPE_CHECKING:
    import multiprocessing as mp

    from pivot.types import OutputMessage


def _path_ref(tag: types.ArtifactTag = types.ArtifactTag.DATA) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="pathret/test", key=None),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=tag,
    )


@pytest.mark.parametrize("path_form", ["direct", "relative", "outward_symlink", "inward_symlink"])
def test_path_output_rejects_lexical_and_resolved_project_membership(
    tmp_path: pathlib.Path, path_form: str
) -> None:
    project_root = tmp_path / "project"
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_input = raw_dir / "input.bin"
    raw_input.write_bytes(b"input")
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external_input = external_dir / "input.bin"
    external_input.write_bytes(b"external")
    outward = project_root / "data" / "outward"
    outward.symlink_to(external_dir, target_is_directory=True)
    inward = tmp_path / "inward"
    inward.symlink_to(raw_dir, target_is_directory=True)

    source = {
        "direct": raw_input,
        "relative": pathlib.Path("data/raw/input.bin"),
        "outward_symlink": outward / "input.bin",
        "inward_symlink": inward / "input.bin",
    }[path_form]

    with pytest.raises(exceptions.StageDefinitionError, match="inside the project"):
        worker._reject_path_inside_project(_path_ref(), source, project_root)
    assert raw_input.read_bytes() == b"input"
    assert external_input.read_bytes() == b"external"


def test_directory_pathonly_members_cannot_move_project_files(tmp_path: pathlib.Path) -> None:
    project_root = tmp_path / "project"
    (project_root / ".pivot").mkdir(parents=True)
    source = project_root / "data" / "raw" / "input.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"input")
    store = store_mod.WorkspaceStore(
        project_root=project_root,
        pipeline_name="pathret",
        input_bindings={},
        cache_dir=project_root / ".pivot" / "cache" / "files",
    )

    with pytest.raises(exceptions.StageDefinitionError, match="inside the project"):
        worker._write_output_with_store(
            _path_ref(types.ArtifactTag.DIRECTORY),
            {"member.bin": source},
            store,
            project_root,
        )
    assert source.read_bytes() == b"input"


@pytest.mark.parametrize("source_kind", ["equal", "child", "parent"])
def test_path_output_rejects_dependency_overlap(tmp_path: pathlib.Path, source_kind: str) -> None:
    project_root = tmp_path / "project"
    dependency = tmp_path / "external" / "bundle"
    dependency.mkdir(parents=True)
    input_file = dependency / "input.bin"
    input_file.write_bytes(b"input")
    child = dependency / "child.bin"
    child.write_bytes(b"child")

    source = {
        "equal": input_file,
        "child": child,
        "parent": dependency,
    }[source_kind]

    with pytest.raises(exceptions.StageDefinitionError, match="overlaps dependency"):
        worker._reject_path_inside_project(
            _path_ref(),
            source,
            project_root,
            dependency_paths=(input_file if source_kind == "equal" else dependency,),
        )
    assert input_file.read_bytes() == b"input"


def _passthrough_external_path(data: pathlib.Path) -> pathlib.Path:
    return data


def test_stage_returning_external_path_dependency_does_not_move_user_file(
    worker_env: pathlib.Path,
    tmp_path: pathlib.Path,
    output_queue: mp.Queue[OutputMessage],
) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external-input.bin"
    external.write_bytes(b"user data")
    state_dir = tmp_path / ".pivot"
    dep = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="external", key=None),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.INPUT,
    )
    out = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="pathret/passthrough", key=None),
        format=loaders.PathOnly(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DATA,
    )
    stage_info = worker.WorkerStageInfo(
        func=_passthrough_external_path,
        fingerprint={"self:test": "abc123"},
        deps={"data": dep},
        outs=[out],
        store_spec={
            "kind": "workspace",
            "cache_dir": str(worker_env / "files"),
            "project_root": str(tmp_path),
            "pipeline_name": "pathret",
            "input_bindings": {"external": str(external)},
            "checkout_modes": cache.DEFAULT_CHECKOUT_MODE_ORDER.copy(),
        },
        signature=None,
        params=None,
        variant=None,
        overrides={},
        run_id="test_run",
        force=False,
        no_commit=True,
        params_arg_name=None,
        project_root=tmp_path,
        state_dir=state_dir,
        project_state_dir=state_dir,
        collection_params={},
        incremental_inputs={},
    )

    result = worker.execute_stage("pathret/passthrough", stage_info, worker_env, output_queue)

    assert result["status"] == types.StageStatus.FAILED
    assert "overlaps dependency" in result["reason"]
    assert external.read_bytes() == b"user data"
