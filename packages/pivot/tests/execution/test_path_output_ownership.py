from __future__ import annotations

import pathlib

import pytest

from pivot import exceptions, loaders, types
from pivot.executor import worker
from pivot.storage import store as store_mod


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
