# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from pivot import exceptions, loaders, types
from pivot.registry import RegistryStageInfo
from pivot.storage import track
from pivot.storage.presentation import present

if TYPE_CHECKING:
    import pathlib


def _make_stage(
    name: str,
    tag: types.ArtifactTag,
    fmt: loaders.Reader[Any] | loaders.Writer[Any] | loaders.Loader[Any, Any],
    *,
    key: str | None,
) -> RegistryStageInfo:
    """Test helper: create a minimal RegistryStageInfo with one output."""
    return RegistryStageInfo(
        func=lambda: None,
        name=name,
        deps={},
        outs=[
            types.ArtifactRef(
                identity=types.ArtifactIdentity(name, key),
                format=fmt,
                python_type=object,
                tag=tag,
            )
        ],
        incremental_inputs={},
        params=None,
        resources={},
        variant=None,
        signature=None,
        fingerprint=dict[str, str](),
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


def test_presentation_creates_symlinks_for_single_output(tmp_path: pathlib.Path) -> None:
    """present() should symlink workspace path to CAS ref for key=None outputs."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    refs_dir = cache_dir / "refs"

    ref_path = refs_dir / "train" / "_single"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text("fake data")

    stages = {"train": _make_stage("train", types.ArtifactTag.DATA, loaders.CSV(), key=None)}

    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "data" / "train.csv"
    assert display_path.is_symlink()
    assert display_path.resolve() == ref_path.resolve()


def test_presentation_creates_symlinks_for_keyed_output(tmp_path: pathlib.Path) -> None:
    """present() should symlink keyed outputs (metrics, plots) correctly."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    refs_dir = cache_dir / "refs"

    ref_path = refs_dir / "eval" / "accuracy"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text("{}")

    stages = {"eval": _make_stage("eval", types.ArtifactTag.METRIC, loaders.JSON(), key="accuracy")}

    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "metrics" / "eval" / "accuracy.json"
    assert display_path.is_symlink()


def test_presentation_skips_missing_refs(tmp_path: pathlib.Path) -> None:
    """present() should silently skip outputs with no CAS ref on disk."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    (cache_dir / "refs").mkdir(parents=True)

    stages = {"missing": _make_stage("missing", types.ArtifactTag.DATA, loaders.CSV(), key=None)}

    present(
        project_root=tmp_path,
        pipeline_name="default",
        cache_dir=cache_dir,
        stages=stages,
    )

    display_path = tmp_path / "data" / "missing.csv"
    assert not display_path.exists()


def test_presentation_refuses_to_replace_output_holding_tracked_file(
    tmp_path: pathlib.Path,
) -> None:
    """present() never replaces a display directory that holds a tracked file."""
    cache_dir = tmp_path / ".pivot" / "cache" / "files"
    ref_path = cache_dir / "refs" / "tree" / "_single"
    ref_path.mkdir(parents=True)
    tracked = tmp_path / "data" / "tree" / "saved.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("user input")
    pvt_path = track.get_pvt_path(tracked)
    track.write_pvt_file(
        pvt_path, track.PvtData(path="saved.txt", hash="abc1230000000000", size=10)
    )
    stages = {
        "tree": _make_stage("tree", types.ArtifactTag.DIRECTORY, loaders.PathOnly(), key=None)
    }

    with pytest.raises(exceptions.PivotError, match=r"saved\.txt\.pvt"):
        present(project_root=tmp_path, pipeline_name="default", cache_dir=cache_dir, stages=stages)

    assert tracked.read_text() == "user input"
    assert pvt_path.exists()
