"""Red tests for mutants of PR #523 that the PR's own tests let survive (MutantsRed review).

Each test passed at head 282d7462 and fails on the mutant it names. M39 is adapted to the rule
round 2 set: `track` refuses only a name that collides with another tracked path's files.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import xxhash
import yaml

from pivot import import_artifact, loaders, project, types
from pivot.cli import helpers as cli_helpers
from pivot.registry import RegistryStageInfo
from pivot.show import selection as selection_module
from pivot.storage import listing, lock
from pivot.storage import store as store_mod
from pivot.types import ArtifactIdentity, DirManifestEntry, LockData

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture

    GitRepo = tuple[pathlib.Path, Callable[[str], str]]

_PIVOT = str(pathlib.Path(sys.executable).parent / "pivot")
_FIRST = "abcd123400000000"
_NOTES = "beadfeed00000000"
# A directory output with two files, as pivot 0.5.9 recorded it inline in a lock entry.
_OLD_MANIFEST = [
    {"relpath": "first.csv", "hash": _FIRST, "size": 4, "isexec": False},
    {"relpath": "notes.yaml", "hash": _NOTES, "size": 5, "isexec": False},
]


def _helper_hashed(relpath: str) -> listing.HashedDirectory:
    members = listing.DirListing(
        files=[DirManifestEntry(relpath=relpath, hash="1111111111111111", size=1, isexec=False)],
        empty_dirs=[],
    )
    data = listing.encode(members)
    return listing.HashedDirectory(
        record=listing.record_of(members, data), listing=members, data=data
    )


def _helper_old_lock(outs: list[dict[str, object]]) -> bytes:
    """A lock as pivot 0.5.9 wrote it: schema 2, directory entries listing their files inline."""
    return yaml.safe_dump(
        {
            "schema_version": 2,
            "code_manifest": {"self:stage": "0123456789abcdef"},
            "params": {},
            "deps": [],
            "outs": outs,
        }
    ).encode()


# =============================================================================
# M02: StageLock.write writes the listings after the lock
# =============================================================================


def test_no_lock_is_written_when_writing_its_listing_fails(
    tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Listings are written before the lock, so a failure (or a crash) while writing a listing
    leaves the previous lock in place: no lock names a listing that was never written."""
    identity = ArtifactIdentity("p/stage", None)
    previous = _helper_hashed("old.csv")
    replacement = _helper_hashed("new.csv")
    stage_lock = lock.StageLock("p/stage", tmp_path)
    old_data = LockData(
        code_manifest={}, params={}, dep_hashes={}, output_hashes={identity: previous["record"]}
    )
    stage_lock.write(old_data, hashed_dirs={previous["record"]["hash"]: previous})
    old_lock = stage_lock.path.read_bytes()

    mocker.patch.object(listing, "write", side_effect=OSError("disk full"), autospec=True)
    with pytest.raises(OSError, match="disk full"):
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={identity: replacement["record"]},
            ),
            hashed_dirs={replacement["record"]["hash"]: replacement},
        )

    assert stage_lock.path.read_bytes() == old_lock, "the lock moved on before its listing"


def test_write_refuses_two_listings_that_map_to_the_same_short_name(
    tmp_path: pathlib.Path,
) -> None:
    """When two overlong output keys both hash to the same XXH64 short name, StageLock.write
    refuses before writing anything, naming both paths."""
    from pivot import exceptions
    from xxh64_forge import forge_value

    # Build two distinct overlong keys whose XXH64 short names collide.
    target_hex = xxhash.xxh64(b"seed").hexdigest()
    key_a = forge_value(lambda v: os.fsencode(listing._enc(v)), target_hex, prefix="a" * 250)
    key_b = forge_value(lambda v: os.fsencode(listing._enc(v)), target_hex, prefix="b" * 250)
    assert key_a != key_b
    assert xxhash.xxh64(os.fsencode(listing._enc(key_a))).hexdigest() == target_hex
    assert xxhash.xxh64(os.fsencode(listing._enc(key_b))).hexdigest() == target_hex

    hashed_a = _helper_hashed("a.csv")
    hashed_b = _helper_hashed("b.csv")

    stage_lock = lock.StageLock("p/stage", tmp_path)
    with pytest.raises(exceptions.InvalidPathError, match="two listings to the same file"):
        stage_lock.write(
            LockData(
                code_manifest={},
                params={},
                dep_hashes={},
                output_hashes={
                    ArtifactIdentity("p/stage", key_a): hashed_a["record"],
                    ArtifactIdentity("p/stage", key_b): hashed_b["record"],
                },
            ),
            hashed_dirs={
                hashed_a["record"]["hash"]: hashed_a,
                hashed_b["record"]["hash"]: hashed_b,
            },
        )


# =============================================================================
# M13b: `pivot track` no longer warns when git ignores a file it wrote
# =============================================================================


def test_track_warns_naming_a_listing_file_git_ignores(tmp_path: pathlib.Path) -> None:
    """PR body: `repro` and `track` warn, naming each file, when git ignores one they wrote."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    init = subprocess.run(
        [_PIVOT, "init"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert init.returncode == 0, init.stdout + init.stderr
    (tmp_path / ".gitignore").write_text("*.listing\n")
    corpus = tmp_path / "data" / "raw" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("a")

    tracked = subprocess.run(
        [_PIVOT, "track", "data/raw/corpus"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    out = tracked.stdout + tracked.stderr
    assert tracked.returncode == 0, out
    assert (tmp_path / "data" / "raw" / "corpus.pvt.listing").is_file(), out
    assert "git ignores data/raw/corpus.pvt.listing" in out, out


# =============================================================================
# M39: `pivot track` no longer refuses a target that collides with tracking files
# =============================================================================


def _helper_track_project(root: pathlib.Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    init = subprocess.run(
        [_PIVOT, "init"], cwd=root, env=env, capture_output=True, text=True, check=False
    )
    assert init.returncode == 0, init.stdout + init.stderr
    return env


@pytest.mark.parametrize("name", ["corpus.pvt", "corpus.pvt.listing"])
def test_track_refuses_a_target_that_is_another_tracked_paths_file(
    tmp_path: pathlib.Path, name: str
) -> None:
    """docs/cli/index.md: `track` refuses a target that is the .pvt or listing file of another
    tracked path, naming that path."""
    env = _helper_track_project(tmp_path)
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("a")
    first = subprocess.run(
        [_PIVOT, "track", "corpus"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stdout + first.stderr
    before = (tmp_path / name).read_bytes()

    tracked = subprocess.run(
        [_PIVOT, "track", name], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )

    out = tracked.stdout + tracked.stderr
    assert tracked.returncode != 0, out
    assert f"Cannot track '{name}'" in out and "'corpus'" in out, out
    assert not (tmp_path / f"{name}.pvt").exists(), out
    assert (tmp_path / name).read_bytes() == before


def test_track_refuses_a_target_whose_tracking_file_is_a_tracked_path(
    tmp_path: pathlib.Path,
) -> None:
    env = _helper_track_project(tmp_path)
    (tmp_path / "x.pvt").write_text("data")
    (tmp_path / "x").write_text("other")
    first = subprocess.run(
        [_PIVOT, "track", "x.pvt"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stdout + first.stderr

    tracked = subprocess.run(
        [_PIVOT, "track", "x"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )

    out = tracked.stdout + tracked.stderr
    assert tracked.returncode != 0, out
    assert "Cannot track 'x'" in out and "'x.pvt'" in out, out
    assert (tmp_path / "x.pvt").read_text() == "data"


@pytest.mark.parametrize("name", ["notes.pvt", "corpus.pvt.listing"])
def test_track_tracks_a_name_ending_in_pvt_that_collides_with_nothing(
    tmp_path: pathlib.Path, name: str
) -> None:
    """The base tracks these names; a name collides only with another tracked path's files."""
    env = _helper_track_project(tmp_path)
    (tmp_path / name).write_text("x")

    tracked = subprocess.run(
        [_PIVOT, "track", name], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )

    out = tracked.stdout + tracked.stderr
    assert tracked.returncode == 0, out
    assert (tmp_path / f"{name}.pvt").is_file(), out


# =============================================================================
# M12b, M12c: an over-long input or tracked listing name keeps its full name
# =============================================================================


def test_an_input_listing_name_longer_than_a_file_name_uses_a_short_name() -> None:
    """PR body: a listing name over 255 bytes uses `~<xxh64>` instead."""
    input_path = "data/raw/" + "k" * 250
    stages = pathlib.Path("/s")

    path = listing.input_listing_path(stages, input_path)

    assert path == stages / "=inputs" / f"~{xxhash.xxh64(input_path.encode()).hexdigest()}.listing"
    assert len(os.fsencode(path.name)) <= 255


def test_a_tracked_listing_name_longer_than_a_file_name_uses_a_short_name() -> None:
    """PR body: a listing name over 255 bytes uses `~<xxh64>` instead; for a tracked directory,
    `.~<xxh64>.pvt.listing` beside its `.pvt` (listing.tracked_listing_path)."""
    pvt_path = pathlib.Path("/d") / ("k" * 250 + ".pvt")

    path = listing.tracked_listing_path(pvt_path)

    short = xxhash.xxh64(pvt_path.name.encode()).hexdigest()
    assert path == pathlib.Path("/d") / f".~{short}.pvt.listing"
    assert len(os.fsencode(path.name)) <= 255


# =============================================================================
# M34: diff/data against a HEAD lock an older pivot wrote ignores its inline listing
# =============================================================================


def _helper_stage_info(name: str, outs: list[types.ArtifactRef]) -> RegistryStageInfo:
    def _stage_func() -> None:
        pass

    return RegistryStageInfo(
        func=_stage_func,
        name=name,
        deps={},
        outs=outs,
        incremental_inputs={},
        params=None,
        resources={},
        variant=None,
        signature=inspect.signature(_stage_func),
        fingerprint={"_code": "fake_hash"},
        params_arg_name=None,
        state_dir=None,
        collection_params={},
        no_fingerprint=False,
        code_deps=[],
        input_paths={},
        unsafe_fingerprint=False,
    )


def test_data_selection_reads_a_directory_from_a_head_lock_an_older_pivot_wrote(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, git_repo: GitRepo
) -> None:
    """PR body, rollout: historical reads (`diff`/`data` against HEAD) read an old lock's
    inline list in memory. HEAD holds train's lock as pivot 0.5.9 wrote it, and no listing."""
    monkeypatch.setattr(project, "_project_root_cache", tmp_path)
    workspace_store = store_mod.WorkspaceStore(
        project_root=tmp_path, pipeline_name="eval", input_bindings={}
    )
    monkeypatch.setattr(cli_helpers, "get_workspace_store", lambda: workspace_store)
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="train", key=None),
        format=loaders.CSV(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DIRECTORY,
    )
    stage_info = {"train": _helper_stage_info("train", [ref])}
    monkeypatch.setattr(cli_helpers, "get_all_stages", lambda: stage_info)
    root, commit = git_repo
    lock_path = root / ".pivot" / "stages" / "train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(
        _helper_old_lock(
            [{"key": None, "hash": "0123456789abcdef", "manifest": _OLD_MANIFEST, "empty_dirs": []}]
        )
    )
    commit("written by pivot 0.5.9")

    selection = selection_module.select_data_files(["."])

    assert selection is not None
    assert selection["head_hashes"] == {"data/train/first.csv": _FIRST}


# =============================================================================
# M35: `pivot import` from a source revision an older pivot wrote ignores its inline list
# =============================================================================


async def test_import_resolves_one_file_of_a_directory_from_an_old_source_lock(
    mocker: MockerFixture,
) -> None:
    """PR body, rollout: `pivot import` from an older source revision reads an old lock's
    inline list in memory. The source revision has no listing file."""
    repo_url = "https://github.com/org/repo"
    lock_bytes = _helper_old_lock(
        [{"key": None, "hash": "0123456789abcdef", "manifest": _OLD_MANIFEST, "empty_dirs": []}]
    )
    mocker.patch("pivot.import_artifact.github.list_directory", autospec=True).return_value = [
        {"name": "images.lock", "type": "file"}
    ]
    mocker.patch(
        "pivot.import_artifact.github.resolve_ref", autospec=True
    ).return_value = "deadbeef"

    async def _read_file(
        _owner: str, _repo: str, path: str, _ref: str, _token: str | None = None, **_kw: object
    ) -> bytes | None:
        if path == ".pivot/config.yaml":
            return yaml.safe_dump(
                {"remotes": {"origin": "s3://bucket/prefix/"}, "default_remote": "origin"}
            ).encode()
        if path == ".pivot/stages/images.lock":
            return lock_bytes
        return None  # no listing file: the revision predates listings

    mocker.patch("pivot.import_artifact.github.read_file", autospec=True).side_effect = _read_file

    resolved = await import_artifact.resolve_remote_path(repo_url, "images:first.csv", "main", None)

    assert resolved["hash"] == _FIRST
    assert resolved["size"] == 4
