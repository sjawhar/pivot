# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false, reportAny=false, reportExplicitAny=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnusedParameter=false, reportUnusedCallResult=false
from __future__ import annotations

import errno
import logging
import multiprocessing
import os
import pathlib
import pickle
import stat
from typing import TYPE_CHECKING, cast

import pytest

from pivot import config, exceptions, loaders, types
from pivot.storage import cache, listing, ownership, store, track

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


def _helper_ref(
    producer: str,
    key: str | None,
    tag: types.ArtifactTag,
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object],
) -> types.ArtifactRef:
    return types.ArtifactRef(
        identity=types.ArtifactIdentity(producer=producer, key=key),
        format=fmt,
        python_type=pathlib.Path,
        tag=tag,
    )


def test_cache_store_round_trip(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    state_dir = tmp_path / ".pivot"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_dir=state_dir)

    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("hello cache")

    cache_store.commit(ref, output_path)
    checkout_path = cache_store.checkout(ref)

    assert checkout_path.read_text() == "hello cache"
    assert cache_store.exists(ref) is True


def test_cache_store_prepare_output_commit_pattern(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_dir=None)

    ref = _helper_ref("stage", "output", types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("content")

    output_hash = cache_store.commit(ref, output_path)
    assert len(output_hash["hash"]) == 16
    assert cache_store.checkout(ref).read_text() == "content"


def test_cache_store_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_dir=None)
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = cache_store.prepare_output(ref)
    output_path.write_text("pickle cache")
    cache_store.commit(ref, output_path)

    restored = pickle.loads(pickle.dumps(cache_store))
    assert restored.checkout(ref).read_text() == "pickle cache"


def test_cache_store_hash_artifact_types(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_store = store.CacheStore(cache_dir=cache_dir, state_dir=None)

    file_ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    file_path = cache_store.prepare_output(file_ref)
    file_path.write_text("hash me")
    cache_store.commit(file_ref, file_path)

    file_hash = cache_store.hash_artifact(file_ref)
    assert "hash" in file_hash
    assert "manifest" not in file_hash

    dir_ref = _helper_ref("stage", "dir", types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    dir_path = cache_store.prepare_output(dir_ref)
    (dir_path / "a.txt").write_text("a")
    cache_store.commit(dir_ref, dir_path)

    hashed_dirs = dict[str, listing.HashedDirectory]()
    dir_hash = cache_store.hash_artifact(dir_ref, hashed_dirs=hashed_dirs)
    assert types.is_dir_hash(dir_hash)
    assert {entry["relpath"] for entry in hashed_dirs[dir_hash["hash"]]["listing"]["files"]} == {
        "a.txt"
    }


def test_workspace_store_round_trip(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("hello workspace")

    workspace_store.commit(ref, output_path)
    checkout_path = workspace_store.checkout(ref)

    assert checkout_path.read_text() == "hello workspace"
    assert workspace_store.exists(ref) is True


def test_workspace_store_prepare_output_commit_pattern(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", "out", types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("content")

    output_hash = workspace_store.commit(ref, output_path)
    assert len(output_hash["hash"]) == 16
    assert workspace_store.checkout(ref).read_text() == "content"


def test_workspace_store_commit_and_restore(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "cache" / "files"
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={},
        cache_dir=cache_dir,
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("hello cache")

    output_hash = workspace_store.commit(ref, output_path)

    output_path.unlink()
    assert not output_path.exists()

    restored = workspace_store.restore_from_cache(ref, output_hash)
    assert restored is True
    assert output_path.read_text() == "hello cache"


def test_workspace_store_keeps_a_tracked_file_that_lands_during_restore_staging(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run-cache restore refuses a user file and sidecar that land while its directory
    replacement is staged."""
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={},
        cache_dir=tmp_path / "cache" / "files",
        checkout_modes=[config.CheckoutMode.COPY],
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    (output_path / "cached.txt").write_text("cached")
    output_hash = workspace_store.commit(ref, output_path)
    cache.remove_output(output_path)

    original_checkout = cache._checkout_with_fallback
    landed = False

    def stage_then_copy_user_file(
        path: pathlib.Path,
        cache_path: pathlib.Path,
        checkout_modes: list[cache.CheckoutMode],
        *,
        executable: bool | None = None,
        staging_path: pathlib.Path | None = None,
        keep_existing_copy: bool = False,
        display_path: pathlib.Path | None = None,
        before_swap: Callable[[pathlib.Path], None] | None = None,
        cache_root: cache.CacheRoot | None = None,
    ) -> os.stat_result:
        nonlocal landed
        staged = original_checkout(
            path,
            cache_path,
            checkout_modes,
            executable=executable,
            staging_path=staging_path,
            keep_existing_copy=keep_existing_copy,
            display_path=display_path,
            before_swap=before_swap,
            cache_root=cache_root,
        )
        if not landed:
            output_path.mkdir()
            user_file = output_path / "user.txt"
            user_file.write_text("user bytes")
            track.write_pvt_file(
                track.get_pvt_path(user_file),
                track.PvtData(path="user.txt", hash="0123456789abcdef", size=10),
            )
            landed = True
        return staged

    monkeypatch.setattr(cache, "_checkout_with_fallback", stage_then_copy_user_file)

    with pytest.raises(exceptions.OutputOwnershipError, match="user.txt.pvt"):
        workspace_store.restore_from_cache(ref, output_hash)

    assert landed
    assert (output_path / "user.txt").read_text() == "user bytes"
    assert track.get_pvt_path(output_path / "user.txt").is_file()


def test_workspace_store_names_a_late_tracked_file_before_the_stage_writer_runs(
    tmp_path: pathlib.Path,
) -> None:
    """The check after preparing a forced run's output uses the forced-restore refusal, naming
    the copied sidecar and target instead of letting the stage writer report a directory error."""
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={},
        cache_dir=tmp_path / "cache" / "files",
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    user_file = output_path / "user.txt"
    user_file.write_text("user bytes")
    track.write_pvt_file(
        track.get_pvt_path(user_file),
        track.PvtData(path="user.txt", hash="0123456789abcdef", size=10),
    )

    with pytest.raises(exceptions.OutputOwnershipError) as error:
        workspace_store.refuse_output_replacement_at_write(ref)

    message = str(error.value)
    assert "user.txt.pvt" in message
    assert "data/stage" in message


def _copy_user_file(user_file: pathlib.Path) -> None:
    """Copy a user's tracked file and its sidecar to ``user_file``, as a user would by hand."""
    user_file.write_text("user bytes")
    track.write_pvt_file(
        track.get_pvt_path(user_file),
        track.PvtData(path=user_file.name, hash="0123456789abcdef", size=10),
    )


def _copy_user_file_after_the_first_check(
    monkeypatch: pytest.MonkeyPatch, workspace_store: store.WorkspaceStore, user_file: pathlib.Path
) -> None:
    """Copy a user's tracked file and sidecar to ``user_file`` right after prepare_output's first
    ownership check returns, before the old output is removed."""
    first_check = workspace_store.refuse_output_replacements

    def check_then_copy_user_file(
        refs: Iterable[types.ArtifactRef],
        claims: ownership.StageClaims | None = None,
        returned: Mapping[types.ArtifactRef, ownership.StageMembers] | None = None,
    ) -> None:
        first_check(refs, claims, returned)
        _copy_user_file(user_file)

    monkeypatch.setattr(workspace_store, "refuse_output_replacements", check_then_copy_user_file)


def _helper_backups(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.glob(f"{cache._BACKUP_TEMP_PREFIX}*"))


def _helper_stage_files(output_path: pathlib.Path) -> None:
    for index in range(3):
        (output_path / f"f{index}.txt").write_text(f"stage {index}")


def test_workspace_store_keeps_a_tracked_file_copied_in_after_the_prepare_check(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user file and sidecar copied into a directory output after prepare_output's first check,
    before the old output is removed, are found once the output is renamed aside: the store
    refuses naming them and the output, and puts them back with the output, never deleted."""
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={},
        cache_dir=tmp_path / "cache" / "files",
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    _helper_stage_files(output_path)
    user_file = output_path / "user.txt"
    _copy_user_file_after_the_first_check(monkeypatch, workspace_store, user_file)

    refusal: exceptions.OutputOwnershipError | None = None
    try:
        workspace_store.prepare_output(ref)
    except exceptions.OutputOwnershipError as error:
        refusal = error

    left = sorted(path.name for path in output_path.iterdir()) if output_path.is_dir() else []
    assert user_file.is_file(), f"the user file was deleted; the output holds {left}"
    assert track.get_pvt_path(user_file).is_file(), f"the sidecar was deleted; output: {left}"
    assert user_file.read_text() == "user bytes"
    assert refusal is not None
    message = str(refusal)
    assert "user.txt.pvt" in message
    assert "data/stage" in message
    assert left == ["f0.txt", "f1.txt", "f2.txt", "user.txt", "user.txt.pvt"]
    assert _helper_backups(output_path.parent) == []


def test_workspace_store_keeps_a_tracked_file_moved_over_a_file_output_after_the_check(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file output is checked again in place, just before it is unlinked: a user's tracked file
    written over it, with its sidecar beside it, after prepare_output's first check is found
    there, refused, and kept at the output's path."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("stage")
    _copy_user_file_after_the_first_check(monkeypatch, workspace_store, output_path)

    with pytest.raises(exceptions.OutputOwnershipError, match=r"data/stage\.txt\.pvt"):
        workspace_store.prepare_output(ref)

    assert output_path.read_text() == "user bytes"
    assert track.get_pvt_path(output_path).is_file()
    assert sorted(path.name for path in output_path.parent.iterdir()) == [
        "stage.txt",
        "stage.txt.pvt",
    ]


def _record_renames(monkeypatch: pytest.MonkeyPatch) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Record every ``pathlib.Path.rename`` from now on, and let each one go ahead."""
    original_rename = pathlib.Path.rename
    renames = list[tuple[pathlib.Path, pathlib.Path]]()

    def rename(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        renames.append((source, pathlib.Path(target)))
        return original_rename(source, target)

    monkeypatch.setattr(pathlib.Path, "rename", rename)
    return renames


def test_prepare_output_unlinks_a_file_output_without_renaming_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file output is checked in place and unlinked, never renamed aside: on overlayfs,
    renaming a file on the lower layer first copies all of it into the writable layer. Under the
    restore lock no backup is made either."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("stage")
    renames = _record_renames(monkeypatch)

    assert workspace_store.prepare_output(ref, state_dir=tmp_path / ".pivot") == output_path

    assert renames == []
    assert not output_path.exists()
    assert _helper_backups(output_path.parent) == []


def test_prepare_output_never_removes_a_directory_that_takes_a_file_outputs_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory a user puts at a file output's path right after its in-place check is not
    removed: the unlink fails on it, the stage write fails, and the directory keeps its files."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("stage")
    original_check = ownership.swap_refusal

    def check_then_replace_with_a_directory(
        project_root: pathlib.Path, target: ownership.ForcedRestore, entry: pathlib.Path
    ) -> exceptions.OutputOwnershipError | None:
        refusal = original_check(project_root, target, entry)
        output_path.unlink()
        output_path.mkdir()
        _copy_user_file(output_path / "user.txt")
        return refusal

    monkeypatch.setattr(ownership, "swap_refusal", check_then_replace_with_a_directory)

    with pytest.raises(IsADirectoryError):
        workspace_store.prepare_output(ref)

    assert sorted(path.name for path in output_path.iterdir()) == ["user.txt", "user.txt.pvt"]


def test_workspace_store_puts_the_output_back_when_its_removal_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When removing the renamed-aside output fails with no user file in it (here a file added to
    it through an open handle after its check), the output goes back to its path and the
    removal's own error propagates."""
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    (output_path / "f0.txt").write_text("stage")
    original_unlink = os.unlink
    landed = list[pathlib.Path]()

    def unlink_after_a_stray_file_lands(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *, dir_fd: int | None = None
    ) -> None:
        if not landed:
            # shutil.rmtree has listed the backup: a file added to it now fails its rmdir
            (backup,) = _helper_backups(output_path.parent)
            landed.append(backup)
            (backup / "stray.txt").write_text("stray")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", unlink_after_a_stray_file_lands)

    with pytest.raises(OSError, match="Directory not empty") as error:
        workspace_store.prepare_output(ref)

    assert len(landed) == 1
    # The restore token's backup name: the next directory restore of the path checks a killed
    # run's backup before removing it (cache._cleanup_fresh_restore_orphans)
    assert landed[0].name.startswith(
        f"{cache._BACKUP_TEMP_PREFIX}{cache._restore_target_token(output_path)}."
    )
    assert error.value.errno == errno.ENOTEMPTY
    assert (output_path / "stray.txt").read_text() == "stray"
    assert _helper_backups(output_path.parent) == []


def test_prepare_output_clears_a_directory_output_in_a_read_only_parent(
    tmp_path: pathlib.Path,
) -> None:
    """Renaming the old output aside makes a read-only parent writable, as removing it did."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    (output_path / "old.txt").write_text("old")
    parent_mode = output_path.parent.stat().st_mode
    os.chmod(output_path.parent, stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = workspace_store.prepare_output(ref)
    finally:
        os.chmod(output_path.parent, parent_mode)

    assert result == output_path
    assert list(output_path.iterdir()) == []


def _rename_failing_aside(monkeypatch: pytest.MonkeyPatch, error_number: int) -> list[pathlib.Path]:
    """Make every rename to a backup name fail with ``error_number``, as overlayfs with
    redirect_dir off fails a directory on its lower layer with EXDEV; return the entries refused."""
    original_rename = pathlib.Path.rename
    refused = list[pathlib.Path]()

    def rename(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if pathlib.Path(target).name.startswith(cache._BACKUP_TEMP_PREFIX):
            refused.append(source)
            raise OSError(error_number, os.strerror(error_number), str(source), str(target))
        return original_rename(source, target)

    monkeypatch.setattr(pathlib.Path, "rename", rename)
    return refused


@pytest.mark.parametrize("error_number", [errno.EXDEV, errno.ENOTSUP], ids=["EXDEV", "ENOTSUP"])
def test_prepare_output_clears_an_output_it_cannot_rename_aside(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    """Where the filesystem cannot rename the old output aside, prepare_output checks it in place
    and removes it, rather than fail every stage with an existing directory output."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    _helper_stage_files(output_path)
    refused = _rename_failing_aside(monkeypatch, error_number)

    assert workspace_store.prepare_output(ref) == output_path

    assert refused == [output_path]
    assert list(output_path.iterdir()) == []
    assert _helper_backups(output_path.parent) == []


def test_prepare_output_refuses_a_tracked_file_in_an_output_it_cannot_rename_aside(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check made in place of the rename still finds a user file and sidecar copied into the
    output after the first check, refuses naming them, and removes nothing."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    _helper_stage_files(output_path)
    _copy_user_file_after_the_first_check(monkeypatch, workspace_store, output_path / "user.txt")
    refused = _rename_failing_aside(monkeypatch, errno.EXDEV)

    with pytest.raises(exceptions.OutputOwnershipError, match=r"user\.txt\.pvt"):
        workspace_store.prepare_output(ref)

    assert refused == [output_path]
    assert sorted(path.name for path in output_path.iterdir()) == [
        "f0.txt",
        "f1.txt",
        "f2.txt",
        "user.txt",
        "user.txt.pvt",
    ]


def test_prepare_output_fails_on_any_other_rename_error_and_keeps_the_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a filesystem that cannot rename the entry gets the in-place removal: any other rename
    error (EBUSY: the output is a mount point) fails before anything is removed."""
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path = workspace_store.prepare_output(ref)
    _helper_stage_files(output_path)
    _ = _rename_failing_aside(monkeypatch, errno.EBUSY)

    with pytest.raises(OSError) as error:
        workspace_store.prepare_output(ref)

    assert error.value.errno == errno.EBUSY
    assert sorted(path.name for path in output_path.iterdir()) == ["f0.txt", "f1.txt", "f2.txt"]


def _helper_cached_directory_output(
    workspace_store: store.WorkspaceStore, ref: types.ArtifactRef
) -> tuple[pathlib.Path, types.HashInfo]:
    """A committed directory output whose workspace copy then changed: a restore replaces it."""
    output_path = workspace_store.prepare_output(ref)
    (output_path / "cached.txt").write_text("cached")
    output_hash = workspace_store.commit(ref, output_path)
    (output_path / "cached.txt").write_text("changed")
    (output_path / "extra.txt").write_text("extra")
    return output_path, output_hash


def test_workspace_store_restores_over_a_directory_it_cannot_rename_aside(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore's swap over a directory the filesystem cannot rename aside (a forced checkout on
    overlayfs) checks it in place, removes it, and puts the restored directory there."""
    workspace_store = store.WorkspaceStore(
        tmp_path,
        "pipe",
        {},
        cache_dir=tmp_path / "cache" / "files",
        checkout_modes=[config.CheckoutMode.COPY],
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path, output_hash = _helper_cached_directory_output(workspace_store, ref)
    refused = _rename_failing_aside(monkeypatch, errno.EXDEV)

    assert workspace_store.restore_from_cache(ref, output_hash) is True

    assert refused == [output_path]
    assert sorted(path.name for path in output_path.iterdir()) == ["cached.txt"]
    assert (output_path / "cached.txt").read_text() == "cached"
    assert _helper_backups(output_path.parent) == []


def test_workspace_store_restore_refuses_a_tracked_file_in_a_directory_it_cannot_rename_aside(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The swap's check made in place of the rename finds a user file and sidecar copied into
    the output while its replacement was built, refuses naming them, and removes nothing."""
    workspace_store = store.WorkspaceStore(
        tmp_path,
        "pipe",
        {},
        cache_dir=tmp_path / "cache" / "files",
        checkout_modes=[config.CheckoutMode.COPY],
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())
    output_path, output_hash = _helper_cached_directory_output(workspace_store, ref)
    original_checkout = cache._checkout_with_fallback

    def stage_then_copy_user_file(
        path: pathlib.Path,
        cache_path: pathlib.Path,
        checkout_modes: list[cache.CheckoutMode],
        *,
        executable: bool | None = None,
        staging_path: pathlib.Path | None = None,
        keep_existing_copy: bool = False,
        display_path: pathlib.Path | None = None,
        before_swap: Callable[[pathlib.Path], None] | None = None,
        cache_root: cache.CacheRoot | None = None,
    ) -> os.stat_result:
        staged = original_checkout(
            path,
            cache_path,
            checkout_modes,
            executable=executable,
            staging_path=staging_path,
            keep_existing_copy=keep_existing_copy,
            display_path=display_path,
            before_swap=before_swap,
            cache_root=cache_root,
        )
        if not (output_path / "user.txt").exists():
            _copy_user_file(output_path / "user.txt")
        return staged

    monkeypatch.setattr(cache, "_checkout_with_fallback", stage_then_copy_user_file)
    refused = _rename_failing_aside(monkeypatch, errno.EXDEV)

    with pytest.raises(exceptions.OutputOwnershipError, match=r"user\.txt\.pvt"):
        workspace_store.restore_from_cache(ref, output_hash)

    assert refused == [output_path]
    assert sorted(path.name for path in output_path.iterdir()) == [
        "cached.txt",
        "extra.txt",
        "user.txt",
        "user.txt.pvt",
    ]
    assert (output_path / "user.txt").read_text() == "user bytes"


_DIRECTORY_REF = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.PathOnly())


def _helper_prepare_output_killed_after_its_rename(
    project_root: pathlib.Path, state_dir: pathlib.Path
) -> None:
    """Run in a child process: a stage write killed the moment it renamed the old output aside."""
    original_rename = pathlib.Path.rename

    def rename_and_die(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        result = original_rename(source, target)
        if pathlib.Path(target).name.startswith(cache._BACKUP_TEMP_PREFIX):
            os._exit(73)
        return result

    pathlib.Path.rename = rename_and_die
    workspace_store = store.WorkspaceStore(project_root, "pipe", {})
    workspace_store.prepare_output(_DIRECTORY_REF, state_dir=state_dir)


def _helper_killed_stage_write(
    project_root: pathlib.Path, state_dir: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write an old directory output, then kill a stage write of it right after it renamed the
    output aside; return the output's path and the backup the killed write left."""
    output_path = store.WorkspaceStore(project_root, "pipe", {}).prepare_output(
        _DIRECTORY_REF, state_dir=state_dir
    )
    (output_path / "old.txt").write_text("old")
    process = multiprocessing.get_context("spawn").Process(
        target=_helper_prepare_output_killed_after_its_rename, args=(project_root, state_dir)
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 73
    assert not output_path.exists()
    (backup,) = _helper_backups(output_path.parent)
    token = cache._restore_target_token(output_path)
    assert backup.name == f"{cache._BACKUP_TEMP_PREFIX}{token}.old"
    return output_path, backup


def test_prepare_output_removes_the_backup_a_killed_stage_write_left(
    tmp_path: pathlib.Path,
) -> None:
    """A stage write killed between renaming its old directory output aside and removing it
    leaves the output beside its path. The next stage write of that path, under its restore lock,
    checks that backup, finds no user file, and removes it, even though nothing is at the path."""
    state_dir = tmp_path / ".pivot"
    output_path, _ = _helper_killed_stage_write(tmp_path, state_dir)
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})

    assert workspace_store.prepare_output(_DIRECTORY_REF, state_dir=state_dir) == output_path

    assert _helper_backups(output_path.parent) == []


def test_prepare_output_keeps_a_killed_stage_writes_backup_that_holds_a_user_file(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A user's tracked file in that backup (copied into the output just before the killed write
    renamed it aside) keeps the backup, with a warning naming the sidecar; the stage write still
    clears its path, through another backup it removes."""
    state_dir = tmp_path / ".pivot"
    output_path, backup = _helper_killed_stage_write(tmp_path, state_dir)
    _copy_user_file(backup / "user.txt")
    output_path.mkdir()
    (output_path / "new.txt").write_text("a later run's file")
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})

    with caplog.at_level(logging.WARNING, logger="pivot.storage.cache"):
        assert workspace_store.prepare_output(_DIRECTORY_REF, state_dir=state_dir) == output_path

    assert list(output_path.iterdir()) == []
    assert _helper_backups(output_path.parent) == [backup]
    assert sorted(path.name for path in backup.iterdir()) == ["old.txt", "user.txt", "user.txt.pvt"]
    assert "Keeping interrupted restore backup" in caplog.text
    assert "user.txt.pvt" in caplog.text


def test_workspace_store_restore_records_pvt_members_before_writing_them(
    tmp_path: pathlib.Path,
) -> None:
    """A restore of a hash no lock records (a run-cache restore) records each `*.pvt` member it
    writes, with the content it will hold, after the ownership check and before it writes
    anything."""
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={},
        cache_dir=tmp_path / "cache" / "files",
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DIRECTORY, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    notes = output_path / "notes.pvt"
    notes.write_text("stage notes")
    (output_path / "value.txt").write_text("v")
    output_hash = workspace_store.commit(
        ref, output_path, written={ownership.normalized(notes): frozenset[str]()}
    )
    cache.remove_output(output_path)
    recorded = list[dict[pathlib.Path, str]]()

    def record(members: Mapping[pathlib.Path, str]) -> None:
        assert not output_path.exists()
        recorded.append(dict(members))

    claims = ownership.StageClaims(locked={}, no_commit={}, recorder=record, cache_dir=None)
    assert workspace_store.restore_from_cache(ref, output_hash, claims=claims) is True

    assert recorded == [{ownership.normalized(notes): cache.hash_file(notes)[0]}]
    assert notes.read_text() == "stage notes"


def test_workspace_store_pickle_round_trip(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output_path = workspace_store.prepare_output(ref)
    output_path.write_text("pickle workspace")
    workspace_store.commit(ref, output_path)

    restored = pickle.loads(pickle.dumps(workspace_store))
    assert restored.checkout(ref).read_text() == "pickle workspace"


@pytest.mark.parametrize(
    ("tag", "key", "expected"),
    [
        pytest.param(
            types.ArtifactTag.DATA,
            None,
            "data/stage.txt",
            id="data-single",
        ),
        pytest.param(
            types.ArtifactTag.DATA,
            "out",
            "data/stage/out.txt",
            id="data-multi",
        ),
        pytest.param(
            types.ArtifactTag.METRIC,
            "score",
            "metrics/stage/score.json",
            id="metric",
        ),
        pytest.param(
            types.ArtifactTag.PLOT,
            "loss",
            "plots/stage/loss.png",
            id="plot",
        ),
        pytest.param(
            types.ArtifactTag.DIRECTORY,
            "dir",
            "data/stage/dir",
            id="directory",
        ),
    ],
)
def test_workspace_store_path_resolution(
    tmp_path: pathlib.Path,
    tag: types.ArtifactTag,
    key: str | None,
    expected: str,
) -> None:
    fmt: loaders.Writer[object] | loaders.Reader[object] | loaders.Loader[object, object]
    if tag == types.ArtifactTag.METRIC:
        fmt = cast("loaders.Writer[object]", loaders.JSON[dict[str, object]]())
    elif tag == types.ArtifactTag.PLOT:
        fmt = cast("loaders.Writer[object]", loaders.MatplotlibFigure())
    elif tag == types.ArtifactTag.DIRECTORY:
        fmt = cast("loaders.Reader[object]", loaders.PathOnly())
    else:
        fmt = cast("loaders.Writer[object]", loaders.Text())

    workspace_store = store.WorkspaceStore(
        project_root=tmp_path, pipeline_name="pipe", input_bindings={}
    )
    ref = _helper_ref("stage", key, tag, fmt)

    resolved = workspace_store.prepare_output(ref)
    assert resolved == tmp_path / expected


def test_workspace_store_input_bindings(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={"custom": "data/external/custom.csv"},
    )
    custom_ref = _helper_ref("custom", None, types.ArtifactTag.INPUT, loaders.Text())
    default_ref = _helper_ref("rawdata", None, types.ArtifactTag.INPUT, loaders.Text())

    assert workspace_store.checkout(custom_ref) == tmp_path / "data/external/custom.csv"
    assert workspace_store.checkout(default_ref) == tmp_path / "data/raw/rawdata"


def test_workspace_store_collision_detection(tmp_path: pathlib.Path) -> None:
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="pipe",
        input_bindings={"foo": "data/raw/bar"},
    )
    foo_ref = _helper_ref("foo", None, types.ArtifactTag.INPUT, loaders.Text())
    bar_ref = _helper_ref("bar", None, types.ArtifactTag.INPUT, loaders.Text())

    _ = workspace_store.checkout(foo_ref)
    with pytest.raises(ValueError, match="Collision"):
        _ = workspace_store.checkout(bar_ref)


def test_workspace_store_resolves_external_input(tmp_path: pathlib.Path) -> None:
    (tmp_path / "data" / "external").mkdir(parents=True)
    (tmp_path / "data" / "external" / "ext.jsonl").write_text("[]")

    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="test",
        input_bindings={"ext.jsonl": "data/external/ext.jsonl"},
    )
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity("ext.jsonl", None),
        format=loaders.DataFrameJSONL(),
        python_type=list,
        tag=types.ArtifactTag.INPUT,
    )

    assert workspace_store.exists(ref)


def test_store_spec_round_trip(tmp_path: pathlib.Path) -> None:
    spec: store.StoreSpec = {
        "kind": "workspace",
        "cache_dir": str(tmp_path / "cache"),
        "project_root": str(tmp_path),
        "pipeline_name": "pipe",
        "input_bindings": {"raw": "data/raw/raw"},
        "checkout_modes": [config.CheckoutMode.COPY],
    }

    round_trip = pickle.loads(pickle.dumps(spec))
    rebuilt = store.store_from_spec(round_trip)
    assert isinstance(rebuilt, store.WorkspaceStore)


def test_prepare_output_unlinks_readonly_cas_hardlink(tmp_path: pathlib.Path) -> None:
    ws = store.WorkspaceStore(tmp_path, "pipe", {})
    ref = _helper_ref("pipe/stage", None, types.ArtifactTag.DATA, loaders.YAML())

    output_path = ws._resolve_output_path(ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("old cached data")
    os.chmod(output_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    assert not os.access(output_path, os.W_OK)

    result = ws.prepare_output(ref)
    assert result == output_path
    assert not output_path.exists(), "prepare_output should unlink read-only file"


def test_workspace_store_refuses_an_output_resolving_outside_project(
    tmp_path: pathlib.Path,
) -> None:
    workspace_store = store.WorkspaceStore(tmp_path, "pipe", {})
    sentinel = tmp_path.parent / f"{tmp_path.name}-sentinel.txt"
    sentinel.write_text("PRESERVE")
    ref = _helper_ref(
        "pipe/stage",
        str(sentinel),
        types.ArtifactTag.DATA,
        loaders.Text(),
    )

    with pytest.raises(ValueError, match="outside project root"):
        workspace_store.prepare_output(ref)

    assert sentinel.read_text() == "PRESERVE"


def test_resolve_output_path_uses_prefixed_producer(tmp_path: pathlib.Path) -> None:
    ws = store.WorkspaceStore(tmp_path, "ignored", {})
    ref = types.ArtifactRef(
        identity=types.ArtifactIdentity(producer="horizon/train", key="output"),
        format=loaders.CSV(),
        python_type=pathlib.Path,
        tag=types.ArtifactTag.DATA,
    )
    path = ws._resolve_output_path(ref)
    assert path == tmp_path / "data" / "horizon" / "train" / "output.csv"
    assert "ignored" not in str(path)  # pipeline_name should not appear


def test_output_identities_by_path_maps_display_paths_to_identities(
    tmp_path: pathlib.Path,
) -> None:
    ws = store.WorkspaceStore(tmp_path, "pipe", {})
    single = _helper_ref("smoke/total", None, types.ArtifactTag.DATA, loaders.YAML())
    keyed = _helper_ref("smoke/train", "model", types.ArtifactTag.DATA, loaders.CSV())
    metric = _helper_ref("smoke/train", "score", types.ArtifactTag.METRIC, loaders.JSON())

    mapping = ws.output_identities_by_path([single, keyed, metric])

    assert mapping == {
        "data/smoke/total.yaml": single.identity,
        "data/smoke/train/model.csv": keyed.identity,
        "metrics/smoke/train/score.json": metric.identity,
    }


def test_store_spec_cache_uses_the_configured_state_dir(tmp_path: pathlib.Path) -> None:
    """A reconstructed cache store hashes outputs through the configured StateDB."""
    (tmp_path / ".pivot").mkdir()
    (tmp_path / ".pivot" / "config.yaml").write_text("core:\n  state_dir: alternate-state\n")
    spec: store.StoreSpec = {
        "kind": "cache",
        "cache_dir": str(tmp_path / "cache"),
        "project_root": str(tmp_path),
        "pipeline_name": "pipe",
        "input_bindings": {},
        "checkout_modes": [config.CheckoutMode.SYMLINK],
    }

    round_trip = pickle.loads(pickle.dumps(spec))
    rebuilt = store.store_from_spec(round_trip)
    assert isinstance(rebuilt, store.CacheStore)
    ref = _helper_ref("stage", None, types.ArtifactTag.DATA, loaders.Text())
    output = rebuilt.prepare_output(ref)
    output.write_text("state path")
    rebuilt.commit(ref, output)

    assert (tmp_path / "alternate-state" / "state.lmdb").is_dir()


@pytest.mark.parametrize("mode", list(config.CheckoutMode))
def test_workspace_store_refuses_replacing_tracked_file_inside_directory_output(
    tmp_path: pathlib.Path, mode: config.CheckoutMode
) -> None:
    """Writing, restoring or committing a directory output leaves a tracked child and its
    sidecar where they are, in every checkout mode."""
    output_ref = _helper_ref(
        "b/tree",
        None,
        types.ArtifactTag.DIRECTORY,
        cast("loaders.Reader[object]", loaders.PathOnly()),
    )
    workspace_store = store.WorkspaceStore(
        project_root=tmp_path,
        pipeline_name="b",
        input_bindings={},
        cache_dir=tmp_path / "cache" / "files",
        checkout_modes=[mode],
    )
    output_path = workspace_store.prepare_output(output_ref)
    (output_path / "generated.txt").write_text("generated")
    output_hash = workspace_store.commit(output_ref, output_path)
    cache.remove_output(output_path)
    tracked_path = output_path / "saved.txt"
    tracked_path.parent.mkdir(parents=True)
    tracked_path.write_text("user input")
    sidecar_path = track.get_pvt_path(tracked_path)
    track.write_pvt_file(
        sidecar_path,
        track.PvtData(path="saved.txt", hash="abc1230000000000", size=10),
    )

    with pytest.raises(exceptions.PivotError, match="user-owned"):
        workspace_store.restore_from_cache(output_ref, output_hash)

    assert tracked_path.read_text() == "user input"
    assert sidecar_path.exists()

    with pytest.raises(exceptions.PivotError, match="user-owned"):
        workspace_store.prepare_output(output_ref)

    assert tracked_path.read_text() == "user input"
    assert sidecar_path.exists()

    # `pivot commit` of an output that holds a tracked file links it into the cache: refused
    with pytest.raises(exceptions.PivotError, match="user-owned"):
        workspace_store.commit(output_ref, output_path)

    assert not output_path.is_symlink()
    assert tracked_path.read_text() == "user input"
    assert sidecar_path.exists()
