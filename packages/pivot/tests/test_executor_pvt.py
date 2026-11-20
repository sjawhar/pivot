"""Execution with tracked (.pvt) file and directory inputs."""

from __future__ import annotations

import os
import pathlib  # noqa: TC003 - compose resolves postponed stage annotations at runtime
import pickle
import shutil
from typing import TYPE_CHECKING

import pytest
import xxhash

from pivot import cli, exceptions, executor, project
from pivot.executor import core as executor_core
from pivot.storage import cache, ownership, track
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

    import click.testing

    from pivot.registry import PipelineLike as Pipeline


@pytest.fixture
def pipeline_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Set a real project root for CLI track/checkout plus direct executor calls."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pivot").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(project, "get_project_root", lambda: tmp_path)
    return tmp_path


def _process_data(data: pathlib.Path) -> str:
    return f"processed: {len(data.read_text())} bytes"


def _process_simple(data: pathlib.Path) -> str:
    _ = data
    return "done"


def _process_data_content(data: pathlib.Path) -> str:
    return f"processed: {data.read_text()}"


def _count_images(images: pathlib.Path) -> str:
    return str(len(list(images.iterdir())))


def _process_images(images: pathlib.Path) -> str:
    _ = images
    return "done"


def _process_mixed(data: pathlib.Path, config: pathlib.Path) -> str:
    return f"{data.read_text()}|{config.read_text()}"


def _process_uppercase(data: pathlib.Path) -> str:
    return data.read_text().upper()


def _output(stage: str) -> pathlib.Path:
    return helpers.get_output_path(stage)


def _register_data_stage(func: Callable[..., object], *, name: str = "process") -> None:
    helpers.register_test_stage(func, name=name, deps={"data": "data.csv"})


def test_run_succeeds_with_existing_tracked_file(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """Pipeline runs successfully when a tracked input exists."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("col1,col2\n1,2\n")
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt", {"path": "data.csv", "hash": "placeholder", "size": 100}
    )

    _register_data_stage(_process_data)
    results = executor.run(pipeline=test_pipeline)

    assert results["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: 14 bytes"


def test_run_fails_when_tracked_file_missing(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """Pipeline fails with a helpful error if a tracked input is absent."""
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt", {"path": "data.csv", "hash": "abc123", "size": 100}
    )
    _register_data_stage(_process_simple)

    with pytest.raises(exceptions.TrackedFileMissingError, match=r"checkout|restore|missing"):
        executor.run(pipeline=test_pipeline)


def test_run_succeeds_with_hash_mismatch(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """A tracked input hash mismatch warns but still permits execution."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original content")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.write_text("modified content that changes the hash")

    _register_data_stage(_process_simple)
    results = executor.run(pipeline=test_pipeline)

    assert results["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "done"


def test_tracked_file_change_triggers_downstream_rerun(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Re-tracking a changed file re-executes its dependent stage."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("original")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    _register_data_stage(_process_data_content)

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: original"

    data_file.write_text("modified")
    assert runner.invoke(cli.cli, ["track", "--force", "data.csv"]).exit_code == 0

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "processed: modified"


def test_unchanged_tracked_file_allows_skip(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """An unchanged tracked input permits a cache skip."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("content")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    _register_data_stage(_process_data_content)

    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "cached"


def test_tracked_directory_change_triggers_rerun(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """Re-tracking an added directory file re-executes the dependent stage."""
    data_dir = pipeline_dir / "images"
    data_dir.mkdir()
    (data_dir / "a.jpg").write_bytes(b"image1")
    (data_dir / "b.jpg").write_bytes(b"image2")
    assert runner.invoke(cli.cli, ["track", "images"]).exit_code == 0
    helpers.register_test_stage(_count_images, name="count_images", deps={"images": "images"})

    assert executor.run(pipeline=test_pipeline)["test/count_images"]["status"] == "ran"
    assert _output("count_images").read_text() == "2"

    (data_dir / "c.jpg").write_bytes(b"image3")
    assert runner.invoke(cli.cli, ["track", "--force", "images"]).exit_code == 0

    assert executor.run(pipeline=test_pipeline)["test/count_images"]["status"] == "ran"
    assert _output("count_images").read_text() == "3"


def test_run_fails_when_tracked_directory_missing(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """A missing tracked directory, recorded by its .pvt and .pvt.listing, raises the
    tracked-file restore error."""
    listing_data = b"0123456789abcdef 500 - a.jpg\nfedcba9876543210 500 - b.jpg\n"
    (pipeline_dir / "images.pvt.listing").write_bytes(listing_data)
    track.write_pvt_file(
        pipeline_dir / "images.pvt",
        {
            "path": "images",
            "hash": xxhash.xxh64(listing_data).hexdigest(),
            "size": 1000,
            "nfiles": 2,
        },
    )
    helpers.register_test_stage(_process_images, name="process", deps={"images": "images"})

    with pytest.raises(exceptions.TrackedFileMissingError, match=r"checkout|restore|missing"):
        executor.run(pipeline=test_pipeline)


def test_mixed_tracked_and_regular_dependencies(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """A stage can depend on both a tracked and regular file."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("tracked data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    config_file = pipeline_dir / "config.txt"
    config_file.write_text("setting=1")

    helpers.register_test_stage(
        _process_mixed,
        name="process",
        deps={"data": "data.csv", "config": "config.txt"},
    )
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"

    config_file.write_text("setting=2")
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"


def test_checkout_then_run_succeeds(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """checkout restores a tracked input that a compositional stage then consumes."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("important data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.unlink()
    assert runner.invoke(cli.cli, ["checkout", "data.csv"]).exit_code == 0
    assert data_file.exists()

    _register_data_stage(_process_uppercase)
    assert executor.run(pipeline=test_pipeline)["test/process"]["status"] == "ran"
    assert _output("process").read_text() == "IMPORTANT DATA"


def test_checkout_missing_restores_tracked_input_before_running(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """run(checkout_missing=True) restores a deleted tracked input from cache, then uses it."""
    data_file = pipeline_dir / "data.csv"
    data_file.write_text("cached data")
    assert runner.invoke(cli.cli, ["track", "data.csv"]).exit_code == 0
    data_file.unlink()
    _register_data_stage(_process_uppercase)

    results = executor.run(pipeline=test_pipeline, checkout_missing=True)

    assert results["test/process"]["status"] == "ran"
    assert data_file.read_text() == "cached data", "the tracked input is back in the workspace"
    assert _output("process").read_text() == "CACHED DATA"


def test_checkout_missing_restores_tracked_directory_from_its_listing_file(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """run(checkout_missing=True) restores a deleted tracked directory from the cache, taking
    its members from the committed .pvt.listing when the cache holds no listing object."""
    data_dir = pipeline_dir / "images"
    data_dir.mkdir()
    (data_dir / "a.jpg").write_bytes(b"image1")
    (data_dir / "b.jpg").write_bytes(b"image2")
    assert runner.invoke(cli.cli, ["track", "images"]).exit_code == 0
    pvt_data = track.read_pvt_file(pipeline_dir / "images.pvt")
    assert pvt_data is not None
    cache_dir = pipeline_dir / ".pivot" / "cache" / "files"
    cache.get_cache_path(cache_dir, pvt_data["hash"]).unlink()
    shutil.rmtree(data_dir)
    helpers.register_test_stage(_count_images, name="count_images", deps={"images": "images"})

    results = executor.run(pipeline=test_pipeline, checkout_missing=True)

    assert results["test/count_images"]["status"] == "ran"
    assert (data_dir / "a.jpg").read_bytes() == b"image1", "a.jpg is back in the workspace"
    assert (data_dir / "b.jpg").read_bytes() == b"image2", "b.jpg is back in the workspace"
    assert _output("count_images").read_text() == "2"


def test_checkout_missing_keeps_a_tracked_file_that_lands_during_directory_restore(
    test_pipeline: Pipeline,
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user file and sidecar copied into a missing tracked directory while its cache restore
    stages its replacement are refused and left in place."""
    data_dir = pipeline_dir / "data"
    data_dir.mkdir()
    (data_dir / "cached.txt").write_text("cached data")
    assert runner.invoke(cli.cli, ["track", "data"]).exit_code == 0
    cache.remove_output(data_dir)

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
            data_dir.mkdir()
            user_file = data_dir / "user.txt"
            user_file.write_text("user bytes")
            track.write_pvt_file(
                track.get_pvt_path(user_file),
                track.PvtData(path="user.txt", hash="0123456789abcdef", size=10),
            )
            landed = True
        return staged

    monkeypatch.setattr(cache, "_checkout_with_fallback", stage_then_copy_user_file)

    with pytest.raises(exceptions.OutputOwnershipError, match="user.txt.pvt"):
        executor_core.verify_tracked_files(pipeline_dir, checkout_missing=True)

    assert landed
    assert (data_dir / "user.txt").read_text() == "user bytes"
    assert track.get_pvt_path(data_dir / "user.txt").is_file()


def test_checkout_missing_reports_tracked_input_absent_from_cache(
    test_pipeline: Pipeline, pipeline_dir: pathlib.Path
) -> None:
    """A missing tracked input that is not cached fails, suggesting a pull, and nothing runs."""
    track.write_pvt_file(
        pipeline_dir / "data.csv.pvt",
        {"path": "data.csv", "hash": "0123456789abcdef", "size": 10},
    )
    _register_data_stage(_process_simple)

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor.run(pipeline=test_pipeline, checkout_missing=True)

    assert "not in local cache" in str(exc_info.value)
    assert exc_info.value.get_suggestion() == "Run 'pivot pull' to fetch from remote storage"
    assert not _output("process").exists()


_NOT_ON_DISK = "These files are tracked by Pivot but don't exist on disk."


def _left_missing(
    *paths: pathlib.Path, checkout_attempted: bool = False
) -> ownership.MissingTracked:
    """``paths`` left missing by the pre-flight in leave-out mode, none of them by a failed
    restore."""
    return ownership.MissingTracked(
        paths=sorted(str(path) for path in paths),
        failed=frozenset(),
        checkout_attempted=checkout_attempted,
    )


@pytest.mark.parametrize(
    "outputs",
    [
        pytest.param({}, id="no_outputs"),
        pytest.param({"data/result.csv": "test/result"}, id="an_output_elsewhere"),
    ],
)
def test_tolerate_missing_returns_a_missing_file_that_overlaps_no_output(
    pipeline_dir: pathlib.Path, outputs: dict[str, str]
) -> None:
    """With tolerate_missing, a missing tracked file that overlaps no output is returned, not
    raised."""
    data_file = pipeline_dir / "data.csv"
    helpers.track_without_data(data_file)

    missing = executor_core.verify_tracked_files(
        pipeline_dir,
        tolerate_missing=True,
        outputs={
            ownership.normalized(pipeline_dir / path): stage for path, stage in outputs.items()
        },
    )

    assert missing == _left_missing(data_file)


@pytest.mark.parametrize(
    ("tracked", "is_directory", "output", "stage", "description"),
    [
        pytest.param(
            "data/b/out.csv",
            False,
            "data/b/out.csv",
            "b/out",
            "it is output 'data/b/out.csv' of stage 'b/out'",
            id="is",
        ),
        pytest.param(
            "data/b/tree/x.csv",
            False,
            "data/b/tree",
            "b/tree",
            "it is inside output 'data/b/tree' of stage 'b/tree'",
            id="is_inside",
        ),
        pytest.param(
            "data/b",
            True,
            "data/b/tree/out.csv",
            "b/tree",
            "it contains output 'data/b/tree/out.csv' of stage 'b/tree'",
            id="contains",
        ),
    ],
)
def test_tolerate_missing_raises_for_only_the_missing_file_that_overlaps_an_output(
    pipeline_dir: pathlib.Path,
    runner: click.testing.CliRunner,
    tracked: str,
    is_directory: bool,
    output: str,
    stage: str,
    description: str,
) -> None:
    """A missing tracked path that is, lies inside or contains an output stops the run even with
    tolerate_missing. The error names that path alone, with the output and its stage."""
    tracked_path = pipeline_dir / tracked
    if is_directory:
        tracked_path.mkdir(parents=True)
        (tracked_path / "member.csv").write_text("tracked member")
    else:
        tracked_path.parent.mkdir(parents=True)
        tracked_path.write_text("tracked data")
    helpers.track_then_remove(runner, pipeline_dir, tracked)
    helpers.track_without_data(pipeline_dir / "elsewhere.csv")

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor_core.verify_tracked_files(
            pipeline_dir,
            tolerate_missing=True,
            outputs={ownership.normalized(pipeline_dir / output): stage},
        )

    assert str(exc_info.value) == (
        "The following tracked files are missing:\n"
        + f"  - {tracked_path}: {description}\n\n"
        + f"{_NOT_ON_DISK}\n{helpers.OVERLAP_STOPS}"
    )


@pytest.mark.parametrize("through", ["output", "tracked_path"])
def test_tolerate_missing_raises_for_a_missing_file_that_reaches_an_output_through_a_symlink(
    pipeline_dir: pathlib.Path, through: str
) -> None:
    """A missing tracked path that lies inside an output only once a symlink is resolved stops
    the run too, naming what both resolve to: the output's directory ``data/b`` is a symlink to
    ``store`` and the tracked path is under ``store/tree``, or the tracked path runs through a
    symlink ``data/raw/link`` to the output's own directory (a ``.pvt`` whose ``path:`` names
    ``link/f.txt``)."""
    real_root = pathlib.Path(os.path.realpath(pipeline_dir))
    if through == "output":
        (pipeline_dir / "store" / "tree").mkdir(parents=True)
        (pipeline_dir / "data").mkdir()
        (pipeline_dir / "data" / "b").symlink_to(pipeline_dir / "store")
        tracked_path = pipeline_dir / "store" / "tree" / "f.txt"
        helpers.track_without_data(tracked_path)
        real_output = real_root / "store" / "tree"
    else:
        (pipeline_dir / "data" / "b" / "tree").mkdir(parents=True)
        (pipeline_dir / "data" / "raw").mkdir()
        (pipeline_dir / "data" / "raw" / "link").symlink_to(pipeline_dir / "data" / "b" / "tree")
        tracked_path = pipeline_dir / "data" / "raw" / "link" / "f.txt"
        track.write_pvt_file(
            pipeline_dir / "data" / "raw" / "f.txt.pvt",
            track.PvtData(path="link/f.txt", hash="0123456789abcdef", size=10),
        )
        real_output = real_root / "data" / "b" / "tree"

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor_core.verify_tracked_files(
            pipeline_dir,
            tolerate_missing=True,
            outputs={ownership.normalized(pipeline_dir / "data" / "b" / "tree"): "b/tree"},
        )

    assert str(exc_info.value) == (
        "The following tracked files are missing:\n"
        + f"  - {tracked_path}: it is inside output 'data/b/tree' of stage 'b/tree' (the path "
        + f"resolves to '{real_output / 'f.txt'}'; the output resolves to '{real_output}')\n\n"
        + f"{_NOT_ON_DISK}\n{helpers.OVERLAP_STOPS}"
    )


def test_tolerate_missing_compares_path_parts_not_string_prefixes(
    pipeline_dir: pathlib.Path,
) -> None:
    """``data-backup/x`` does not lie inside the directory output ``data``, although its path
    string starts with that output's path string."""
    missing_file = pipeline_dir / "data-backup" / "x"
    helpers.track_without_data(missing_file)

    missing = executor_core.verify_tracked_files(
        pipeline_dir,
        tolerate_missing=True,
        outputs={ownership.normalized(pipeline_dir / "data"): "test/data"},
    )

    assert missing == _left_missing(missing_file)


def test_tolerate_missing_names_the_first_contained_output_in_sorted_path_order(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """A missing tracked directory that contains several outputs is described by the first of
    them in path-part order (``a/out.csv`` before ``a-1.csv``, which a string sort puts first),
    whatever order the outputs come in."""
    tree = pipeline_dir / "data" / "b"
    tree.mkdir(parents=True)
    (tree / "member.csv").write_text("tracked member")
    helpers.track_then_remove(runner, pipeline_dir, "data/b")

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor_core.verify_tracked_files(
            pipeline_dir,
            tolerate_missing=True,
            outputs={
                ownership.normalized(tree / "z.csv"): "b/z",
                ownership.normalized(tree / "a-1.csv"): "b/a-1",
                ownership.normalized(tree / "a" / "out.csv"): "b/a",
            },
        )

    assert str(exc_info.value) == (
        "The following tracked files are missing:\n"
        + f"  - {tree}: it contains output 'data/b/a/out.csv' of stage 'b/a'\n\n"
        + f"{_NOT_ON_DISK}\n{helpers.OVERLAP_STOPS}"
    )


def test_tolerate_missing_restores_a_cached_file_and_returns_the_uncached_one(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """With checkout_missing, the pre-flight restores a cached tracked file first, so only the
    file the cache does not hold is returned."""
    cached = pipeline_dir / "cached.csv"
    cached.write_text("cached data")
    helpers.track_then_remove(runner, pipeline_dir, "cached.csv")
    uncached = pipeline_dir / "uncached.csv"
    helpers.track_without_data(uncached)

    missing = executor_core.verify_tracked_files(
        pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
    )

    assert missing == _left_missing(uncached, checkout_attempted=True)
    assert cached.read_text() == "cached data", "the cached file is back in the workspace"


def test_tolerate_missing_without_outputs_is_refused_before_anything_is_read(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """tolerate_missing with no outputs is a caller's error, not a check against no output: it
    raises ValueError before it restores anything."""
    cached = pipeline_dir / "cached.csv"
    cached.write_text("cached data")
    helpers.track_then_remove(runner, pipeline_dir, "cached.csv")

    with pytest.raises(ValueError, match="tolerate_missing needs outputs"):
        executor_core.verify_tracked_files(
            pipeline_dir, checkout_missing=True, tolerate_missing=True
        )

    assert not cached.exists(), "nothing is restored"


def test_without_tolerate_missing_every_missing_file_raises_with_the_same_text(
    pipeline_dir: pathlib.Path,
) -> None:
    """Without tolerate_missing, every missing tracked file raises, with the 0.10.4 text, even
    when one of them lies inside a given output."""
    first = pipeline_dir / "data.csv"
    second = pipeline_dir / "raw" / "labels.csv"
    helpers.track_without_data(first)
    helpers.track_without_data(second)

    with pytest.raises(exceptions.TrackedFileMissingError) as exc_info:
        executor_core.verify_tracked_files(
            pipeline_dir, outputs={ownership.normalized(pipeline_dir / "raw"): "test/raw"}
        )

    assert str(exc_info.value) == (
        f"The following tracked files are missing:\n  - {first}\n  - {second}\n\n{_NOT_ON_DISK}"
    )


def test_tracked_file_missing_error_keeps_its_overlaps_through_pickling() -> None:
    """The error crosses a process boundary with its files, its checkout flag, each file's
    overlap description and the files whose restore failed."""
    path = "/project/data/b/tree/x.csv"
    failed = "/project/data/b/tree/y.csv"
    description = "it is inside output 'data/b/tree' of stage 'b/tree'"
    original = exceptions.TrackedFileMissingError(
        [path, failed],
        checkout_attempted=True,
        overlaps={path: description, failed: description},
        restore_failed=[failed],
    )

    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, exceptions.TrackedFileMissingError)
    assert restored.__reduce__() == (
        exceptions.TrackedFileMissingError,
        ([path, failed], True, {path: description, failed: description}, [failed]),
    )
    assert str(restored) == (
        "The following tracked files are missing:\n"
        + f"  - {path}: {description}\n"
        + f"  - {failed}: {description}; its restore failed\n\n"
        + "A 'Cannot restore' warning above names the cause of each restore that failed. "
        + f"The other files are not in local cache.\n{helpers.OVERLAP_STOPS}"
    )
    assert restored.get_suggestion() == "Run 'pivot pull' to fetch from remote storage"


def test_tracked_file_missing_error_names_as_failed_only_the_files_it_lists() -> None:
    """A failed restore of a file the error does not list (one that overlaps no output) changes
    nothing in it: the file it lists, which the local cache lacks, is named as such, with the
    pull suggestion, and the error keeps no failed file."""
    path = "/project/data/b/tree/x.csv"
    description = "it is inside output 'data/b/tree' of stage 'b/tree'"

    error = exceptions.TrackedFileMissingError(
        [path],
        checkout_attempted=True,
        overlaps={path: description},
        restore_failed=["/project/elsewhere.csv"],
    )

    assert error.__reduce__() == (
        exceptions.TrackedFileMissingError,
        ([path], True, {path: description}, []),
    )
    assert str(error) == (
        f"The following tracked files are missing:\n  - {path}: {description}\n\n"
        + f"These files are not in local cache.\n{helpers.OVERLAP_STOPS}"
    )
    assert error.get_suggestion() == "Run 'pivot pull' to fetch from remote storage"


def test_find_missing_tracked_files_says_what_the_cache_restores_and_restores_nothing(
    pipeline_dir: pathlib.Path, runner: click.testing.CliRunner
) -> None:
    """The dry run's read-only view of the pre-flight: every missing tracked path, and with
    checkout_missing those the local cache restores (a file whose object it holds, a directory
    whose listing and member objects it holds), apart from the others, with their .pvt data and
    every path their restore creates inside them. It restores nothing, and the pre-flight's own
    restore then agrees with it."""
    cache_dir = pipeline_dir / ".pivot" / "cache" / "files"
    cached_file = pipeline_dir / "cached.csv"
    cached_file.write_text("cached data")
    cached_dir = pipeline_dir / "images"
    (cached_dir / "sub" / "deep").mkdir(parents=True)
    (cached_dir / "a.jpg").write_bytes(b"image1")
    (cached_dir / "b.jpg").write_bytes(b"image2")
    (cached_dir / "sub" / "deep" / "c.jpg").write_bytes(b"image3")
    (cached_dir / "empty").mkdir()
    broken_dir = pipeline_dir / "scans"
    broken_dir.mkdir()
    (broken_dir / "c.jpg").write_bytes(b"scan1")
    (broken_dir / "d.jpg").write_bytes(b"scan2")
    for name in ("cached.csv", "images", "scans"):
        helpers.track_then_remove(runner, pipeline_dir, name)
    broken_pvt = track.read_pvt_file(track.get_pvt_path(broken_dir))
    assert broken_pvt is not None
    lost = track.tracked_members(broken_dir, broken_pvt, cache_dir)["files"][0]["hash"]
    cache.get_cache_path(cache_dir, lost).unlink()
    uncached_file = pipeline_dir / "uncached.csv"
    helpers.track_without_data(uncached_file)
    missing = sorted(str(path) for path in (cached_file, cached_dir, broken_dir, uncached_file))
    restorable = dict[str, track.PvtData]()
    for path in (cached_file, cached_dir):
        record = track.read_pvt_file(track.get_pvt_path(path))
        assert record is not None
        restorable[str(path)] = record

    found = executor_core.find_missing_tracked_files(pipeline_dir, checkout_missing=True)

    assert found == executor_core.MissingTrackedFiles(
        missing=_left_missing(broken_dir, uncached_file, checkout_attempted=True),
        restorable=restorable,
        restored={
            str(cached_file): set(),
            str(cached_dir): {
                ("a.jpg",),
                ("b.jpg",),
                ("sub",),
                ("sub", "deep"),
                ("sub", "deep", "c.jpg"),
                ("empty",),
            },
        },
    )
    assert list(found.restorable) == [str(cached_file), str(cached_dir)]
    assert [path for path in missing if pathlib.Path(path).exists()] == [], "nothing restored"
    assert executor_core.find_missing_tracked_files(pipeline_dir) == (
        executor_core.MissingTrackedFiles(
            missing=ownership.MissingTracked(
                paths=missing, failed=frozenset(), checkout_attempted=False
            ),
            restorable={},
            restored={},
        )
    )
    assert (
        executor_core.verify_tracked_files(
            pipeline_dir, checkout_missing=True, tolerate_missing=True, outputs={}
        )
        == found.missing
    ), "the restore leaves the same files out"
    assert (cached_dir / "empty").is_dir(), "the restore creates the empty directory it lists"
