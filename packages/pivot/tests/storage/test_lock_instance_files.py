"""``lock.instance_files``: the files an instance of a mapped stage keeps beside its lock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.storage import lock

if TYPE_CHECKING:
    import pathlib


def test_each_instance_has_its_lock_listings_and_restart_proof_the_lock_last(
    tmp_path: pathlib.Path,
) -> None:
    stages_dir = tmp_path / "stages"
    (stages_dir / "p").mkdir(parents=True)
    names = [
        "cut@a.lock",
        "cut@a=.listing",
        "cut@a=parts.listing",
        "cut@a.incremental",
        "cut@b=.listing",  # a listing whose lock a removal cut short already removed
        "cut.lock",  # the mapped stage's own name: no instance
        "cut@x@a.lock",  # an instance of the mapped variant p/cut@x, not of p/cut
        "cut@bad key.lock",  # no key the variant-name rule allows
        "cutter@a.lock",  # another stage's instance
        "cut@a.lock.tmp",  # no file Pivot names after a stage
    ]
    for name in names:
        (stages_dir / "p" / name).write_text("")
    (stages_dir / "p" / "cut@d.lock").mkdir()  # a directory is no lock

    found = lock.instance_files([stages_dir], "p/cut")

    assert sorted(found) == ["p/cut@a", "p/cut@b"], found
    assert [path.name for path in found["p/cut@a"]][-1] == "cut@a.lock"
    assert sorted(path.name for path in found["p/cut@a"]) == [
        "cut@a.incremental",
        "cut@a.lock",
        "cut@a=.listing",
        "cut@a=parts.listing",
    ]
    assert [path.name for path in found["p/cut@b"]] == ["cut@b=.listing"]
    assert sorted(lock.instance_files([stages_dir], "p/cut@x")) == ["p/cut@x@a"]


def test_both_stages_directories_count_and_every_lock_comes_last(tmp_path: pathlib.Path) -> None:
    """A project of the 0.5.2 layout keeps locks in two stages directories."""
    configured = tmp_path / "state" / "stages"
    legacy = tmp_path / ".pivot" / "stages"
    for stages_dir, names in ((configured, ["cut@a.lock"]), (legacy, ["cut@a=.listing"])):
        (stages_dir / "p").mkdir(parents=True)
        for name in names:
            (stages_dir / "p" / name).write_text("")

    found = lock.instance_files([configured, legacy], "p/cut")

    assert found == {
        "p/cut@a": [legacy / "p" / "cut@a=.listing", configured / "p" / "cut@a.lock"]
    }, found


def test_a_stages_directory_that_does_not_exist_holds_no_instance(tmp_path: pathlib.Path) -> None:
    assert lock.instance_files([tmp_path / "stages"], "p/cut") == {}


def test_remove_stage_files_removes_each_file_and_tolerates_one_already_gone(
    tmp_path: pathlib.Path,
) -> None:
    stages_dir = tmp_path / "stages"
    (stages_dir / "p").mkdir(parents=True)
    for name in ("cut@a=.listing", "cut@a.lock"):
        (stages_dir / "p" / name).write_text("")
    files = lock.instance_files([stages_dir], "p/cut")["p/cut@a"]
    (stages_dir / "p" / "cut@a=.listing").unlink()

    lock.remove_stage_files(files)

    assert list((stages_dir / "p").iterdir()) == []
