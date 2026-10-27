import pathlib

import pytest

from pivot.exceptions import SecurityValidationError
from pivot.storage import track

# write_pvt_file / read_pvt_file


def test_write_read_pvt_file_roundtrip(tmp_path: pathlib.Path) -> None:
    """PvtData can be written and read back identically."""
    pvt_path = tmp_path / "data.csv.pvt"
    data: track.PvtData = {
        "path": "data.csv",
        "hash": "abc123def456",
        "size": 1024,
    }

    track.write_pvt_file(pvt_path, data)
    result = track.read_pvt_file(pvt_path)

    assert result == data


def test_write_pvt_file_creates_parent_directories(tmp_path: pathlib.Path) -> None:
    """write_pvt_file creates parent directories if needed."""
    pvt_path = tmp_path / "nested" / "deep" / "data.csv.pvt"
    data: track.PvtData = {"path": "data.csv", "hash": "abc123", "size": 100}

    track.write_pvt_file(pvt_path, data)

    assert pvt_path.exists()
    assert track.read_pvt_file(pvt_path) == data


def test_write_pvt_file_atomic_no_partial_on_error(tmp_path: pathlib.Path) -> None:
    """Write failure does not leave partial .pvt file."""
    pvt_path = tmp_path / "atomic.pvt"
    # Write initial valid data
    track.write_pvt_file(pvt_path, {"path": "x", "hash": "y", "size": 1})

    # Verify no .tmp files remain
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


def test_pvt_file_for_directory_with_manifest(tmp_path: pathlib.Path) -> None:
    """Directory .pvt files include manifest and num_files."""
    pvt_path = tmp_path / "images.pvt"
    data: track.PvtData = {
        "path": "images",
        "hash": "tree_hash_123",
        "size": 5120,
        "num_files": 2,
        "manifest": [
            {"relpath": "cat.jpg", "hash": "aaa111", "size": 2048, "isexec": False},
            {"relpath": "dog.jpg", "hash": "bbb222", "size": 3072, "isexec": False},
        ],
    }

    track.write_pvt_file(pvt_path, data)
    result = track.read_pvt_file(pvt_path)

    assert result == data
    assert result is not None
    assert result.get("num_files") == 2
    manifest = result.get("manifest")
    assert manifest is not None
    assert len(manifest) == 2


# read_pvt_file error handling


def test_read_pvt_file_missing_returns_none(tmp_path: pathlib.Path) -> None:
    """Reading non-existent .pvt file returns None."""
    pvt_path = tmp_path / "nonexistent.pvt"

    result = track.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_invalid_yaml_returns_none(tmp_path: pathlib.Path) -> None:
    """Invalid YAML syntax returns None."""
    pvt_path = tmp_path / "invalid.pvt"
    pvt_path.write_text("path: [unclosed bracket\n")

    result = track.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_non_dict_returns_none(tmp_path: pathlib.Path) -> None:
    """Non-dict YAML (list, string) returns None."""
    pvt_path = tmp_path / "list.pvt"
    pvt_path.write_text("- item1\n- item2\n")

    result = track.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_binary_garbage_returns_none(tmp_path: pathlib.Path) -> None:
    """Binary garbage returns None."""
    pvt_path = tmp_path / "binary.pvt"
    pvt_path.write_bytes(b"\xff\xfe\x00\x01\x80\x81")

    result = track.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_empty_returns_none(tmp_path: pathlib.Path) -> None:
    """Empty file returns None."""
    pvt_path = tmp_path / "empty.pvt"
    pvt_path.write_text("")

    result = track.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_missing_required_fields_returns_none(
    tmp_path: pathlib.Path,
) -> None:
    """Missing required fields (path, hash, size) returns None."""
    pvt_path = tmp_path / "incomplete.pvt"
    pvt_path.write_text("path: data.csv\n")  # Missing hash and size

    result = track.read_pvt_file(pvt_path)

    assert result is None


# get_pvt_path


def test_get_pvt_path_for_file() -> None:
    """File path gets .pvt suffix appended."""
    data_path = pathlib.Path("data/train.csv")

    pvt_path = track.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/train.csv.pvt")


def test_get_pvt_path_for_directory() -> None:
    """Directory path gets .pvt suffix (no trailing slash)."""
    data_path = pathlib.Path("data/images")

    pvt_path = track.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/images.pvt")


def test_get_pvt_path_for_directory_with_trailing_slash() -> None:
    """Directory with trailing slash handled correctly."""
    # pathlib normalizes this, but test explicitly
    data_path = pathlib.Path("data/images/")

    pvt_path = track.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/images.pvt")


def test_get_pvt_path_preserves_parent_directories() -> None:
    """Parent directories preserved in .pvt path."""
    data_path = pathlib.Path("project/data/nested/file.csv")

    pvt_path = track.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("project/data/nested/file.csv.pvt")


# get_data_path


def test_get_data_path_from_file_pvt() -> None:
    """Get data path from file.csv.pvt."""
    pvt_path = pathlib.Path("data/train.csv.pvt")

    data_path = track.get_data_path(pvt_path)

    assert data_path == pathlib.Path("data/train.csv")


def test_get_data_path_from_directory_pvt() -> None:
    """Get data path from directory.pvt."""
    pvt_path = pathlib.Path("data/images.pvt")

    data_path = track.get_data_path(pvt_path)

    assert data_path == pathlib.Path("data/images")


def test_get_data_path_invalid_suffix_raises() -> None:
    """Non-.pvt file raises ValueError."""
    not_pvt = pathlib.Path("data/file.txt")

    with pytest.raises(ValueError, match=r"\.pvt"):
        track.get_data_path(not_pvt)


# discover_pvt_files


def test_discover_pvt_files_finds_files(tmp_path: pathlib.Path) -> None:
    """Discovers .pvt files in directory tree."""
    # Create structure:
    # tmp_path/
    #   data.csv.pvt
    #   subdir/
    #     model.pkl.pvt
    track.write_pvt_file(
        tmp_path / "data.csv.pvt",
        {"path": "data.csv", "hash": "hash1", "size": 100},
    )
    (tmp_path / "subdir").mkdir()
    track.write_pvt_file(
        tmp_path / "subdir" / "model.pkl.pvt",
        {"path": "model.pkl", "hash": "hash2", "size": 200},
    )

    result = track.discover_pvt_files(tmp_path)

    assert len(result) == 2
    # Keys should be absolute paths to data files
    data_csv_path = str(tmp_path / "data.csv")
    model_pkl_path = str(tmp_path / "subdir" / "model.pkl")
    assert data_csv_path in result
    assert model_pkl_path in result
    assert result[data_csv_path]["hash"] == "hash1"
    assert result[model_pkl_path]["hash"] == "hash2"


def test_discover_pvt_files_empty_directory(tmp_path: pathlib.Path) -> None:
    """Empty directory returns empty dict."""
    result = track.discover_pvt_files(tmp_path)

    assert result == {}


def test_discover_pvt_files_skips_invalid(tmp_path: pathlib.Path) -> None:
    """Invalid .pvt files are skipped."""
    # Valid file
    track.write_pvt_file(
        tmp_path / "valid.csv.pvt",
        {"path": "valid.csv", "hash": "hash1", "size": 100},
    )
    # Invalid file
    (tmp_path / "invalid.csv.pvt").write_text("not: valid: yaml: [")

    result = track.discover_pvt_files(tmp_path)

    assert len(result) == 1
    assert str(tmp_path / "valid.csv") in result


def test_discover_pvt_files_with_directory_manifest(tmp_path: pathlib.Path) -> None:
    """Directory .pvt files with manifest are discovered correctly."""
    track.write_pvt_file(
        tmp_path / "images.pvt",
        {
            "path": "images",
            "hash": "tree_hash",
            "size": 5000,
            "num_files": 3,
            "manifest": [
                {"relpath": "a.jpg", "hash": "h1", "size": 1000, "isexec": False},
                {"relpath": "b.jpg", "hash": "h2", "size": 2000, "isexec": False},
                {"relpath": "c.jpg", "hash": "h3", "size": 2000, "isexec": False},
            ],
        },
    )

    result = track.discover_pvt_files(tmp_path)

    assert len(result) == 1
    images_path = str(tmp_path / "images")
    assert images_path in result
    assert result[images_path].get("num_files") == 3
    manifest = result[images_path].get("manifest")
    assert manifest is not None
    assert len(manifest) == 3


def test_discover_pvt_files_keeps_a_backslash_in_the_data_path(tmp_path: pathlib.Path) -> None:
    """A backslash in a directory or file name stays in the key: it is never a separator."""
    (tmp_path / "dir\\with\\backslash").mkdir()
    track.write_pvt_file(
        tmp_path / "dir\\with\\backslash" / "f.csv.pvt",
        track.PvtData(path="f.csv", hash="hash1", size=1),
    )
    track.write_pvt_file(
        tmp_path / "a\\b.csv.pvt", track.PvtData(path="a\\b.csv", hash="hash2", size=1)
    )

    result = track.discover_pvt_files(tmp_path)

    assert sorted(result) == sorted(
        [str(tmp_path / "dir\\with\\backslash" / "f.csv"), str(tmp_path / "a\\b.csv")]
    )


def test_discover_pvt_files_never_reads_a_backslash_as_a_step_out_of_the_root(
    tmp_path: pathlib.Path,
) -> None:
    """``..\\..\\escaped.csv`` is one name beside its sidecar, not two steps up."""
    (tmp_path / "sub").mkdir()
    track.write_pvt_file(
        tmp_path / "sub" / "x.pvt",
        track.PvtData(path="..\\..\\escaped.csv", hash="hash1", size=1),
    )

    result = track.discover_pvt_files(tmp_path)

    assert list(result) == [str(tmp_path / "sub" / "..\\..\\escaped.csv")]


# Validation


def test_write_pvt_file_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    """Path with .. is rejected."""
    pvt_path = tmp_path / "data.pvt"
    data: track.PvtData = {
        "path": "../../../etc/passwd",
        "hash": "evil",
        "size": 666,
    }

    with pytest.raises(SecurityValidationError, match=r"path traversal|\.\."):
        track.write_pvt_file(pvt_path, data)


# restore_target: a .pvt restores its file only inside the project


def test_restore_target_takes_the_name_pivot_track_writes(tmp_path: pathlib.Path) -> None:
    """The sidecar's own name is the file beside it, with nothing to say about it."""
    (tmp_path / "sub").mkdir()

    target = track.restore_target(tmp_path / "sub" / "data.csv.pvt", "data.csv", tmp_path)

    assert target == track.RestoreTarget(tmp_path / "sub" / "data.csv", None)


@pytest.mark.parametrize("value", ["", ".", "./"])
def test_restore_target_skips_a_pvt_whose_path_names_no_file(
    tmp_path: pathlib.Path, value: str
) -> None:
    """Such a value names the directory holding the .pvt, which a restore would replace."""
    (tmp_path / "sub").mkdir()

    target = track.restore_target(tmp_path / "sub" / "dot.pvt", value, tmp_path)

    assert target.path is None
    assert target.warning == (
        f"Skipping sub/dot.pvt: its path {value!r} names no file; set path: to the name of the "
        "file or directory beside the .pvt."
    )


def test_restore_target_skips_a_sidecar_literally_named_pvt(tmp_path: pathlib.Path) -> None:
    """A file named '.pvt' has own name ''; clause b must run before clause a, or '' matches
    clause a and the .pvt restores its own directory, which checkout --force then replaces."""
    (tmp_path / "sub").mkdir()

    target = track.restore_target(tmp_path / "sub" / ".pvt", "", tmp_path)

    assert target.path is None, target
    assert "names no file" in (target.warning or ""), target


def test_restore_target_skips_a_pvt_whose_file_lies_outside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """The file's directory is resolved: a symlink out of the project takes the file with it."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    target = track.restore_target(root / "x.pvt", "link/x.csv", root)

    assert target.path is None
    assert target.warning == (
        f"Skipping x.pvt: its path 'link/x.csv' resolves to '{outside / 'x.csv'}', outside the "
        f"project root '{root}'. A .pvt restores its file only inside the project; move it "
        "beside its data and set path: to the data's name."
    )


def test_restore_target_skips_a_pvt_whose_path_cannot_be_resolved(tmp_path: pathlib.Path) -> None:
    """Where a path leads is unknown when it cannot be resolved, so it is not restored."""
    target = track.restore_target(tmp_path / "x.pvt", "sub\0/x.csv", tmp_path)

    assert target.path is None
    assert "outside the project root" in (target.warning or "")


def test_restore_target_takes_a_file_under_a_symlink_inside_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """An in-project symlink keeps working, and the reader is told the names disagree."""
    (tmp_path / "data_real").mkdir()
    (tmp_path / "dlink").symlink_to("data_real", target_is_directory=True)

    target = track.restore_target(tmp_path / "y.pvt", "dlink/y.csv", tmp_path)

    assert target.path == tmp_path / "dlink" / "y.csv"
    assert target.warning == (
        "y.pvt: its path 'dlink/y.csv' is not the .pvt's own name 'y'. 'pivot checkout' with no "
        "target, 'pivot pull' and 'pivot repro --checkout-missing' restore 'dlink/y.csv'; "
        "'pivot checkout y.pvt', 'pivot get', 'pivot update' and 'pivot push' use the file named "
        "'y' beside the .pvt. Set path: to that name, the form 'pivot track' writes."
    )


def test_discover_pvt_files_leaves_out_a_pvt_that_leads_out_of_the_project(
    tmp_path: pathlib.Path,
) -> None:
    """Every command reads the discovery, so one skip keeps all of them inside the project."""
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "x.pvt").write_text("path: link/x.csv\nhash: hash1\nsize: 5\n")
    (root / "dot.pvt").write_text("path: .\nhash: hash2\nsize: 5\n")
    track.write_pvt_file(root / "kept.csv.pvt", track.PvtData(path="kept.csv", hash="h", size=1))

    result = track.discover_pvt_files(root)

    assert list(result) == [str(root / "kept.csv")]
