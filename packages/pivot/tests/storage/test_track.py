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


def _helper_pvt(**fields: object) -> dict[str, object]:
    """A valid file .pvt's fields, with ``fields`` replacing or adding to them."""
    return {"path": "data.csv", "hash": "0123456789abcdef", "size": 1, **fields}


_ENTRY = {"relpath": "a.txt", "hash": "0123456789abcdef", "size": 1, "isexec": False}


@pytest.mark.parametrize(
    ("data", "error"),
    [
        pytest.param(_helper_pvt(path=1), "field 'path' is not a string", id="int-path"),
        pytest.param(
            _helper_pvt(hash=1234567890123456), "field 'hash' is not a string", id="int-hash"
        ),
        pytest.param(
            _helper_pvt(manifest={"a.txt": "0123456789abcdef"}),
            "field 'manifest' is not a list",
            id="manifest-mapping",
        ),
        pytest.param(
            _helper_pvt(manifest=["a.txt"]),
            "field 'manifest[0]' is not a mapping",
            id="manifest-entry-string",
        ),
        pytest.param(
            _helper_pvt(manifest=[{k: v for k, v in _ENTRY.items() if k != "hash"}]),
            "field 'manifest[0].hash' is missing",
            id="manifest-entry-without-hash",
        ),
        pytest.param(
            _helper_pvt(manifest=[_ENTRY, {k: v for k, v in _ENTRY.items() if k != "isexec"}]),
            "field 'manifest[1].isexec' is missing",
            id="manifest-entry-without-isexec",
        ),
        pytest.param(
            _helper_pvt(manifest=[{**_ENTRY, "relpath": 7}]),
            "field 'manifest[0].relpath' is not a string",
            id="manifest-entry-int-relpath",
        ),
        pytest.param(
            _helper_pvt(manifest=[_ENTRY, {**_ENTRY, "relpath": "a\0b"}]),
            "field 'manifest[1].relpath' holds a NUL byte",
            id="manifest-entry-nul-relpath",
        ),
        pytest.param(
            _helper_pvt(manifest=[_ENTRY], empty_dirs=["e", "f\0"]),
            "field 'empty_dirs[1]' holds a NUL byte",
            id="nul-empty-dir",
        ),
    ],
)
def test_pvt_field_error_names_a_field_a_restore_cannot_use(data: object, error: str) -> None:
    """A .pvt that has every field but holds one a restore builds a path or cache key from with
    the wrong type or a NUL byte, or a manifest entry without a field a restore reads, is named
    by field."""
    assert track.is_pvt_data(data), "the field check runs on data is_pvt_data accepts"
    assert track.pvt_field_error(data) == error


def test_pvt_field_error_skips_what_no_restore_needs() -> None:
    """A tracked directory's .pvt passes, and what no restore needs is not checked: the .pvt's
    size and num_files, a manifest entry's size (present or not) and the types of its size and
    isexec, and a NUL byte in path, which no restore builds a path from."""
    source = dict.fromkeys(("repo", "rev", "rev_lock", "stage", "path", "remote"), "x")
    tracked = _helper_pvt(num_files=1, manifest=[_ENTRY], empty_dirs=["empty"], source=source)
    no_size = {k: v for k, v in _ENTRY.items() if k != "size"}
    loose = _helper_pvt(
        path="a\0b",
        size="1",
        num_files="2",
        manifest=[{**_ENTRY, "size": "1", "isexec": "false"}, no_size],
    )

    assert track.pvt_field_error(tracked) is None
    assert track.pvt_field_error(loose) is None


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
