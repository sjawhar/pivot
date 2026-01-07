"""Tests for .pvt file handling."""

import pathlib

import pytest

from pivot import pvt

# write_pvt_file / read_pvt_file


def test_write_read_pvt_file_roundtrip(tmp_path: pathlib.Path) -> None:
    """PvtData can be written and read back identically."""
    pvt_path = tmp_path / "data.csv.pvt"
    data: pvt.PvtData = {
        "path": "data.csv",
        "hash": "abc123def456",
        "size": 1024,
    }

    pvt.write_pvt_file(pvt_path, data)
    result = pvt.read_pvt_file(pvt_path)

    assert result == data


def test_write_pvt_file_creates_parent_directories(tmp_path: pathlib.Path) -> None:
    """write_pvt_file creates parent directories if needed."""
    pvt_path = tmp_path / "nested" / "deep" / "data.csv.pvt"
    data: pvt.PvtData = {"path": "data.csv", "hash": "abc123", "size": 100}

    pvt.write_pvt_file(pvt_path, data)

    assert pvt_path.exists()
    assert pvt.read_pvt_file(pvt_path) == data


def test_write_pvt_file_atomic_no_partial_on_error(tmp_path: pathlib.Path) -> None:
    """Write failure does not leave partial .pvt file."""
    pvt_path = tmp_path / "atomic.pvt"
    # Write initial valid data
    pvt.write_pvt_file(pvt_path, {"path": "x", "hash": "y", "size": 1})

    # Verify no .tmp files remain
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert len(tmp_files) == 0, f"Temporary files remain: {tmp_files}"


def test_pvt_file_for_directory_with_manifest(tmp_path: pathlib.Path) -> None:
    """Directory .pvt files include manifest and num_files."""
    pvt_path = tmp_path / "images.pvt"
    data: pvt.PvtData = {
        "path": "images",
        "hash": "tree_hash_123",
        "size": 5120,
        "num_files": 2,
        "manifest": [
            {"relpath": "cat.jpg", "hash": "aaa111", "size": 2048},
            {"relpath": "dog.jpg", "hash": "bbb222", "size": 3072, "isexec": False},
        ],
    }

    pvt.write_pvt_file(pvt_path, data)
    result = pvt.read_pvt_file(pvt_path)

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

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_invalid_yaml_returns_none(tmp_path: pathlib.Path) -> None:
    """Invalid YAML syntax returns None."""
    pvt_path = tmp_path / "invalid.pvt"
    pvt_path.write_text("path: [unclosed bracket\n")

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_non_dict_returns_none(tmp_path: pathlib.Path) -> None:
    """Non-dict YAML (list, string) returns None."""
    pvt_path = tmp_path / "list.pvt"
    pvt_path.write_text("- item1\n- item2\n")

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_binary_garbage_returns_none(tmp_path: pathlib.Path) -> None:
    """Binary garbage returns None."""
    pvt_path = tmp_path / "binary.pvt"
    pvt_path.write_bytes(b"\xff\xfe\x00\x01\x80\x81")

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_empty_returns_none(tmp_path: pathlib.Path) -> None:
    """Empty file returns None."""
    pvt_path = tmp_path / "empty.pvt"
    pvt_path.write_text("")

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


def test_read_pvt_file_missing_required_fields_returns_none(
    tmp_path: pathlib.Path,
) -> None:
    """Missing required fields (path, hash, size) returns None."""
    pvt_path = tmp_path / "incomplete.pvt"
    pvt_path.write_text("path: data.csv\n")  # Missing hash and size

    result = pvt.read_pvt_file(pvt_path)

    assert result is None


# get_pvt_path


def test_get_pvt_path_for_file() -> None:
    """File path gets .pvt suffix appended."""
    data_path = pathlib.Path("data/train.csv")

    pvt_path = pvt.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/train.csv.pvt")


def test_get_pvt_path_for_directory() -> None:
    """Directory path gets .pvt suffix (no trailing slash)."""
    data_path = pathlib.Path("data/images")

    pvt_path = pvt.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/images.pvt")


def test_get_pvt_path_for_directory_with_trailing_slash() -> None:
    """Directory with trailing slash handled correctly."""
    # pathlib normalizes this, but test explicitly
    data_path = pathlib.Path("data/images/")

    pvt_path = pvt.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("data/images.pvt")


def test_get_pvt_path_preserves_parent_directories() -> None:
    """Parent directories preserved in .pvt path."""
    data_path = pathlib.Path("project/data/nested/file.csv")

    pvt_path = pvt.get_pvt_path(data_path)

    assert pvt_path == pathlib.Path("project/data/nested/file.csv.pvt")


# get_data_path


def test_get_data_path_from_file_pvt() -> None:
    """Get data path from file.csv.pvt."""
    pvt_path = pathlib.Path("data/train.csv.pvt")

    data_path = pvt.get_data_path(pvt_path)

    assert data_path == pathlib.Path("data/train.csv")


def test_get_data_path_from_directory_pvt() -> None:
    """Get data path from directory.pvt."""
    pvt_path = pathlib.Path("data/images.pvt")

    data_path = pvt.get_data_path(pvt_path)

    assert data_path == pathlib.Path("data/images")


def test_get_data_path_invalid_suffix_raises() -> None:
    """Non-.pvt file raises ValueError."""
    not_pvt = pathlib.Path("data/file.txt")

    with pytest.raises(ValueError, match=r"\.pvt"):
        pvt.get_data_path(not_pvt)


# discover_pvt_files


def test_discover_pvt_files_finds_files(tmp_path: pathlib.Path) -> None:
    """Discovers .pvt files in directory tree."""
    # Create structure:
    # tmp_path/
    #   data.csv.pvt
    #   subdir/
    #     model.pkl.pvt
    pvt.write_pvt_file(
        tmp_path / "data.csv.pvt",
        {"path": "data.csv", "hash": "hash1", "size": 100},
    )
    (tmp_path / "subdir").mkdir()
    pvt.write_pvt_file(
        tmp_path / "subdir" / "model.pkl.pvt",
        {"path": "model.pkl", "hash": "hash2", "size": 200},
    )

    result = pvt.discover_pvt_files(tmp_path)

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
    result = pvt.discover_pvt_files(tmp_path)

    assert result == {}


def test_discover_pvt_files_skips_invalid(tmp_path: pathlib.Path) -> None:
    """Invalid .pvt files are skipped."""
    # Valid file
    pvt.write_pvt_file(
        tmp_path / "valid.csv.pvt",
        {"path": "valid.csv", "hash": "hash1", "size": 100},
    )
    # Invalid file
    (tmp_path / "invalid.csv.pvt").write_text("not: valid: yaml: [")

    result = pvt.discover_pvt_files(tmp_path)

    assert len(result) == 1
    assert str(tmp_path / "valid.csv") in result


def test_discover_pvt_files_with_directory_manifest(tmp_path: pathlib.Path) -> None:
    """Directory .pvt files with manifest are discovered correctly."""
    pvt.write_pvt_file(
        tmp_path / "images.pvt",
        {
            "path": "images",
            "hash": "tree_hash",
            "size": 5000,
            "num_files": 3,
            "manifest": [
                {"relpath": "a.jpg", "hash": "h1", "size": 1000},
                {"relpath": "b.jpg", "hash": "h2", "size": 2000},
                {"relpath": "c.jpg", "hash": "h3", "size": 2000},
            ],
        },
    )

    result = pvt.discover_pvt_files(tmp_path)

    assert len(result) == 1
    images_path = str(tmp_path / "images")
    assert images_path in result
    assert result[images_path].get("num_files") == 3
    manifest = result[images_path].get("manifest")
    assert manifest is not None
    assert len(manifest) == 3


# Validation


def test_write_pvt_file_rejects_path_traversal(tmp_path: pathlib.Path) -> None:
    """Path with .. is rejected."""
    pvt_path = tmp_path / "data.pvt"
    data: pvt.PvtData = {
        "path": "../../../etc/passwd",
        "hash": "evil",
        "size": 666,
    }

    with pytest.raises(ValueError, match=r"path traversal|\.\."):
        pvt.write_pvt_file(pvt_path, data)
