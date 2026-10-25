import pathlib

from pivot.storage import track


def test_pvt_import_source_roundtrip(tmp_path: pathlib.Path) -> None:
    """PvtData with source writes/reads back identically."""
    pvt_path = tmp_path / "data.csv.pvt"
    data: track.PvtData = {
        "path": "data.csv",
        "hash": "abc123def456",
        "size": 1024,
        "source": track.ImportSource(
            repo="https://github.com/example/repo",
            rev="main",
            rev_lock="abc123def456789",
            stage="upstream_stage",
            path="outputs/data.csv",
            remote="s3://bucket/remote",
        ),
    }

    track.write_pvt_file(pvt_path, data)
    result = track.read_pvt_file(pvt_path)

    assert result == data, "Source data should round-trip identically"


def test_pvt_backward_compat(tmp_path: pathlib.Path) -> None:
    """is_pvt_data() accepts dicts without source key."""
    pvt_path = tmp_path / "data.csv.pvt"
    data: track.PvtData = {
        "path": "data.csv",
        "hash": "abc123",
        "size": 100,
    }

    track.write_pvt_file(pvt_path, data)
    result = track.read_pvt_file(pvt_path)

    assert result is not None, "Should read old-format PvtData without source"
    assert "source" not in result, "Old format should not have source key"


def test_is_import_true() -> None:
    """is_import() returns True for pvt with source."""
    data: track.PvtData = {
        "path": "data.csv",
        "hash": "abc123",
        "size": 100,
        "source": track.ImportSource(
            repo="https://github.com/example/repo",
            rev="main",
            rev_lock="abc123",
            stage="upstream",
            path="data.csv",
            remote="s3://bucket",
        ),
    }

    assert track.is_import(data), "Should return True for PvtData with source"


def test_is_import_false() -> None:
    """is_import() returns False for pvt without source."""
    data: track.PvtData = {
        "path": "data.csv",
        "hash": "abc123",
        "size": 100,
    }

    assert not track.is_import(data), "Should return False for PvtData without source"


def test_discover_import_pvt_files(tmp_path: pathlib.Path) -> None:
    """discover_import_pvt_files() filters to imports only."""
    # Create import file
    track.write_pvt_file(
        tmp_path / "imported.csv.pvt",
        {
            "path": "imported.csv",
            "hash": "hash1",
            "size": 100,
            "source": track.ImportSource(
                repo="https://github.com/example/repo",
                rev="main",
                rev_lock="abc123",
                stage="upstream",
                path="data.csv",
                remote="s3://bucket",
            ),
        },
    )
    # Create non-import file
    track.write_pvt_file(
        tmp_path / "local.csv.pvt",
        {
            "path": "local.csv",
            "hash": "hash2",
            "size": 200,
        },
    )

    result = track.discover_import_pvt_files(tmp_path)

    assert len(result) == 1, "Should find only import files"
    imported_path = str(tmp_path / "imported.csv")
    assert imported_path in result, "Should include imported.csv"
    assert "source" in result[imported_path], "Result should have source field"


def test_pvt_source_with_non_string_values_rejected(tmp_path: pathlib.Path) -> None:
    """is_pvt_data() rejects source dict where values are not strings."""
    pvt_path = tmp_path / "bad_source.pvt"
    content = (
        "path: data.csv\nhash: abc123\nsize: 100\n"
        + "source:\n  repo: https://example.com\n  rev: main\n"
        + "  rev_lock: abc123\n  stage: 42\n  path: data.csv\n  remote: s3://bucket\n"
    )
    pvt_path.write_text(content)

    result = track.read_pvt_file(pvt_path)

    assert result is None, "Should reject source with non-string values"


def test_pvt_with_unknown_keys_rejected(tmp_path: pathlib.Path) -> None:
    """is_pvt_data() still rejects unknown keys."""
    pvt_path = tmp_path / "invalid.pvt"
    pvt_path.write_text("path: data.csv\nhash: abc123\nsize: 100\nunknown_key: value\n")

    result = track.read_pvt_file(pvt_path)

    assert result is None, "Should reject PvtData with unknown keys"
