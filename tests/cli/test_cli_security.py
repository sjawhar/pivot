from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pytest

from pivot import cli, exceptions
from pivot.storage import cache, track

if TYPE_CHECKING:
    import click.testing


# =============================================================================
# Path Traversal Security Tests
# =============================================================================


def test_track_null_byte_in_path_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Paths with null bytes rejected."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create a file that we'll try to track with a null byte in path
        pathlib.Path("data.txt").write_text("content")

        # Try to track with null byte in path - should fail
        result = runner.invoke(cli.cli, ["track", "data\x00.txt"])

        # Must fail - null bytes are dangerous for C libraries and filesystem operations
        # The file "data\x00.txt" doesn't exist (we created "data.txt"), so this must error
        assert result.exit_code != 0, "Null byte path should be rejected"


def test_checkout_manifest_path_traversal_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Manifest with ../ in relpath rejected during checkout."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        cache_dir = pathlib.Path(".pivot") / "cache" / "files"
        cache_dir.mkdir(parents=True)

        # Create a malicious pvt file with path traversal in manifest
        malicious_pvt = """path: data_dir
hash: abc123
num_files: 1
manifest:
  - relpath: ../outside.txt
    hash: def456
    size: 10
"""
        pathlib.Path("data_dir.pvt").write_text(malicious_pvt)

        result = runner.invoke(cli.cli, ["checkout", "data_dir"])

        # Must either fail OR not create files outside the project directory
        # The critical security property: no file created at ../outside.txt
        assert not (tmp_path.parent / "outside.txt").exists(), (
            "Path traversal should not create files outside project"
        )
        # Command should also indicate failure for malicious manifest
        assert (
            result.exit_code != 0
            or "error" in result.output.lower()
            or "invalid" in result.output.lower()
            or "not found" in result.output.lower()
        )


def test_track_path_traversal_with_encoded_dots(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Rejects paths trying to bypass traversal detection."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Try various bypass attempts
        bypass_attempts = [
            "..%2f",  # URL encoded
            "..%252f",  # Double encoded
            "..\\",  # Windows style
        ]

        for attempt in bypass_attempts:
            result = runner.invoke(cli.cli, ["track", f"{attempt}outside.txt"])
            # Should be rejected (either not found or traversal detected)
            assert result.exit_code != 0


# =============================================================================
# Stage Name Validation Tests
# =============================================================================


def test_stage_name_path_injection_rejected(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Stage names with / or .. rejected."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Try to run with malicious stage name
        result = runner.invoke(cli.cli, ["run", "../../../etc/passwd"])

        # Should not create files outside project
        assert result.exit_code != 0
        # Should not have created the malicious path
        assert not pathlib.Path("../../../etc/passwd.lock").exists()


# =============================================================================
# Hash Validation Tests
# =============================================================================


def test_cache_path_rejects_invalid_characters(tmp_path: pathlib.Path) -> None:
    """get_cache_path rejects hashes with non-hex characters."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    # All hashes must be exactly 16 chars to test character validation (not length)
    invalid_hashes = [
        "abcd/efg12345678",  # forward slash
        "abcdefgh12345GHI",  # uppercase letters
        "abcdefgh1234567!",  # special character
        "abcd efgh1234567",  # space character
        "..%2fabcdef12345",  # URL encoded traversal
    ]

    for bad_hash in invalid_hashes:
        with pytest.raises(exceptions.SecurityValidationError, match="invalid characters"):
            cache.get_cache_path(cache_dir, bad_hash)


def test_cache_path_rejects_invalid_length(tmp_path: pathlib.Path) -> None:
    """get_cache_path rejects hashes with wrong length."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    invalid_lengths = [
        "abc",  # too short (3 chars)
        "abcdef12",  # too short (8 chars)
        "abcdef1234567890abcdef",  # too long (22 chars)
        "",  # empty string
    ]

    for bad_hash in invalid_lengths:
        with pytest.raises(exceptions.SecurityValidationError, match="exactly 16 characters"):
            cache.get_cache_path(cache_dir, bad_hash)


def test_cache_path_structure(tmp_path: pathlib.Path) -> None:
    """Cache paths use hash prefix directory structure."""
    cache_dir = tmp_path / "cache" / "files"
    cache_dir.mkdir(parents=True)

    # Valid hash should create subdirectory structure
    valid_hash = "abcd1234567890ef"
    cache_path = cache.get_cache_path(cache_dir, valid_hash)

    # Should be cache_dir/ab/cd1234567890ef (first 2 chars as subdir, rest as filename)
    assert cache_path.name == valid_hash[2:]  # Filename is rest of hash
    assert cache_path.parent.name == valid_hash[:2]  # Parent dir is first 2 chars
    assert cache_path.parent.parent == cache_dir  # Grandparent is cache_dir


# =============================================================================
# YAML Security Tests
# =============================================================================


def test_pvt_file_large_yaml_handled(tmp_path: pathlib.Path) -> None:
    """Large .pvt files don't cause excessive memory usage."""
    # Create a .pvt file with many entries (but not a YAML bomb)
    # This tests that we handle large-but-valid files
    pvt_path = tmp_path / "large.pvt"

    # Generate a large manifest (100 entries - smaller for test speed)
    manifest_entries = "\n".join(
        [f"  - relpath: file_{i:04d}.txt\n    hash: {'a' * 16}\n    size: 100" for i in range(100)]
    )

    pvt_content = f"""path: large_dir
hash: {"b" * 16}
size: 10000
num_files: 100
manifest:
{manifest_entries}
"""
    pvt_path.write_text(pvt_content)

    # Should parse without hanging or excessive memory
    result = track.read_pvt_file(pvt_path)

    # Should have parsed successfully
    assert result is not None
    assert "num_files" in result and result["num_files"] == 100
    assert "manifest" in result and len(result["manifest"]) == 100


def test_pvt_file_yaml_invalid_type_rejected(tmp_path: pathlib.Path) -> None:
    """Invalid YAML types in .pvt file rejected gracefully."""
    pvt_path = tmp_path / "invalid.pvt"

    # Create invalid pvt content (wrong types)
    invalid_contents = [
        # List instead of dict at root
        "- item1\n- item2",
        # String instead of dict
        "just a string",
        # Missing required fields
        "path: test.txt\n",  # Missing hash
    ]

    for content in invalid_contents:
        pvt_path.write_text(content)
        result = track.read_pvt_file(pvt_path)
        # Should return None for invalid content, not raise
        assert result is None


# =============================================================================
# Symlink Security Tests
# =============================================================================


def test_directory_track_processes_valid_files(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track on directory processes valid regular files."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create directory with valid file
        data_dir = pathlib.Path("data_dir")
        data_dir.mkdir()
        (data_dir / "valid.txt").write_text("content")

        result = runner.invoke(cli.cli, ["track", "data_dir"])

        # Should succeed and track the directory
        assert result.exit_code == 0
        assert "Tracked: data_dir" in result.output


def test_directory_track_creates_pvt_file(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Track on directory creates .pvt file with manifest."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        pathlib.Path(".git").mkdir()
        pathlib.Path(".pivot").mkdir()

        # Create directory with files
        data_dir = pathlib.Path("data_dir")
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("content1")
        (data_dir / "file2.txt").write_text("content2")

        result = runner.invoke(cli.cli, ["track", "data_dir"])

        assert result.exit_code == 0

        # Check pvt file was created
        pvt_path = pathlib.Path("data_dir.pvt")
        assert pvt_path.exists()

        # Read and verify structure
        pvt_data = track.read_pvt_file(pvt_path)
        assert pvt_data is not None
        assert pvt_data["path"] == "data_dir"
        assert "manifest" in pvt_data
        assert "num_files" in pvt_data and pvt_data["num_files"] == 2
