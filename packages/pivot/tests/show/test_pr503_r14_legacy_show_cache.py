"""Round 14 B2: show readers resolve release cache objects in an upgraded layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pivot import project
from pivot.remote import sync
from pivot.show import data, metrics
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib

    import pytest

_HASH = "abc1234567890def"


def _legacy_cache_file(root: pathlib.Path, text: str) -> pathlib.Path:
    path = root / ".pivot" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _configure_state_dir(root: pathlib.Path) -> None:
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")
    project.enable_state_dir_warnings()


def test_data_cache_reader_uses_legacy_cache_object(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _configure_state_dir(set_project_root)
    _legacy_cache_file(set_project_root, "id,value\n1,old\n")

    restored = data.restore_data_from_cache("data.csv", _HASH)

    assert restored is not None
    assert restored.read_text() == "id,value\n1,old\n"
    restored.unlink()
    assert "Pivot 0.5.2 stored locks and cache" in caplog.text


def test_metrics_cache_reader_uses_legacy_cache_object(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _configure_state_dir(set_project_root)
    _legacy_cache_file(set_project_root, '{"score": 1}\n')

    found = metrics.collect_metrics_from_head(["metrics/test.json"], {"metrics/test.json": _HASH})

    assert found == {"metrics/test.json": {"score": 1}}
    assert "Pivot 0.5.2 stored locks and cache" in caplog.text


def test_legacy_cache_scan_reports_the_migration_warning(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _configure_state_dir(set_project_root)
    _legacy_cache_file(set_project_root, "cached\n")

    hashes = sync.get_local_cache_hashes(set_project_root / "altstate" / "cache")

    assert hashes == {_HASH}
    assert "Pivot 0.5.2 stored locks and cache" in caplog.text


def test_current_cache_scan_stays_silent(
    set_project_root: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    _configure_state_dir(set_project_root)
    current = set_project_root / "altstate" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    current.parent.mkdir(parents=True)
    current.write_text("cached\n")

    hashes = sync.get_local_cache_hashes(set_project_root / "altstate" / "cache")

    assert hashes == {_HASH}
    assert "Pivot 0.5.2 stored locks and cache" not in caplog.text


def test_cache_object_resolver_rejects_a_primary_symlink(
    set_project_root: pathlib.Path,
) -> None:
    _configure_state_dir(set_project_root)
    primary = set_project_root / "altstate" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    primary.parent.mkdir(parents=True)
    secret = set_project_root / "secret"
    secret.write_text("untrusted\n")
    primary.symlink_to(secret)

    resolved = cache.resolve_cache_object_path(
        set_project_root / "altstate" / "cache" / "files", _HASH
    )

    assert resolved is None


def test_cache_object_resolver_rejects_a_legacy_symlink_even_when_a_sentinel_exists(
    set_project_root: pathlib.Path,
) -> None:
    _configure_state_dir(set_project_root)
    primary_files = set_project_root / "altstate" / "cache" / "files"
    legacy = set_project_root / ".pivot" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    legacy.parent.mkdir(parents=True)
    secret = set_project_root / "secret"
    secret.write_text("untrusted\n")
    legacy.symlink_to(secret)
    sentinel = primary_files / _HASH[:2] / f".pivot-missing-cache-object-{_HASH[2:]}"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("must not be returned\n")

    resolved = cache.resolve_cache_object_path(primary_files, _HASH)

    assert resolved is None
