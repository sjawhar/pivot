"""Show readers resolve release cache objects in an upgraded layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pivot import state_dir as state_dir_mod
from pivot.remote import sync
from pivot.show import data, metrics
from pivot.storage import cache

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator

_HASH = "abc1234567890def"


def _legacy_cache_file(root: pathlib.Path, text: str) -> pathlib.Path:
    path = root / ".pivot" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _configure_state_dir(root: pathlib.Path) -> None:
    (root / ".pivot" / "config.yaml").write_text("core:\n  state_dir: altstate\n")


@pytest.fixture(autouse=True)
def _report_state_dir_warnings() -> Generator[None]:
    """Report state-directory warnings for this test only, as one CLI invocation does."""
    token = state_dir_mod.begin_invocation(report=True)
    yield
    state_dir_mod.end_invocation(token)


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


def test_cache_object_resolver_rejects_a_primary_directory(
    set_project_root: pathlib.Path,
) -> None:
    _configure_state_dir(set_project_root)
    primary = set_project_root / "altstate" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    primary.mkdir(parents=True)

    resolved = cache.resolve_cache_object_path(
        set_project_root / "altstate" / "cache" / "files", _HASH
    )

    assert resolved is None


def test_cache_object_resolver_rejects_a_legacy_directory(
    set_project_root: pathlib.Path,
) -> None:
    _configure_state_dir(set_project_root)
    legacy = set_project_root / ".pivot" / "cache" / "files" / _HASH[:2] / _HASH[2:]
    legacy.mkdir(parents=True)

    resolved = cache.resolve_cache_object_path(
        set_project_root / "altstate" / "cache" / "files", _HASH
    )

    assert resolved is None


@pytest.mark.parametrize("prefix_kind", ["file", "symlink-loop"])
@pytest.mark.parametrize("layout", ["configured", "legacy"])
def test_cache_object_resolver_reports_absent_when_the_prefix_is_not_a_directory(
    set_project_root: pathlib.Path, layout: str, prefix_kind: str
) -> None:
    """Path.exists() called this absent (ENOTDIR, ELOOP), as did 0.5.2; the resolver agrees."""
    _configure_state_dir(set_project_root)
    state = set_project_root / ("altstate" if layout == "configured" else ".pivot")
    prefix = state / "cache" / "files" / _HASH[:2]
    prefix.parent.mkdir(parents=True)
    if prefix_kind == "file":
        prefix.write_text("not a cache prefix directory\n")
    else:
        prefix.symlink_to(prefix.name)

    resolved = cache.resolve_cache_object_path(
        set_project_root / "altstate" / "cache" / "files", _HASH
    )

    assert resolved is None
