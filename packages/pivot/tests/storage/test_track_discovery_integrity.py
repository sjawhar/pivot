import pathlib

import pytest

from pivot.storage import track


@pytest.mark.parametrize("local_first", [False, True], ids=["import-first", "local-first"])
def test_non_import_sidecar_cannot_overwrite_import_discovery(
    tmp_path: pathlib.Path, local_first: bool
) -> None:
    imported = track.PvtData(
        path="imported.csv",
        hash="abc123",
        size=10,
        source=track.ImportSource(
            repo="https://github.com/example/upstream",
            rev="main",
            rev_lock="abcdef123456",
            stage="upstream/data",
            path="data.csv",
            remote="s3://bucket/cache",
        ),
    )
    records = [
        (tmp_path / "imported.csv.pvt", imported),
        (tmp_path / "local.csv.pvt", track.PvtData(path="imported.csv", hash="def456", size=20)),
    ]
    for path, data in reversed(records) if local_first else records:
        track.write_pvt_file(path, data)

    discovered = track.discover_import_pvt_files(tmp_path)

    assert discovered == {str(tmp_path / "imported.csv"): imported}
