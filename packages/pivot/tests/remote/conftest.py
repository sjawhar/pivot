"""Shared fixtures for S3 remote storage tests using moto."""

from __future__ import annotations

import pytest

from pivot.cli import helpers as cli_helpers
from pivot.remote import storage as remote_mod


@pytest.fixture(autouse=True)
def _mock_get_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock cli_helpers.get_stage to return all-cached outputs.

    Sync functions filter non-cached outputs via registry; tests without a
    pipeline can return empty outs to allow hashes through.
    """

    def _get_stage(name: str) -> dict[str, object]:
        return {"outs": []}

    monkeypatch.setattr(cli_helpers, "get_stage", _get_stage)


@pytest.fixture
def s3_remote(moto_s3_bucket: str) -> remote_mod.S3Remote:
    """Create S3Remote instance pointing to moto test bucket.

    Creates a real S3Remote instance configured to use the moto-mocked S3
    bucket. This allows testing S3Remote methods against a real implementation
    without hitting actual AWS infrastructure.

    Args:
        moto_s3_bucket: The moto bucket name from the moto_s3_bucket fixture.

    Returns:
        S3Remote: Configured S3Remote instance with test-prefix.
    """
    return remote_mod.S3Remote(f"s3://{moto_s3_bucket}/test-prefix/")
