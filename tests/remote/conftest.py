"""Shared fixtures for S3 remote storage tests using moto."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

from pivot.remote import storage as remote_mod

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from types_aiobotocore_s3 import S3Client


@pytest.fixture
async def moto_s3_bucket(
    moto_patch_session: object, aioboto3_s3_client: S3Client
) -> AsyncGenerator[str]:
    """Create a test bucket in moto with unique name for xdist support.

    Uses pytest-aioboto3's moto_patch_session to mock S3 and aioboto3_s3_client
    to create the bucket. Bucket name includes a unique suffix to prevent
    conflicts when running tests in parallel with pytest-xdist.

    Yields:
        str: The bucket name (e.g., "test-bucket-a1b2c3d4").
    """
    bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    await aioboto3_s3_client.create_bucket(Bucket=bucket_name)
    yield bucket_name
    # Cleanup handled by moto_patch_session teardown


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
