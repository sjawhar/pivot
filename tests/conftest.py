"""Shared test fixtures for Fastpipe tests.

Provides common fixtures used across multiple test files.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from fastpipe.registry import REGISTRY


@pytest.fixture
def tmp_pipeline_dir() -> Generator[Path]:
    """Create temporary directory for pipeline tests.

    Yields:
        Path to temporary directory (cleaned up after test)

    Example:
        >>> def test_something(tmp_pipeline_dir):
        ...     data_file = tmp_pipeline_dir / "data.csv"
        ...     data_file.write_text("id,value\\n1,10\\n")
        ...     assert data_file.exists()
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_file(tmp_pipeline_dir: Path) -> Path:
    """Create sample CSV data file.

    Args:
        tmp_pipeline_dir: Temporary directory fixture

    Returns:
        Path to created CSV file

    Example:
        >>> def test_data_processing(sample_data_file):
        ...     import pandas as pd
        ...     df = pd.read_csv(sample_data_file)
        ...     assert len(df) == 3
    """
    data_file = tmp_pipeline_dir / "data.csv"
    data_file.write_text("id,value\n1,10\n2,20\n3,30\n")
    return data_file


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None]:
    """Reset stage registry before and after each test.

    Ensures test isolation by clearing the global registry.

    Example:
        >>> def test_stage_registration():
        ...     # Registry automatically cleaned before this test
        ...     assert len(REGISTRY.list_stages()) == 0
    """
    original_stages = REGISTRY._stages.copy()
    REGISTRY.clear()
    yield
    REGISTRY._stages = original_stages
