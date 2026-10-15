from __future__ import annotations

import pathlib
import unicodedata
from typing import ClassVar

import pydantic

# Key used in out_specs for single-output stages (non-TypedDict returns)
SINGLE_OUTPUT_KEY = "_single"


class StageParams(pydantic.BaseModel):
    """Base class for stage parameters (Pydantic model).

    Use as a simple base class for stage configuration:

        class TrainParams(StageParams):
            learning_rate: float = 0.01
            batch_size: int = 32

        @pivot.stage
        def train(config: TrainParams, data: DataFrame) -> DataFrame:
            ...

    For testing, call the function directly:

        result = train(TrainParams(learning_rate=0.5), test_df)

    Unknown fields are refused, whether they come from code (``TrainParams(learnig_rate=1)``)
    or from ``params.yaml``. A subclass's own ``model_config`` merges with this one, so it
    keeps refusing them unless it sets ``extra`` itself.
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")


def validate_directory_out_key(key: str, output_name: str) -> str:
    """Validate and normalize one key of a directory output's dict (a path inside the directory).

    A directory output is a stage return value of ``dict[str, T]``; each key names a file the
    worker writes under the output directory. Returns the normalized key (redundant
    separators removed, Unicode NFC normalized).

    Raises:
        ValueError: If key is invalid (empty, whitespace-only, absolute, contains .., no extension)
    """
    if not key or not key.strip():
        raise ValueError(
            f"Directory output '{output_name}': empty or whitespace-only key not allowed"
        )

    # Normalize Unicode to NFC for consistent cross-platform behavior
    # (macOS HFS+ uses NFD, which could cause duplicate detection issues)
    key = unicodedata.normalize("NFC", key)

    # Normalize the key (handles "foo//bar.yaml" -> "foo/bar.yaml")
    path_obj = pathlib.PurePosixPath(key)
    normalized = path_obj.as_posix()

    # Check for absolute path
    if normalized.startswith("/"):
        raise ValueError(f"Directory output '{output_name}': absolute path not allowed: {key!r}")

    # Check for path traversal (..)
    if ".." in path_obj.parts:
        raise ValueError(f"Directory output '{output_name}': path traversal not allowed: {key!r}")

    # Extension required to ensure files match the loader's expected format
    # Use suffix instead of "." in name to correctly reject hidden files without extensions
    if not path_obj.suffix:
        raise ValueError(
            f"Directory output '{output_name}': key must include file extension: {key!r}"
        )

    # Reject filenames where the stem (name without extension) is empty or whitespace-only
    # e.g., "   .json" has stem="   " which would create problematic filenames
    if not path_obj.stem.strip():
        raise ValueError(
            f"Directory output '{output_name}': filename cannot be empty or whitespace-only: {key!r}"
        )

    return normalized
