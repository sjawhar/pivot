from __future__ import annotations

import pathlib
import unicodedata
from typing import Any, ClassVar, cast

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


def normalize_directory_out_key(key: str) -> str:
    """Normalize one key of a directory output's dict, a file path inside the directory.

    A directory output is a stage return value of ``dict[str, T]`` tagged ``directory``; each
    key names a file the worker writes under the output directory, and a consumer names one of
    those files the same way. Returns the key with redundant separators and ``.`` segments
    removed, Unicode NFC normalized.

    Raises:
        ValueError: Saying what is wrong with the key (empty, absolute, contains ``..``, no
            extension).
    """
    if not key or not key.strip():
        raise ValueError("empty or whitespace-only key not allowed")

    # Normalize Unicode to NFC for consistent cross-platform behavior
    # (macOS HFS+ uses NFD, which could cause duplicate detection issues)
    key = unicodedata.normalize("NFC", key)

    # Normalize the key (handles "foo//bar.yaml" -> "foo/bar.yaml")
    path_obj = pathlib.PurePosixPath(key)
    normalized = path_obj.as_posix()

    if normalized.startswith("/"):
        raise ValueError(f"absolute path not allowed: {key!r}")

    if ".." in path_obj.parts:
        raise ValueError(f"path traversal not allowed: {key!r}")

    # Extension required to ensure files match the loader's expected format
    # Use suffix instead of "." in name to correctly reject hidden files without extensions
    if not path_obj.suffix:
        raise ValueError(f"key must include file extension: {key!r}")

    # Reject filenames where the stem (name without extension) is empty or whitespace-only
    # e.g., "   .json" has stem="   " which would create problematic filenames
    if not path_obj.stem.strip():
        raise ValueError(f"filename cannot be empty or whitespace-only: {key!r}")

    return normalized


def directory_out_members(value: object, output_name: str) -> list[tuple[str, Any]]:
    """Check a directory output's returned dict and pair each normalized key with its value.

    Refuses a value that is not a dict, an empty dict, a non-string key, any key
    ``normalize_directory_out_key`` refuses, two keys that normalize to one path, and two keys
    that differ only by case (they would name one file on a case-insensitive filesystem).
    """
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Directory output '{output_name}' expects dict, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"Directory output '{output_name}': dict must be non-empty")

    # Cast needed: isinstance narrows object to dict[Unknown, Unknown]; keys are checked below
    value_dict = cast("dict[object, Any]", value)
    members = list[tuple[str, Any]]()
    seen_normalized = dict[str, str]()  # normalized -> original
    seen_lowercase = dict[str, str]()  # lowercase -> normalized
    for key, item_value in value_dict.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Directory output '{output_name}': keys must be strings, got {type(key).__name__}"
            )
        try:
            normalized = normalize_directory_out_key(key)
        except ValueError as error:
            raise ValueError(f"Directory output '{output_name}': {error}") from None

        if normalized in seen_normalized:
            first, second = sorted([key, seen_normalized[normalized]])
            raise ValueError(
                f"Directory output '{output_name}': duplicate key after normalization: "
                + f"{first!r} and {second!r} both normalize to {normalized!r}"
            )
        seen_normalized[normalized] = key

        lowercase = normalized.lower()
        if lowercase in seen_lowercase:
            first, second = sorted([normalized, seen_lowercase[lowercase]])
            raise ValueError(
                f"Directory output '{output_name}': keys would collide on case-insensitive "
                + f"filesystems: {first!r} and {second!r}"
            )
        seen_lowercase[lowercase] = normalized

        members.append((normalized, item_value))
    return members
