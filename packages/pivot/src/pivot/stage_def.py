from __future__ import annotations

import pathlib
import unicodedata
from typing import Any, ClassVar, cast

import pydantic

from pivot import exceptions

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
        ValueError: Saying what is wrong with the key (empty, a backslash, colon or null byte,
            absolute, contains ``..``, no extension). The message leaves naming the key to the
            caller.
    """
    if not key or not key.strip():
        raise ValueError("empty or whitespace-only key not allowed")

    # Each file is named by its path in an artifact identity ("stage:path"), which cannot
    # hold these characters, so no stage could depend on such a file
    if any(char in key for char in "\\:\0"):
        raise ValueError(
            "backslashes, colons and null bytes are not allowed, because Pivot names each file "
            + "by its path"
        )

    # Normalize Unicode to NFC for consistent cross-platform behavior
    # (macOS HFS+ uses NFD, which could cause duplicate detection issues)
    key = unicodedata.normalize("NFC", key)

    # Normalize the key (handles "foo//bar.yaml" -> "foo/bar.yaml")
    path_obj = pathlib.PurePosixPath(key)
    normalized = path_obj.as_posix()

    if normalized.startswith("/"):
        raise ValueError("absolute path not allowed")

    if ".." in path_obj.parts:
        raise ValueError("path traversal not allowed")

    # Extension required to ensure files match the loader's expected format
    # Use suffix instead of "." in name to correctly reject hidden files without extensions
    if not path_obj.suffix:
        raise ValueError("key must include file extension")

    # Reject filenames where the stem (name without extension) is empty or whitespace-only
    # e.g., "   .json" has stem="   " which would create problematic filenames
    if not path_obj.stem.strip():
        raise ValueError("filename cannot be empty or whitespace-only")

    return normalized


def directory_out_members(value: object, stage_name: str, label: str) -> list[tuple[str, Any]]:
    """Check a directory output's returned dict and pair each normalized key with its value.

    ``label`` names the output in messages (``the return value``, ``output 'parts'``). Refuses
    a value that is not a dict, an empty dict, a non-string key, any key
    ``normalize_directory_out_key`` refuses, two keys that normalize to one path, two keys that
    differ only by case (they would name one file on a case-insensitive filesystem), and a key
    inside another key's path (``a.json`` and ``a.json/b.json``), which cannot be both a file
    and a directory. Because of the last rule, every directory between the output directory
    and a key's file is one the worker creates, never a file or directory a stage returned.

    Raises:
        exceptions.StageDefinitionError: Naming the stage, the output and the key.
    """
    owner = f"Stage '{stage_name}': {label}"
    if not isinstance(value, dict):
        raise exceptions.StageDefinitionError(
            f"{owner} is tagged directory, so it must be a dict from file path to value, "
            + f"not {type(value).__name__}"
        )
    if not value:
        raise exceptions.StageDefinitionError(
            f"{owner} is an empty dict; a directory output must be a non-empty dict"
        )

    # Cast needed: isinstance narrows object to dict[Unknown, Unknown]; keys are checked below
    value_dict = cast("dict[object, Any]", value)
    members = list[tuple[str, Any]]()
    seen_normalized = dict[str, str]()  # normalized -> original
    seen_lowercase = dict[str, str]()  # lowercase -> normalized
    for key, item_value in value_dict.items():
        if not isinstance(key, str):
            raise exceptions.StageDefinitionError(
                f"{owner} has key {key!r} of type {type(key).__name__}; keys must be strings"
            )
        try:
            normalized = normalize_directory_out_key(key)
        except ValueError as error:
            raise exceptions.StageDefinitionError(f"{owner} has key {key!r}: {error}") from None

        if normalized in seen_normalized:
            first, second = sorted([key, seen_normalized[normalized]])
            raise exceptions.StageDefinitionError(
                f"{owner} has keys {first!r} and {second!r}, which both normalize to "
                + f"{normalized!r}"
            )
        seen_normalized[normalized] = key

        lowercase = normalized.lower()
        if lowercase in seen_lowercase:
            first, second = sorted([normalized, seen_lowercase[lowercase]])
            raise exceptions.StageDefinitionError(
                f"{owner} has keys {first!r} and {second!r}, which would collide on "
                + "case-insensitive filesystems"
            )
        seen_lowercase[lowercase] = normalized

        members.append((normalized, item_value))

    for normalized, _ in members:
        for parent in pathlib.PurePosixPath(normalized.lower()).parents:
            if str(parent) in seen_lowercase:
                raise exceptions.StageDefinitionError(
                    f"{owner} has keys where {seen_lowercase[str(parent)]!r} cannot be both a "
                    + f"file and the directory holding {normalized!r}"
                )
    return members
