# pyright: reportImplicitRelativeImport=false, reportMissingImports=false
from __future__ import annotations

import logging
import pathlib  # noqa: TC003 - used at runtime in _write_output
import unicodedata
from collections.abc import Mapping  # noqa: TC003 - used in function signatures
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pydantic

from pivot import outputs

if TYPE_CHECKING:
    from pivot import loaders

logger = logging.getLogger(__name__)

# Key used in out_specs for single-output stages (non-TypedDict returns)
SINGLE_OUTPUT_KEY = "_single"


class StageParams(pydantic.BaseModel):
    """Base class for stage parameters (Pydantic model).

    Use as a simple base class for parameter-only stages:

        class TrainParams(StageParams):
            learning_rate: float = 0.01
            batch_size: int = 32

        def train(
            config: TrainParams,
            data: Annotated[DataFrame, Dep("input.csv", CSV())],
        ) -> TrainOutputs:
            ...

    For testing, just pass the data directly:

        result = train(TrainParams(learning_rate=0.5), test_df)
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict()


# ==============================================================================
# Common validation and write helpers
# ==============================================================================


def _validate_path_not_escaped(path: pathlib.Path, project_root: pathlib.Path) -> None:
    """Validate that resolved path is within project root (no path traversal)."""
    resolved = path.resolve()
    root_resolved = project_root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Path escapes project root: '{path}' resolves to '{resolved}' which is outside '{root_resolved}'"
        )


def _validate_directory_out_key(key: str, output_name: str) -> str:
    """Validate and normalize a DirectoryOut key (relative path within directory).

    Returns the normalized key (redundant separators removed, Unicode NFC normalized).

    Raises:
        ValueError: If key is invalid (empty, whitespace-only, absolute, contains .., no extension)
    """
    if not key or not key.strip():
        raise ValueError(f"DirectoryOut '{output_name}': empty or whitespace-only key not allowed")

    # Normalize Unicode to NFC for consistent cross-platform behavior
    # (macOS HFS+ uses NFD, which could cause duplicate detection issues)
    key = unicodedata.normalize("NFC", key)

    # Normalize the key (handles "foo//bar.yaml" -> "foo/bar.yaml")
    path_obj = pathlib.PurePosixPath(key)
    normalized = path_obj.as_posix()

    # Check for absolute path
    if normalized.startswith("/"):
        raise ValueError(f"DirectoryOut '{output_name}': absolute path not allowed: {key!r}")

    # Check for path traversal (..)
    if ".." in path_obj.parts:
        raise ValueError(f"DirectoryOut '{output_name}': path traversal not allowed: {key!r}")

    # Extension required to ensure files match the loader's expected format
    # Use suffix instead of "." in name to correctly reject hidden files without extensions
    if not path_obj.suffix:
        raise ValueError(f"DirectoryOut '{output_name}': key must include file extension: {key!r}")

    # Reject filenames where the stem (name without extension) is empty or whitespace-only
    # e.g., "   .json" has stem="   " which would create problematic filenames
    if not path_obj.stem.strip():
        raise ValueError(
            f"DirectoryOut '{output_name}': filename cannot be empty or whitespace-only: {key!r}"
        )

    return normalized


def _collect_directory_out_ops(
    name: str,
    spec: outputs.DirectoryOut[Any],
    value: Any,
    project_root: pathlib.Path,
    write_ops: list[tuple[pathlib.Path, Any, loaders.Writer[Any]]],
) -> None:
    """Collect write operations for a DirectoryOut.

    Validates the value dict and each key, then appends write operations to write_ops.
    """
    # Validate value is a dict
    if not isinstance(value, dict):
        raise RuntimeError(
            f"DirectoryOut '{name}' expects dict[str, T], got {type(value).__name__}"
        )

    # Validate dict is non-empty
    if not value:
        raise ValueError(f"DirectoryOut '{name}': dict must be non-empty")

    # spec.path is guaranteed to be str ending with "/" by DirectoryOut.__post_init__
    dir_path = spec.path

    # Track normalized keys to detect duplicates after normalization
    seen_normalized = dict[str, str]()  # normalized -> original
    # Track lowercased keys to detect case collisions (for case-insensitive filesystems)
    seen_lowercase = dict[str, str]()  # lowercase -> original normalized

    # Cast value to dict[Any, Any] - we validated it's a dict above
    value_dict = cast("dict[Any, Any]", value)

    for key, item_value in value_dict.items():
        # Validate key is a string
        if not isinstance(key, str):
            raise ValueError(
                f"DirectoryOut '{name}': keys must be strings, got {type(key).__name__}"
            )

        # Validate and normalize the key
        normalized_key = _validate_directory_out_key(key, name)

        # Check for duplicates after normalization
        if normalized_key in seen_normalized:
            # Sort keys for deterministic error message
            sorted_keys = sorted([key, seen_normalized[normalized_key]])
            raise ValueError(
                f"DirectoryOut '{name}': duplicate key after normalization: "
                + f"{sorted_keys[0]!r} and {sorted_keys[1]!r} both normalize to {normalized_key!r}"
            )
        seen_normalized[normalized_key] = key

        # Check for case collisions (would conflict on case-insensitive filesystems)
        lowercase_key = normalized_key.lower()
        if lowercase_key in seen_lowercase:
            existing_key = seen_lowercase[lowercase_key]
            if existing_key != normalized_key:
                sorted_keys = sorted([normalized_key, existing_key])
                raise ValueError(
                    f"DirectoryOut '{name}': keys would collide on case-insensitive filesystems: "
                    + f"{sorted_keys[0]!r} and {sorted_keys[1]!r}"
                )
        seen_lowercase[lowercase_key] = normalized_key

        # Build full path: directory + normalized key
        full_path = project_root / dir_path / normalized_key

        # Validate path hasn't escaped project root
        _validate_path_not_escaped(full_path, project_root)

        write_ops.append((full_path, item_value, spec.loader))


def save_return_outputs(
    return_value: Mapping[str, Any],
    specs: Mapping[str, outputs.BaseOut],
    project_root: pathlib.Path,
) -> None:
    """Save return value outputs to disk.

    Takes the return value from a stage function and saves each output
    to its configured path using its loader.

    Validates all inputs upfront before writing any files. Path overrides are
    already applied to specs at registration time.

    Args:
        return_value: The dict returned by the stage function
        specs: Output specs with paths already resolved (from registration)
        project_root: Root directory for relative paths

    Raises:
        ValueError: If path escapes project root
        KeyError: If output keys are missing from return_value
        RuntimeError: If value/path count mismatch for sequence outputs
    """
    # Validate all output keys exist
    missing = set(specs.keys()) - set(return_value.keys())
    if missing:
        raise KeyError(
            f"Missing return output keys: {sorted(missing)}. Return value keys: {sorted(return_value.keys())}"
        )

    # Warn about extra keys not declared as outputs
    extra = set(return_value.keys()) - set(specs.keys())
    if extra:
        logger.warning("Extra keys in return value not declared as outputs: %s", sorted(extra))

    # Collect all write operations and validate paths upfront
    write_ops: list[tuple[pathlib.Path, Any, loaders.Writer[Any]]] = []
    for name, spec in specs.items():
        path = spec.path
        value = return_value[name]

        if outputs.is_directory_out(spec):
            # DirectoryOut: value is dict[str, T], keys are relative paths within directory
            _collect_directory_out_ops(name, spec, value, project_root, write_ops)
        elif isinstance(path, (list, tuple)):
            if not isinstance(value, (list, tuple)):
                raise RuntimeError(
                    f"Output '{name}' has sequence path but non-sequence value: {type(value).__name__}"
                )
            value_seq = cast("list[Any] | tuple[Any, ...]", value)
            if len(value_seq) != len(path):
                raise RuntimeError(
                    f"Output '{name}' has {len(path)} paths but {len(value_seq)} values"
                )
            for p, v in zip(path, value_seq, strict=True):
                full_path = project_root / p
                # Defense-in-depth: validate path hasn't escaped (e.g., via symlink attack
                # between registration and execution)
                _validate_path_not_escaped(full_path, project_root)
                write_ops.append((full_path, v, spec.loader))
        else:
            full_path = project_root / path
            # Defense-in-depth: validate path hasn't escaped
            _validate_path_not_escaped(full_path, project_root)
            write_ops.append((full_path, value, spec.loader))

    # All validation passed - now write
    for full_path, value, loader in write_ops:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        loader.save(value, full_path)
