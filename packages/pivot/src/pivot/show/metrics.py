from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any, TypedDict, cast

import flatten_dict
import yaml

from pivot import config, git, project, types, yaml_config
from pivot.show import common
from pivot.storage import cache
from pivot.types import ChangeType, MetricData, MetricValue, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

_METRIC_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".csv"})


class MetricDiff(TypedDict):
    """Diff info for a single metric value."""

    path: str
    key: str
    old: MetricValue
    new: MetricValue
    change_type: ChangeType


class MetricsError(Exception):
    """Error parsing or processing metrics."""


def parse_metric_file(path: pathlib.Path) -> MetricData:
    """Parse metric file, auto-detecting format by extension. Returns flattened dict."""
    suffix = path.suffix.lower()
    if suffix not in _METRIC_EXTENSIONS:
        raise MetricsError(f"Unsupported metric file format: {suffix}")
    try:
        if suffix == ".json":
            return _parse_json(path)
        elif suffix in (".yaml", ".yml"):
            return _parse_yaml(path)
        else:  # .csv
            return _parse_csv(path)
    except (OSError, json.JSONDecodeError, yaml.YAMLError, csv.Error, UnicodeDecodeError) as e:
        raise MetricsError(f"Failed to parse {path}: {e}") from e


def _validate_and_flatten(data: object, format_name: str) -> MetricData:
    """Validate parsed data is dict and flatten nested dicts."""
    if not isinstance(data, dict):
        raise MetricsError(f"Expected dict in {format_name} file, got {type(data).__name__}")
    # Cast needed: isinstance narrows to dict[Unknown, Unknown], but JSON/YAML dicts have str keys
    return _flatten_dict(cast("dict[str, Any]", data))


def _parse_json(path: pathlib.Path) -> MetricData:
    """Parse JSON file and flatten nested dicts."""
    with open(path) as f:
        data = json.load(f)
    return _validate_and_flatten(data, "JSON")


def _parse_yaml(path: pathlib.Path) -> MetricData:
    """Parse YAML file and flatten nested dicts."""
    with open(path) as f:
        data: object = yaml.load(f, Loader=yaml_config.Loader)
    return _validate_and_flatten(data, "YAML")


def _parse_csv(path: pathlib.Path) -> MetricData:
    """Parse CSV file (first column = key, second = value)."""
    result = dict[str, MetricValue]()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if len(row) == 0:
                continue  # Skip empty rows
            if len(row) < 2:
                logger.warning(f"{path}:{line_num}: skipping row with {len(row)} column(s), need 2")
                continue
            key, value = row[0], row[1]
            result[key] = _parse_value(value)
    return result


def _parse_value(value: str) -> MetricValue:
    """Parse string value to appropriate type (int, float, bool, or str)."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _flatten_dict(data: dict[str, object]) -> MetricData:
    """Flatten nested dict using dot notation (e.g., nested.f1)."""
    if not data:
        return {}
    # flatten_dict returns dict[Unknown, Unknown]; cast to MetricData since we know
    # JSON/YAML values become primitives after flattening
    return cast("MetricData", flatten_dict.flatten(data, reducer="dot"))


def _metric_outs_by_stage() -> dict[str, list[types.ArtifactRef]]:
    from pivot.cli import helpers as cli_helpers

    outs_by_stage = dict[str, list[types.ArtifactRef]]()
    for stage_name in cli_helpers.list_stages():
        metric_outs = [
            out
            for out in cli_helpers.get_stage(stage_name)["outs"]
            if out.tag is types.ArtifactTag.METRIC
        ]
        if metric_outs:
            outs_by_stage[stage_name] = metric_outs
    return outs_by_stage


def collect_metrics_from_stages() -> dict[str, dict[str, MetricData]]:
    """Collect metrics by parsing metric output files from disk.

    Returns {stage_name: {path: {key: value}}}, paths relative to the project root.
    """
    from pivot.cli import helpers as cli_helpers

    store = cli_helpers.require_workspace_store()
    project_root = project.get_project_root()
    result = dict[str, dict[str, MetricData]]()
    for stage_name, metric_outs in _metric_outs_by_stage().items():
        stage_metrics = dict[str, MetricData]()
        for out in metric_outs:
            path = store.resolve_display_path(out)
            rel_path = project.to_relative_path(path, project_root)
            if not path.exists():
                logger.warning(f"Metric file not found: {rel_path} (stage: {stage_name})")
                continue
            try:
                stage_metrics[rel_path] = parse_metric_file(path)
            except MetricsError as e:
                logger.warning(f"Failed to parse metrics from {rel_path}: {e}")
        if stage_metrics:
            result[stage_name] = stage_metrics
    return result


def collect_all_stage_metrics_flat() -> dict[str, MetricData]:
    """Collect metrics from all stages, flattened to {path: {key: value}}."""
    result = dict[str, MetricData]()
    for stage_data in collect_metrics_from_stages().values():
        result.update(stage_data)
    return result


def collect_metrics_from_files(
    targets: Sequence[str],
    recursive: bool = False,
    tolerant: bool = False,
) -> dict[str, MetricData]:
    """Parse metric files directly from filesystem. Returns {path: {key: value}}.

    Args:
        targets: File/directory paths to collect metrics from.
        recursive: If True, search directories recursively.
        tolerant: If True, warn and skip missing files. If False (default), raise error.
    """
    result = dict[str, MetricData]()
    for target in targets:
        path = pathlib.Path(target)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in _METRIC_EXTENSIONS:
                    result[str(file_path)] = parse_metric_file(file_path)
        elif path.is_file():
            result[str(path)] = parse_metric_file(path)
        else:
            if tolerant:
                logger.warning(f"Metric file not found: {target}")
                continue
            raise MetricsError(f"Path not found: {target}")
    return result


def diff_metrics(
    old: Mapping[str, Mapping[str, MetricValue]],
    new: Mapping[str, Mapping[str, MetricValue]],
) -> list[MetricDiff]:
    """Compare old vs new metrics. Returns list of diffs."""
    raw_diffs = common.build_two_level_diff(old, new)
    return [
        MetricDiff(path=path, key=key, old=old_val, new=new_val, change_type=change)
        for path, key, old_val, new_val, change in raw_diffs
    ]


def format_metrics_table(
    metrics: Mapping[str, Mapping[str, MetricValue]],
    output_format: OutputFormat | None,
    precision: int,
) -> str:
    """Format metrics for display. output_format: None (plain), 'json', or 'md'."""
    if output_format == OutputFormat.JSON:
        return common.format_json(dict(metrics))

    rows = list[list[str]]()
    for path, values in sorted(metrics.items()):
        for key, value in sorted(values.items()):
            rows.append([path, key, _format_value(value, precision)])

    return common.format_table(rows, ["Path", "Key", "Value"], output_format, "No metrics found.")


def format_diff_table(
    diffs: list[MetricDiff],
    output_format: OutputFormat | None,
    precision: int,
    show_path: bool = True,
) -> str:
    """Format metric diffs for display. output_format: None (plain), 'json', or 'md'."""
    if output_format == OutputFormat.JSON:
        return common.format_json(diffs)

    rows = list[list[str]]()
    for diff in diffs:
        row = [
            diff["key"],
            _format_value(diff["old"], precision),
            _format_value(diff["new"], precision),
            diff["change_type"],
        ]
        if show_path:
            row.insert(0, diff["path"])
        rows.append(row)

    headers = ["Key", "Old", "New", "Change"]
    if show_path:
        headers.insert(0, "Path")

    return common.format_table(rows, headers, output_format, "No metric changes.")


def _format_value(value: MetricValue, precision: int) -> str:
    """Format value with precision for floats, '-' for None."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def parse_metric_content(content: str | bytes, path: str) -> MetricData:
    """Parse metric content string/bytes, auto-detecting format by file extension."""
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    suffix = pathlib.Path(path).suffix.lower()
    if suffix not in _METRIC_EXTENSIONS:
        raise MetricsError(f"Unsupported metric file format: {suffix}")
    try:
        if suffix == ".json":
            data = json.loads(content)
            return _validate_and_flatten(data, "JSON")
        elif suffix in (".yaml", ".yml"):
            data: object = yaml.load(content, Loader=yaml_config.Loader)
            return _validate_and_flatten(data, "YAML")
        else:  # .csv
            return _parse_csv_content(content)
    except (json.JSONDecodeError, yaml.YAMLError, csv.Error, UnicodeDecodeError) as e:
        raise MetricsError(f"Failed to parse {path}: {e}") from e


def _parse_csv_content(content: str) -> MetricData:
    """Parse CSV content string (first column = key, second = value)."""
    result = dict[str, MetricValue]()
    reader = csv.reader(io.StringIO(content))
    for line_num, row in enumerate(reader, start=1):
        if len(row) == 0:
            continue
        if len(row) < 2:
            logger.warning(f"CSV:{line_num}: skipping row with {len(row)} column(s), need 2")
            continue
        key, value = row[0], row[1]
        result[key] = _parse_value(value)
    return result


def get_metric_info_from_head() -> dict[str, str | None]:
    """Map each metric output's workspace path to its hash in the lock files at git HEAD."""
    from pivot.cli import helpers as cli_helpers

    return common.output_hashes_from_head(
        _metric_outs_by_stage(), cli_helpers.require_workspace_store()
    )


def collect_metrics_from_head(
    paths: Sequence[str],
    head_hashes: Mapping[str, str | None],
) -> dict[str, MetricData]:
    """Collect metrics from HEAD, using cache if available, otherwise git.

    Args:
        paths: Relative paths to metric files
        head_hashes: Mapping of path -> hash from lock files at HEAD

    Returns {path: {key: value}} for files found.
    """
    cache_dir = config.locate_cache_dir() / "files"
    result = dict[str, MetricData]()

    # Collect paths that need git fallback
    paths_needing_git = list[str]()

    for rel_path in paths:
        file_hash = head_hashes.get(rel_path)

        # Try cache first if we have a hash; a malformed hash is never turned into a path
        if file_hash is not None and cache.is_valid_hash(file_hash):
            cached_path = cache.resolve_cache_object_path(cache_dir, file_hash)
            if cached_path.exists():
                try:
                    content = cached_path.read_bytes()
                    result[rel_path] = parse_metric_content(content, rel_path)
                    continue
                except (OSError, MetricsError):
                    pass

        # Need to try git
        paths_needing_git.append(rel_path)

    # Read remaining files from git in batch
    if paths_needing_git:
        git_contents = git.read_files_from_head(paths_needing_git)
        for rel_path, content in git_contents.items():
            with contextlib.suppress(MetricsError):
                result[rel_path] = parse_metric_content(content, rel_path)

    return result
