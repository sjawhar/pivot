"""Per-stage lock files for tracking pipeline state.

Each stage gets its own lock file (~1KB) instead of one monolithic file.
This enables parallel writes without contention and O(1) reads per stage.

Atomic Write Pattern:
    We write to a .tmp file first, then rename to the final path. This ensures:
    1. No partial/corrupted files if process is killed mid-write
    2. Readers never see incomplete data (rename is atomic on POSIX)
    3. Original file preserved until new version is complete

    Without this pattern, a crash during write would leave a truncated file
    that fails to parse, breaking all subsequent runs.
"""

import os
import pathlib
import re
import tempfile
from typing import Any

import yaml

try:
    _Loader = yaml.CSafeLoader
    _Dumper = yaml.CSafeDumper
except AttributeError:
    _Loader = yaml.SafeLoader  # type: ignore[assignment]
    _Dumper = yaml.SafeDumper  # type: ignore[assignment]

_VALID_STAGE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class StageLock:
    """Manages lock file for a single pipeline stage."""

    def __init__(self, stage_name: str, cache_dir: pathlib.Path) -> None:
        if not stage_name or not _VALID_STAGE_NAME.match(stage_name):
            raise ValueError(f"Invalid stage name: {stage_name!r}")
        self.stage_name = stage_name
        self.path = cache_dir / "stages" / f"{stage_name}.lock"

    def read(self) -> dict[str, Any] | None:
        """Read lock file, return None if missing or corrupted."""
        try:
            with open(self.path) as f:
                data = yaml.load(f, Loader=_Loader)
            if data is None:
                return None
            if not isinstance(data, dict):
                return None  # Treat corrupted file as missing
            return data
        except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError):
            return None

    def write(self, data: dict[str, Any]) -> None:
        """Write lock file atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        tmp = pathlib.Path(tmp_path)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f, Dumper=_Dumper, sort_keys=False)
            tmp.replace(self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def is_changed(
        self,
        current_fingerprint: dict[str, str],
        current_params: dict[str, Any],
        dep_hashes: dict[str, str],
    ) -> tuple[bool, str]:
        """Check if stage needs re-run."""
        lock_data = self.read()
        if not lock_data:
            return True, "No previous run"

        checks = [
            ("code_manifest", current_fingerprint, "Code changed"),
            ("params", current_params, "Params changed"),
            ("dep_hashes", dep_hashes, "Input dependencies changed"),
        ]
        for key, current, reason in checks:
            if (lock_data.get(key) or {}) != current:
                return True, reason

        return False, ""
