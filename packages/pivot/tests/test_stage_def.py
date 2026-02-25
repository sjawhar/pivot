from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, cast

import pytest

from pivot import outputs, stage_def


@dataclasses.dataclass
class _RecorderWriter:
    writes: list[tuple[pathlib.Path, Any]]

    def save(self, data: Any, path: pathlib.Path) -> None:
        self.writes.append((path, data))
        path.write_text(str(data))


def test_validate_path_not_escaped_rejects_traversal(tmp_path: pathlib.Path) -> None:
    safe = tmp_path / "safe" / "file.txt"
    stage_def._validate_path_not_escaped(safe, tmp_path)  # noqa: SLF001

    with pytest.raises(ValueError, match="escapes project root"):
        stage_def._validate_path_not_escaped(pathlib.Path("/tmp/outside.txt"), tmp_path)  # noqa: SLF001


@pytest.mark.parametrize(
    ("key", "message"),
    [
        pytest.param("", "empty or whitespace-only", id="empty"),
        pytest.param("   ", "empty or whitespace-only", id="whitespace"),
        pytest.param("/abs.json", "absolute path not allowed", id="absolute"),
        pytest.param("../up.json", "path traversal not allowed", id="traversal"),
        pytest.param("no_extension", "must include file extension", id="missing_extension"),
        pytest.param("   .json", "filename cannot be empty", id="blank_stem"),
    ],
)
def test_validate_directory_out_key_rejects_invalid_inputs(key: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stage_def._validate_directory_out_key(key, "out")  # noqa: SLF001


def test_validate_directory_out_key_normalizes_valid_values() -> None:
    normalized = stage_def._validate_directory_out_key("a//b.json", "out")  # noqa: SLF001
    assert normalized == "a/b.json"


def test_collect_directory_out_ops_validates_collisions_and_types(tmp_path: pathlib.Path) -> None:
    writer = _RecorderWriter([])
    spec = outputs.DirectoryOut("results/", cast("Any", writer))
    write_ops: list[tuple[pathlib.Path, Any, Any]] = []

    with pytest.raises(RuntimeError, match="expects dict"):
        stage_def._collect_directory_out_ops("d", spec, 123, tmp_path, write_ops)  # noqa: SLF001
    with pytest.raises(ValueError, match="must be non-empty"):
        stage_def._collect_directory_out_ops("d", spec, {}, tmp_path, write_ops)  # noqa: SLF001
    with pytest.raises(ValueError, match="keys must be strings"):
        stage_def._collect_directory_out_ops("d", spec, {1: "x"}, tmp_path, write_ops)  # noqa: SLF001
    with pytest.raises(ValueError, match="duplicate key after normalization"):
        stage_def._collect_directory_out_ops(  # noqa: SLF001
            "d",
            spec,
            {"a//b.json": "x", "a/b.json": "y"},
            tmp_path,
            write_ops,
        )
    with pytest.raises(ValueError, match="case-insensitive filesystems"):
        stage_def._collect_directory_out_ops(  # noqa: SLF001
            "d",
            spec,
            {"A.json": "x", "a.json": "y"},
            tmp_path,
            write_ops,
        )


def test_save_return_outputs_validates_required_keys_and_shapes(tmp_path: pathlib.Path) -> None:
    writer = _RecorderWriter([])
    specs = {
        "single": outputs.Out("single.txt", cast("Any", writer)),
        "multi": outputs.Out(["a.txt", "b.txt"], cast("Any", writer)),
    }

    with pytest.raises(KeyError, match="Missing return output keys"):
        stage_def.save_return_outputs({"single": "x"}, specs, tmp_path)

    with pytest.raises(RuntimeError, match="sequence path but non-sequence value"):
        stage_def.save_return_outputs({"single": "x", "multi": "bad"}, specs, tmp_path)

    with pytest.raises(RuntimeError, match="2 paths but 1 values"):
        stage_def.save_return_outputs({"single": "x", "multi": ["one"]}, specs, tmp_path)


def test_save_return_outputs_writes_single_sequence_and_directory(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = _RecorderWriter([])
    specs = {
        "single": outputs.Out("single.txt", cast("Any", writer)),
        "multi": outputs.Out(["m1.txt", "m2.txt"], cast("Any", writer)),
        "dir": outputs.DirectoryOut("dir/", cast("Any", writer)),
    }
    return_value = {
        "single": "one",
        "multi": ["a", "b"],
        "dir": {"nested/x.json": {"v": 1}},
        "extra": "ignored",
    }

    stage_def.save_return_outputs(return_value, specs, tmp_path)

    assert any("Extra keys in return value" in rec.message for rec in caplog.records)
    written_paths = {path.relative_to(tmp_path).as_posix() for path, _ in writer.writes}
    assert written_paths == {"single.txt", "m1.txt", "m2.txt", "dir/nested/x.json"}
    assert (tmp_path / "dir" / "nested" / "x.json").read_text() == "{'v': 1}"
