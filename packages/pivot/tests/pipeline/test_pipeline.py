from __future__ import annotations

from typing import TYPE_CHECKING

from pivot.pipeline import pipeline as pipeline_mod

if TYPE_CHECKING:
    import pathlib


def test_include_merges_input_bindings(tmp_path: pathlib.Path) -> None:
    child = pipeline_mod.Pipeline("child", root=tmp_path)
    child.set_input_bindings({"ext.jsonl": "data/external/ext.jsonl"})

    parent = pipeline_mod.Pipeline("parent", root=tmp_path)
    parent.include(child)

    assert parent.input_bindings["ext.jsonl"] == "data/external/ext.jsonl"
