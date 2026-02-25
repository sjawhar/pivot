from __future__ import annotations

import pytest

from pivot import exceptions, trie
from pivot.trie import TrieStageInfo


def test_build_outs_trie_happy_path() -> None:
    stages = {
        "prepare": TrieStageInfo(name="prepare", outs=["/project/data/train.csv"]),
        "eval": TrieStageInfo(name="eval", outs=["/project/data/test.csv"]),
    }

    result = trie.build_outs_trie(stages)

    assert ("/", "project", "data", "train.csv") in result
    assert ("/", "project", "data", "test.csv") in result


def test_build_outs_trie_detects_duplicate_output() -> None:
    stages = {
        "a": TrieStageInfo(name="a", outs=["/project/data/train.csv"]),
        "b": TrieStageInfo(name="b", outs=["/project/data/train.csv"]),
    }

    with pytest.raises(exceptions.OutputDuplicationError, match="produced by both"):
        trie.build_outs_trie(stages)


def test_build_outs_trie_detects_parent_child_overlap_new_parent() -> None:
    stages = {
        "a": TrieStageInfo(name="a", outs=["/project/data/train/model.pkl"]),
        "b": TrieStageInfo(name="b", outs=["/project/data/train"]),
    }

    with pytest.raises(exceptions.OverlappingOutputPathsError, match="parent directory"):
        trie.build_outs_trie(stages)


def test_build_outs_trie_detects_parent_child_overlap_new_child() -> None:
    stages = {
        "a": TrieStageInfo(name="a", outs=["/project/data/train"]),
        "b": TrieStageInfo(name="b", outs=["/project/data/train/model.pkl"]),
    }

    with pytest.raises(exceptions.OverlappingOutputPathsError, match="parent directory"):
        trie.build_outs_trie(stages)
