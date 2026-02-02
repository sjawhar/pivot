---
tags: [python, paths, data-structures, trie]
category: implementation
module: trie
symptoms: ["false positive path overlaps", "directory vs file path confusion", "incorrect dependency detection"]
---

# Path Overlap Detection Requires Trie, Not String Matching

## Problem

When detecting overlapping output paths (e.g., one stage outputs a directory `data/` while another outputs a file `data/file.csv`), naive string prefix matching produces incorrect results:

```python
# Naive approach - WRONG
def paths_overlap(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)

paths_overlap("data/train.csv", "data/")  # True - correct
paths_overlap("data/train.csv", "data/t")  # True - FALSE POSITIVE!
```

The problem: `"data/train.csv".startswith("data/t")` is `True`, but `data/t` is not a parent directory of `data/train.csv`. String matching doesn't understand path component boundaries.

Similarly, `data-backup/` would falsely overlap with `data/` because `"data-backup/".startswith("data")` is `True` at the character level.

## Solution

Use pygtrie's `Trie` with path components (from `Path.parts`) as keys. The trie naturally handles component boundaries because each path segment is a separate key element:

```python
import pathlib
from pygtrie import Trie

def build_output_trie(outputs: list[tuple[str, str]]) -> Trie[tuple[str, str]]:
    """Build trie mapping output paths to (stage_name, path) tuples."""
    trie: Trie[tuple[str, str]] = Trie()

    for stage_name, path in outputs:
        key = pathlib.Path(path).parts  # ("data", "train.csv")

        # Check for exact duplicate
        if key in trie:
            existing_stage, _ = trie[key]
            raise OutputDuplicationError(f"'{path}' produced by both '{stage_name}' and '{existing_stage}'")

        # Check if new output is parent of existing outputs
        if trie.has_subtrie(key):
            child_stage, child_path = next(iter(trie.values(prefix=key)))
            raise OverlappingOutputPathsError(f"'{path}' overlaps with child '{child_path}'")

        # Check if new output is child of existing output
        prefix_item = trie.shortest_prefix(key)
        if prefix_item is not None and prefix_item.value is not None:
            parent_stage, parent_path = prefix_item.value
            raise OverlappingOutputPathsError(f"'{path}' overlaps with parent '{parent_path}'")

        trie[key] = (stage_name, path)

    return trie
```

With path components as keys:
- `("data", "train.csv")` does NOT have `("data", "t")` as a prefix
- `("data-backup",)` does NOT have `("data",)` as a prefix
- `("data", "file.csv")` DOES have `("data",)` as a prefix (correct overlap detection)

## Key Insight

Path relationships are defined by **component boundaries**, not character positions. A trie with `Path.parts` as keys inherently enforces component-level matching because each path segment becomes a distinct node in the trie structure.

The pygtrie library provides efficient O(k) operations where k is the number of path components:
- `has_subtrie(key)` - check if key is a prefix of any stored path
- `shortest_prefix(key)` - find if any stored path is a prefix of key
- Standard dict operations for exact matches

This is used in Pivot's `src/pivot/trie.py` to validate that stage outputs don't overlap (preventing one stage from clobbering another's outputs).
