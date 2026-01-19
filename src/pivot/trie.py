import pathlib
from typing import TypedDict

from pygtrie import Trie

from pivot import exceptions


class TrieStageInfo(TypedDict):
    """Minimal stage info for trie output overlap detection."""

    name: str
    outs: list[str]


def build_outs_trie(stages: dict[str, TrieStageInfo]) -> Trie[tuple[str, str]]:
    """Build trie of output paths from stages.

    Args:
        stages: Dict of stage_name -> stage_info (from registry)

    Returns:
        Trie mapping output paths to (stage_name, output_path) tuples

    Raises:
        OutputDuplicationError: If two stages produce exact same output
        OverlappingOutputPathsError: If output paths overlap

    Example:
        >>> stages = {
        ...     'a': {'outs': ['/project/data/train.csv']},
        ...     'b': {'outs': ['/project/data/test.csv']}
        ... }
        >>> trie = build_outs_trie(stages)
        >>> # Later, check if path overlaps with existing outputs
        >>> trie.has_subtrie(pathlib.Path('/project/data/train.csv').parts)
    """
    outs: Trie[tuple[str, str]] = Trie()

    for stage_name, stage_info in stages.items():
        for out in stage_info["outs"]:
            out_key = pathlib.Path(out).parts

            if out_key in outs:
                existing_stage, _ = outs[out_key]
                raise exceptions.OutputDuplicationError(
                    f"Output '{out}' is produced by both '{stage_name}' and '{existing_stage}'"
                )

            # Case 1: New output is parent of existing output(s)
            if outs.has_subtrie(out_key):
                child_stage, child_path = next(iter(outs.values(prefix=out_key)))
                raise exceptions.OverlappingOutputPathsError(
                    "Output paths overlap:\n"
                    + f"  '{out}' (stage '{stage_name}')\n"
                    + f"  '{child_path}' (stage '{child_stage}')\n"
                    + "One is a parent directory of the other."
                )

            # Case 2: New output is child of existing output
            prefix_item = outs.shortest_prefix(out_key)
            if prefix_item is not None and prefix_item.value is not None:
                parent_stage, parent_path = prefix_item.value
                raise exceptions.OverlappingOutputPathsError(
                    "Output paths overlap:\n"
                    + f"  '{parent_path}' (stage '{parent_stage}')\n"
                    + f"  '{out}' (stage '{stage_name}')\n"
                    + "One is a parent directory of the other."
                )

            outs[out_key] = (stage_name, out)

    return outs
