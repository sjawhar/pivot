"""What ``pivot status`` reports for a stage mapped over a directory output (``Pipeline.map``).

In a fresh process no template is expanded, so the lock, listing and restart-proof files of its
instances belong to no registered stage. They are not orphaned lock files: a run expands the
template again, and removes an instance whose key left the plan at expansion; ``pivot status``
lists such an instance as orphaned (``tests/cli/test_map_removal.py``).
"""

import pathlib
from typing import Annotated

from pivot import compose, loaders, status


@compose.stage
def _helper_plan(
    raw: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], compose.directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in raw.items()}


@compose.stage
def _helper_cut(entry: dict[str, int]) -> dict[str, int]:
    return entry


def test_an_unexpanded_instances_files_are_not_orphans(set_project_root: pathlib.Path) -> None:
    root = set_project_root
    with compose.Pipeline("p", root=root) as pipeline:
        plan = _helper_plan(pipeline.input("plan.yaml", t=dict[str, int]))
        _helper_cut(pipeline.map(plan))
    stages_dir = root / ".pivot" / "stages" / "p"
    stages_dir.mkdir(parents=True)
    for name in ("_helper_cut@a.lock", "_helper_cut@a=.listing", "_helper_cut@a.incremental"):
        (stages_dir / name).write_text("")
    (stages_dir / "gone.lock").write_text("")
    all_stages = {name: pipeline.get_stage(name) for name in pipeline.list_stages()}

    orphaned = status.find_orphaned_lock_files(all_stages, pipeline)

    assert orphaned == ["p/gone"], orphaned
