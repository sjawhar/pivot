"""Model-based test of the skip contract: repro output always equals a from-scratch run.

A small three-stage pipeline (raw input -> scale -> offset -> total, total also reading scale)
runs in a temporary project through ``executor.run``. Rules edit the raw input, edit/revert a
param that changes scale's output, edit a param that does not, hand-edit or delete a stage's
output file, and repro. The model computes every expected output itself.

Invariants after each repro:
- every output file holds the value a from-scratch run on the current input and params produces;
- every stage repro reports as ``ran`` was predicted stale by ``status`` just before (status may
  also mark stages that repro then restores from the run cache, explain.py:128-130);
- a stage whose inputs hold the same bytes as at the previous repro stays ``cached``.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import hypothesis
import yaml
from hypothesis import stateful
from hypothesis import strategies as st

from pivot import compose, executor, project, stage_def, status
from pivot.config import io as config_io
from pivot.storage import store as store_mod
from pivot.types import PipelineStatus


class ScaleParams(stage_def.StageParams):
    factor: int = 2
    label: str = "a"


@compose.stage
def scale(numbers: str, params: ScaleParams) -> str:
    return str(int(numbers) * params.factor)


@compose.stage
def offset(scaled: str) -> str:
    return str(int(scaled) + 1)


@compose.stage
def total(scaled: str, offsetted: str) -> str:
    return str(int(scaled) + int(offsetted))


_STAGES = ("test/scale", "test/offset", "test/total")
_DEPS = {
    "test/scale": (),
    "test/offset": ("test/scale",),
    "test/total": ("test/scale", "test/offset"),
}


def _expected(n: int, factor: int) -> dict[str, str]:
    scaled = n * factor
    return {
        "test/scale": str(scaled),
        "test/offset": str(scaled + 1),
        "test/total": str(scaled + scaled + 1),
    }


class SkipContractMachine(stateful.RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()
        (self.root / ".pivot").mkdir()
        (self.root / ".git").mkdir()
        (self.root / "data" / "raw").mkdir(parents=True)
        self._saved_cwd = os.getcwd()
        self._saved_root = project._project_root_cache
        os.chdir(self.root)
        project._project_root_cache = self.root
        config_io.clear_config_cache()

        with compose.Pipeline("test", root=self.root) as pipeline:
            scaled = scale(pipeline.input("numbers.txt"), params=ScaleParams())
            offsetted = offset(scaled)
            total(scaled, offsetted)
        self.pipeline = pipeline
        workspace = store_mod.WorkspaceStore(
            project_root=self.root,
            pipeline_name=pipeline.name,
            input_bindings=pipeline.input_bindings,
        )
        self.output_paths = {
            name: workspace.resolve_display_path(pipeline.get_stage(name)["outs"][0])
            for name in _STAGES
        }

        self.n = 1
        self.factor = 2
        self.label = "a"
        self.factor_history = list[int]()
        self.reproduced = False
        self.last_values = dict[str, str]()
        self._write_input()
        self._write_params()

    def _write_input(self) -> None:
        (self.root / "data" / "raw" / "numbers.txt").write_text(str(self.n))

    def _write_params(self) -> None:
        overrides = {"test/scale": {"factor": self.factor, "label": self.label}}
        (self.root / "params.yaml").write_text(yaml.safe_dump(overrides))

    @stateful.rule(n=st.integers(min_value=0, max_value=3))
    def edit_raw_input(self, n: int) -> None:
        self.n = n
        self._write_input()

    @stateful.rule(factor=st.integers(min_value=1, max_value=3))
    def edit_param(self, factor: int) -> None:
        self.factor_history.append(self.factor)
        self.factor = factor
        self._write_params()

    @stateful.precondition(lambda self: bool(self.factor_history))
    @stateful.rule()
    def revert_param(self) -> None:
        self.factor = self.factor_history.pop()
        self._write_params()

    @stateful.rule(label=st.sampled_from(["a", "b"]))
    def edit_output_neutral_param(self, label: str) -> None:
        self.label = label
        self._write_params()

    @stateful.precondition(lambda self: self.reproduced)
    @stateful.rule(stage=st.sampled_from(_STAGES))
    def hand_edit_output(self, stage: str) -> None:
        path = self.output_paths[stage]
        if path.exists():
            path.unlink()  # outputs may be links into the cache; never write through them
        path.write_text("999999")

    @stateful.precondition(lambda self: self.reproduced)
    @stateful.rule(stage=st.sampled_from(_STAGES))
    def delete_output(self, stage: str) -> None:
        self.output_paths[stage].unlink(missing_ok=True)

    @stateful.rule()
    def repro(self) -> None:
        predicted, _ = status.get_pipeline_status(
            None,
            single_stage=False,
            all_stages=self.pipeline.snapshot(),
            pipeline=self.pipeline,
        )
        stale = {info["name"] for info in predicted if info["status"] == PipelineStatus.STALE}

        results = executor.run(pipeline=self.pipeline)

        expected = _expected(self.n, self.factor)
        for name in _STAGES:
            assert results[name]["status"] in ("ran", "cached"), (name, results[name])
            actual = self.output_paths[name].read_text()
            assert actual == expected[name], (
                f"{name} holds {actual!r}, a from-scratch run gives {expected[name]!r} "
                + f"(results: {dict(results)})"
            )
            if results[name]["status"] == "ran":
                assert name in stale, f"{name} ran but status predicted only {sorted(stale)}"
            if (
                self.reproduced
                and _DEPS[name]
                and all(expected[dep] == self.last_values[dep] for dep in _DEPS[name])
            ):
                assert results[name]["status"] == "cached", (
                    f"{name} reran although its inputs kept their bytes: {dict(results)}"
                )
        self.last_values = expected
        self.reproduced = True

    def teardown(self) -> None:
        os.chdir(self._saved_cwd)
        project._project_root_cache = self._saved_root
        config_io.clear_config_cache()
        self._tmp.cleanup()


_MACHINE_SETTINGS = hypothesis.settings(
    max_examples=6,
    stateful_step_count=10,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)


def test_repro_matches_from_scratch_run_under_random_edits() -> None:
    stateful.run_state_machine_as_test(SkipContractMachine, settings=_MACHINE_SETTINGS)


def _replay(steps: list[tuple[str, dict[str, object]]]) -> None:
    machine = SkipContractMachine()
    try:
        for rule, kwargs in steps:
            getattr(machine, rule)(**kwargs)
    finally:
        machine.teardown()


def test_raw_input_revisited_after_change_restores_downstream_values() -> None:
    """A -> B -> A on the raw input: the run-cache restore of scale must reach its consumers."""
    _replay(
        [
            ("edit_raw_input", {"n": 1}),
            ("repro", {}),
            ("edit_raw_input", {"n": 2}),
            ("repro", {}),
            ("edit_raw_input", {"n": 1}),
            ("repro", {}),
        ]
    )


def test_output_neutral_param_revisited_keeps_consumers_cached() -> None:
    """A -> B -> A on a param scale ignores: offset and total never rerun."""
    _replay(
        [
            ("repro", {}),
            ("edit_output_neutral_param", {"label": "b"}),
            ("repro", {}),
            ("edit_output_neutral_param", {"label": "a"}),
            ("repro", {}),
        ]
    )


def test_no_op_run_cache_restore_keeps_consumer_on_generation_fast_path() -> None:
    """A run-cache restore that writes no bytes must not cost the consumer its O(1) skip.

    The sequence makes offset record scale's current generation through a run-cache restore
    (n back to 1), then flips a param scale ignores to a value whose run-cache entry holds the
    bytes already in the workspace. That restore writes nothing, so offset must still skip on
    the generation tier, not fall through to rehashing its dependencies.
    """
    machine = SkipContractMachine()
    try:
        machine.repro()
        machine.edit_output_neutral_param(label="b")
        machine.repro()
        machine.edit_raw_input(n=2)
        machine.repro()
        machine.edit_raw_input(n=1)
        machine.repro()
        machine.edit_output_neutral_param(label="a")
        results = executor.run(pipeline=machine.pipeline)
    finally:
        machine.teardown()

    assert results["test/scale"]["reason"] == "unchanged (run cache)"
    assert results["test/offset"]["status"] == "cached"
    assert results["test/offset"]["reason"] == "unchanged (generation)", dict(results)
