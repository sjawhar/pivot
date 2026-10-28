"""The pull-request benchmark must run when it should and measure the pull request's own change.

The bench tests run the workflow's own bench steps, up to and including the base checkout, against
a local origin. actions/checkout is simulated with git, and `${{ }}` expressions are evaluated in a
pull_request context like the one GitHub builds: `github.sha` is the merge commit of
`refs/pull/N/merge`, the commit whose workflow file GitHub runs. The trigger test evaluates the
jobs' `if`s and the concurrency group for each kind of event.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
from typing import Any, cast

import pytest
import yaml

_WORKFLOW = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "benchmark.yaml"
_GIT_ENV = {
    "PATH": os.environ["PATH"],
    "HOME": os.environ.get("HOME", "/"),
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "bench",
    "GIT_AUTHOR_EMAIL": "bench@example.com",
    "GIT_COMMITTER_NAME": "bench",
    "GIT_COMMITTER_EMAIL": "bench@example.com",
}
_TOKEN = re.compile(r"\s*('(?:[^']|'')*'|\|\||&&|==|!=|\(|\)|,|-?\d+(?:\.\d+)?|[A-Za-z_][\w.*-]*)")
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_CHECKOUT_PATH = re.compile(r"(?:^|\s)((?:head|base)/\S+)")


def _helper_git(cwd: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=_GIT_ENV, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _helper_commit(repo: pathlib.Path, name: str, files: tuple[str, ...] = ()) -> str:
    for path in files or (f"{name}.txt",):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(name)
        _helper_git(repo, "add", path)
    _helper_git(repo, "commit", "-q", "-m", name)
    return _helper_git(repo, "rev-parse", "HEAD")


def _helper_lookup(context: dict[str, Any], path: str) -> object:
    """Resolve a property path. A `*` part maps the rest of the path over an array, as GitHub's
    object filter does: `labels.*.name` is the list of the label names."""
    values: list[object] = [context]
    mapped = False
    for part in path.split("."):
        if part == "*":
            mapped = True
            values = [
                item
                for value in values
                if isinstance(value, list)
                for item in cast("list[object]", value)
            ]
            continue
        values = [
            cast("dict[str, object]", value).get(part) if isinstance(value, dict) else None
            for value in values
        ]
    return values if mapped else values[0]


def _helper_equal(left: object, right: object) -> bool:
    """GitHub's `==`: strings compare ignoring case."""
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    return left == right


def _helper_contains(haystack: object, needle: object) -> bool:
    """GitHub's `contains()`: an element of an array, or else a substring, ignoring case."""
    if isinstance(haystack, list):
        return any(_helper_equal(item, needle) for item in cast("list[object]", haystack))
    return str(needle).casefold() in ("" if haystack is None else str(haystack)).casefold()


def _helper_evaluate_expression(text: str, context: dict[str, Any]) -> object:
    """Evaluate the `||`, `&&`, `==`, `!=`, `contains()`, literal and property subset of GitHub
    expressions: `&&` binds tighter than `||`, and each returns one of its operands."""
    tokens: list[str] = _TOKEN.findall(text)
    if "".join(tokens).replace(" ", "") != re.sub(r"\s", "", text):
        raise NotImplementedError(f"unsupported expression: {text!r}")
    position = 0

    def take(expected: str) -> None:
        nonlocal position
        if position >= len(tokens) or tokens[position] != expected:
            raise NotImplementedError(f"expected {expected!r} at token {position} of {text!r}")
        position += 1

    def primary() -> object:
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "(":
            value = either()
            take(")")
            return value
        if token.startswith("'"):
            return token[1:-1].replace("''", "'")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            return float(token) if "." in token else int(token)
        if token in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[token]
        if position < len(tokens) and tokens[position] == "(":
            if token != "contains":
                raise NotImplementedError(f"function {token}() in {text!r}")
            take("(")
            haystack = either()
            take(",")
            needle = either()
            take(")")
            return _helper_contains(haystack, needle)
        return _helper_lookup(context, token)

    def comparison() -> object:
        nonlocal position
        left = primary()
        if position < len(tokens) and tokens[position] in ("==", "!="):
            operator = tokens[position]
            position += 1
            right = primary()
            return _helper_equal(left, right) == (operator == "==")
        return left

    def both() -> object:
        nonlocal position
        value = comparison()
        while position < len(tokens) and tokens[position] == "&&":
            position += 1
            right = comparison()
            value = right if value else value
        return value

    def either() -> object:
        nonlocal position
        value = both()
        while position < len(tokens) and tokens[position] == "||":
            position += 1
            right = both()
            value = value if value else right
        return value

    value = either()
    if position != len(tokens):
        raise NotImplementedError(f"unparsed {tokens[position:]} in {text!r}")
    return value


def _helper_evaluate(value: object, context: dict[str, Any]) -> object:
    if not isinstance(value, str):
        return value
    # One expression that is the whole value keeps its type; `fullmatch` alone would also take
    # `${{ a }}-${{ b }}` as one expression, `a }}-${{ b`.
    whole = _EXPRESSION.fullmatch(value.strip())
    if whole and "${{" not in whole.group(1):
        return _helper_evaluate_expression(whole.group(1).strip(), context)

    def text(match: re.Match[str]) -> str:
        result = _helper_evaluate_expression(match.group(1).strip(), context)
        return "" if result is None else str(result)

    return _EXPRESSION.sub(text, value)


def _helper_checkout(origin: pathlib.Path, path: pathlib.Path, ref: str) -> None:
    """What actions/checkout leaves: a clone with every branch, at `ref`, detached."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _helper_git(path.parent, "clone", "-q", "--no-checkout", str(origin), str(path))
    _helper_git(path, "fetch", "-q", "origin", "+refs/pull/*:refs/remotes/pull/*")
    if ref.startswith("refs/pull/"):
        target = "refs/remotes/pull/" + ref.removeprefix("refs/pull/")
    elif re.fullmatch(r"[0-9a-f]{40}", ref):
        target = ref
    else:
        target = "origin/" + ref.removeprefix("refs/heads/")
    _helper_git(path, "checkout", "-q", "--force", "--detach", target)


def _helper_outputs(path: pathlib.Path) -> dict[str, str]:
    outputs = dict[str, str]()
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            outputs[key] = value
    return outputs


def _helper_bench_steps() -> list[dict[str, Any]]:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    return workflow["jobs"]["bench"]["steps"]


def _helper_run_bench_until_base_checkout(
    origin: pathlib.Path, workspace: pathlib.Path, context: dict[str, Any]
) -> tuple[str, str]:
    """Run the bench job's steps through its base checkout; return (base, head) commits."""
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    env: dict[str, object] = workflow["env"]
    context["env"] = {key: str(_helper_evaluate(value, context)) for key, value in env.items()}
    for index, step in enumerate(_helper_bench_steps()):
        if "if" in step and not _helper_evaluate(step["if"], context):
            continue
        if "uses" in step:
            if not str(step["uses"]).startswith("actions/checkout@"):
                continue
            with_ = {k: _helper_evaluate(v, context) for k, v in step.get("with", {}).items()}
            path = str(with_.get("path") or "")
            ref = str(with_.get("ref") or context["github"]["sha"])
            _helper_checkout(origin, workspace / path, ref)
            if path == "base":
                base = _helper_git(workspace / "base", "rev-parse", "HEAD")
                head = _helper_git(workspace / "head", "rev-parse", "HEAD")
                return base, head
            continue
        output = workspace.parent / f"output-{index}"
        output.touch()
        step_env = {
            k: str(_helper_evaluate(v, context) or "") for k, v in step.get("env", {}).items()
        }
        script = workspace.parent / f"step-{index}.sh"
        script.write_text(str(_helper_evaluate(step["run"], context)))
        subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
            cwd=workspace / str(_helper_evaluate(step.get("working-directory", ""), context)),
            env={
                **_GIT_ENV,
                **context["env"],
                **step_env,
                "GITHUB_OUTPUT": str(output),
                "GITHUB_WORKSPACE": str(workspace),
                "RUNNER_TEMP": str(workspace.parent / "runner-temp"),
            },
            check=True,
        )
        if "id" in step:
            context["steps"][step["id"]] = {"outputs": _helper_outputs(output)}
    raise AssertionError("the bench job never checks out the base")


def _helper_open_pull_request(origin: pathlib.Path, payload_base: str) -> dict[str, Any]:
    """Build refs/pull/1/merge (branch `pr` merged into `main`) and return the event context."""
    pr_head = _helper_git(origin, "rev-parse", "pr")
    _helper_git(origin, "checkout", "-q", "--detach", "main")
    _helper_git(origin, "merge", "-q", "--no-ff", "-m", "merge pr", "pr")
    merge_ref = _helper_git(origin, "rev-parse", "HEAD")
    _helper_git(origin, "update-ref", "refs/pull/1/merge", merge_ref)
    _helper_git(origin, "checkout", "-q", "main")
    return {
        "github": {
            "event_name": "pull_request",
            "sha": merge_ref,
            "ref": "refs/pull/1/merge",
            "base_ref": "main",
            "head_ref": "pr",
            "event": {
                "number": 1,
                "pull_request": {
                    "number": 1,
                    "head": {"sha": pr_head, "ref": "pr"},
                    "base": {"sha": payload_base, "ref": "main"},
                },
            },
        },
        "inputs": {},
        "steps": {},
        "matrix": {"project": "chain-10"},
    }


def _helper_init(origin: pathlib.Path) -> str:
    origin.mkdir()
    _helper_git(origin, "init", "-q", "-b", "main")
    return _helper_commit(origin, "c0")


def _helper_pull_request_event(
    action: str,
    *labels: str,
    added: str | None = None,
    draft: bool = False,
    run_drafts: str | None = None,
) -> dict[str, Any]:
    """A `pull_request` run of pull request 1: its labels after the event and, for `labeled`, the
    label the event added. `run_drafts` is the repository variable CI_RUN_DRAFT_PRS."""
    event: dict[str, Any] = {
        "action": action,
        "pull_request": {"number": 1, "draft": draft, "labels": [{"name": n} for n in labels]},
    }
    if added is not None:
        event["label"] = {"name": added}
    return {
        "github": {
            "event_name": "pull_request",
            "event": event,
            "workflow": "Benchmark",
            "ref": "refs/pull/1/merge",
            "run_id": "999",
        },
        "vars": {} if run_drafts is None else {"CI_RUN_DRAFT_PRS": run_drafts},
    }


def _helper_other_event(event_name: str) -> dict[str, Any]:
    return {
        "github": {
            "event_name": event_name,
            "event": {},
            "workflow": "Benchmark",
            "ref": "refs/heads/main",
            "run_id": "999",
        },
        "vars": {},
    }


def _helper_job_runs(job: dict[str, Any], context: dict[str, Any]) -> bool:
    """A job's `if`, which GitHub evaluates as an expression with or without `${{ }}`."""
    condition = str(job.get("if", "true")).strip()
    whole = _EXPRESSION.fullmatch(condition)
    expression = whole.group(1).strip() if whole else condition
    return bool(_helper_evaluate_expression(expression, context))


@pytest.mark.parametrize(
    "main_moved",
    [
        pytest.param(False, id="base-is-the-branch-point"),
        # The base branch gained a commit the head does not have, and the pull request's
        # base.sha names that commit rather than the head's merge base.
        pytest.param(True, id="base-is-ahead-of-the-head"),
    ],
)
def test_pull_request_bench_measures_only_the_pull_requests_commits(
    tmp_path: pathlib.Path, main_moved: bool
) -> None:
    origin = tmp_path / "origin"
    payload_base = _helper_init(origin)
    _helper_git(origin, "checkout", "-q", "-b", "pr")
    _helper_commit(origin, "p1")
    if main_moved:
        _helper_git(origin, "checkout", "-q", "main")
        payload_base = _helper_commit(origin, "m1")
    context = _helper_open_pull_request(origin, payload_base)

    base, head = _helper_run_bench_until_base_checkout(origin, tmp_path / "workspace", context)

    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, head], cwd=origin, env=_GIT_ENV
    )
    assert is_ancestor.returncode == 0, (
        f"base {base[:8]} is not an ancestor of head {head[:8]}: the base side's commits "
        + f"{_helper_git(origin, 'log', '--format=%s', f'{head}..{base}').split()} would be "
        + "measured as the pull request's change, with the opposite sign"
    )
    measured = set(_helper_git(origin, "rev-list", f"{base}..{head}").split())
    own = set(_helper_git(origin, "rev-list", "main..pr").split()) | {context["github"]["sha"]}
    foreign = [_helper_git(origin, "log", "-1", "--format=%s", sha) for sha in measured - own]
    assert not foreign, f"measured as the pull request's change, but not its commits: {foreign}"


def test_pull_request_bench_runs_a_harness_its_checkout_contains(tmp_path: pathlib.Path) -> None:
    # The pull request branched before the harness reached main. GitHub runs the workflow of
    # the merge commit, which has the bench job, but the head checkout has no harness.
    origin = tmp_path / "origin"
    branch_point = _helper_init(origin)
    _helper_git(origin, "checkout", "-q", "-b", "pr")
    _helper_commit(origin, "p1")
    _helper_git(origin, "checkout", "-q", "main")
    harness = ("benchmarks/run.py", "benchmarks/compare.py", "benchmarks/threshold.json")
    _helper_commit(origin, "add the benchmark harness", harness)
    context = _helper_open_pull_request(origin, branch_point)
    workspace = tmp_path / "workspace"

    _helper_run_bench_until_base_checkout(origin, workspace, context)

    missing = list[str]()
    for step in _helper_bench_steps():
        if "run" not in step:
            continue
        cwd = workspace / str(_helper_evaluate(step.get("working-directory", ""), context))
        text = str(_helper_evaluate(step["run"], context))
        missing += [path for path in _CHECKOUT_PATH.findall(text) if not (cwd / path).is_file()]
    assert not missing, f"the bench job runs files its checkouts do not contain: {missing}"


@pytest.mark.parametrize(
    ("context", "runs"),
    [
        pytest.param(_helper_pull_request_event("opened"), False, id="opened-no-label"),
        pytest.param(_helper_pull_request_event("opened", "benchmark"), True, id="opened-labelled"),
        pytest.param(_helper_pull_request_event("synchronize"), False, id="synchronize-no-label"),
        pytest.param(
            _helper_pull_request_event("synchronize", "benchmark", "bug"),
            True,
            id="synchronize-labelled",
        ),
        # GitHub's contains() ignores case.
        pytest.param(
            _helper_pull_request_event("synchronize", "Benchmark"),
            True,
            id="synchronize-label-in-capitals",
        ),
        pytest.param(_helper_pull_request_event("reopened"), False, id="reopened-no-label"),
        pytest.param(
            _helper_pull_request_event("reopened", "benchmark"), True, id="reopened-labelled"
        ),
        pytest.param(
            _helper_pull_request_event("ready_for_review"), False, id="ready-for-review-no-label"
        ),
        pytest.param(
            _helper_pull_request_event("ready_for_review", "benchmark"),
            True,
            id="ready-for-review-labelled",
        ),
        pytest.param(
            _helper_pull_request_event("labeled", "benchmark", added="benchmark"),
            True,
            id="labeled-benchmark",
        ),
        # Adding an unrelated label while the benchmark runs must not cancel it.
        pytest.param(
            _helper_pull_request_event("labeled", "benchmark", "bug", added="bug"),
            False,
            id="labeled-other-label-with-benchmark",
        ),
        pytest.param(
            _helper_pull_request_event("labeled", "bug", added="bug"),
            False,
            id="labeled-other-label-without-benchmark",
        ),
        pytest.param(
            _helper_pull_request_event("synchronize", "benchmark", draft=True),
            False,
            id="draft-labelled-drafts-variable-unset",
        ),
        pytest.param(
            _helper_pull_request_event("labeled", "benchmark", added="benchmark", draft=True),
            False,
            id="draft-labeled-benchmark-drafts-variable-unset",
        ),
        pytest.param(
            _helper_pull_request_event("synchronize", "benchmark", draft=True, run_drafts="false"),
            False,
            id="draft-labelled-drafts-variable-false",
        ),
        pytest.param(
            _helper_pull_request_event("synchronize", "benchmark", draft=True, run_drafts="true"),
            True,
            id="draft-labelled-drafts-variable-true",
        ),
        pytest.param(
            _helper_pull_request_event("synchronize", draft=True, run_drafts="true"),
            False,
            id="draft-no-label-drafts-variable-true",
        ),
        pytest.param(
            _helper_pull_request_event(
                "labeled", "benchmark", "bug", added="bug", draft=True, run_drafts="true"
            ),
            False,
            id="draft-labeled-other-label-drafts-variable-true",
        ),
        pytest.param(_helper_other_event("schedule"), True, id="schedule"),
        pytest.param(_helper_other_event("workflow_dispatch"), True, id="workflow-dispatch"),
    ],
)
def test_benchmark_jobs_run_exactly_when_the_run_joins_the_pull_requests_group(
    context: dict[str, Any], runs: bool
) -> None:
    # The two jobs' `if`s and the concurrency group state one condition three times, so an edit to
    # one of them alone breaks a run. A run whose jobs skip but which joins the pull request's group
    # cancels the benchmark running there (adding an unrelated label would), and a run whose jobs
    # run in a group of its own is not cancelled by the push that replaces its head.
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    jobs = {name: _helper_job_runs(workflow["jobs"][name], context) for name in ("unit", "bench")}
    group = _helper_evaluate(workflow["concurrency"]["group"], context)

    joins = context["github"]["event_name"] == "pull_request" and runs
    expected_group = "Benchmark-refs/pull/1/merge" if joins else "Benchmark-999"
    assert (jobs, group) == ({"unit": runs, "bench": runs}, expected_group), (
        "unit and bench must run or skip together, and a run must take the pull request's group "
        + "exactly when it is a pull-request run whose jobs run; any other run takes its run id"
    )
