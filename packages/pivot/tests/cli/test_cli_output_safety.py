"""CLI output never leaks a URL credential, a raw control character or a bare ``Infinity``.

Each test drives a real command through ``cli.cli`` with a value that an attacker controls:
a credential in a repository or remote URL, an ESC byte in a ``.pvt`` revision, and a
non-finite ``StageParams`` float.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import pathlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
import yaml

from conftest import init_git_repo, isolated_pivot_dir
from pivot import cli, import_artifact, project
from pivot import status as status_mod
from pivot.engine import agent_rpc
from pivot.remote import config as remote_config
from pivot.remote import sync as transfer
from pivot.storage import track

if TYPE_CHECKING:
    import click.testing
    from pytest_mock import MockerFixture

_SECRET = "SECRETTOKEN123"
_CREDENTIAL_URL = f"https://user:{_SECRET}@host.example/repo.git"
_REDACTED_URL = "https://***@host.example/repo.git"
_ESC_REV = "main\x1b[31m"
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

_PARAMS_PIPELINE = """\
from __future__ import annotations

from pivot.compose import Pipeline, stage
from pivot.stage_def import StageParams


class TrainParams(StageParams):
    threshold: float = THRESHOLD


@stage
def train(input_file: str, params: TrainParams) -> str:
    return f"{input_file.strip()}:{params.threshold}"


with Pipeline("demo") as pipeline:
    train(pipeline.input("input.txt"), params=TrainParams())
"""


def _refuse(name: str) -> object:
    raise ValueError(f"non-standard JSON constant {name}")


def _strict_loads(text: str) -> object:
    """Parse as an RFC 8259 parser does: a bare Infinity, -Infinity or NaN is refused."""
    return json.loads(text, parse_constant=_refuse)


def _write_import_pvt(root: pathlib.Path, repo: str, rev: str) -> None:
    """Write ``total.txt`` and the ``.pvt`` of an import of it from ``repo`` at ``rev``."""
    data_path = root / "total.txt"
    data_path.write_text("v1\n")
    track.write_pvt_file(
        track.get_pvt_path(data_path),
        track.PvtData(
            path="total.txt",
            hash="0123456789abcdef",
            size=3,
            source=track.ImportSource(
                repo=repo,
                rev=rev,
                rev_lock="abc123def4567890abc123def4567890abc123de",
                stage="smoke/total",
                path="smoke/total",
                remote="s3://bucket/cache",
            ),
        ),
    )


def _fail_import_check(mocker: MockerFixture, repo: str, rev: str) -> None:
    """Make the import update check fail the way an unreachable repository does."""
    mocker.patch.object(
        import_artifact,
        "check_for_update",
        new=AsyncMock(
            side_effect=Exception(f"Cannot resolve ref '{rev}' from {repo}"),
        ),
    )


def _write_remote_config(root: pathlib.Path, url: str) -> None:
    (root / ".pivot").mkdir(exist_ok=True)
    (root / ".pivot" / "config.yaml").write_text(
        yaml.safe_dump({"remotes": {"origin": url}, "default_remote": "origin"})
    )


# =============================================================================
# URL credentials
# =============================================================================


@pytest.mark.parametrize("output_json", [False, True], ids=["text", "json"])
def test_status_check_imports_hides_the_credential_of_an_import_url(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
    output_json: bool,
) -> None:
    """The credential of a `.pvt` repository URL reaches neither stream, in either format."""
    root = project.get_project_root()
    _write_import_pvt(root, _CREDENTIAL_URL, "main")
    _fail_import_check(mocker, _CREDENTIAL_URL, "main")

    argv = ["status", "--check-imports", *(["--json"] if output_json else [])]
    result = runner.invoke(cli.cli, argv)

    assert result.exit_code == 0, result.output
    assert _SECRET not in result.stdout, result.stdout
    assert _SECRET not in result.stderr, result.stderr
    assert _REDACTED_URL in result.stdout + result.stderr, result.stdout + result.stderr


def test_status_check_imports_warning_hides_the_credential(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """The warning on stderr names the import and its redacted URL."""
    root = project.get_project_root()
    _write_import_pvt(root, _CREDENTIAL_URL, "main")
    _fail_import_check(mocker, _CREDENTIAL_URL, "main")

    result = runner.invoke(cli.cli, ["status", "--check-imports", "--json"])

    assert result.exit_code == 0, result.output
    assert "Failed to check import total.txt" in result.stderr, result.stderr
    assert _REDACTED_URL in result.stderr, result.stderr


def test_remote_list_hides_the_credential_of_a_remote_url(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot remote list` shows the host of a credentialed remote, never its secret."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_remote_config(cwd, f"s3://key:{_SECRET}@bucket/prefix")

        result = runner.invoke(cli.cli, ["remote", "list"])

    assert result.exit_code == 0, result.output
    assert _SECRET not in result.output, result.output
    assert "s3://***@bucket/prefix" in result.stdout, result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["config", "list"], id="list-text"),
        pytest.param(["config", "list", "--json"], id="list-json"),
        pytest.param(["config", "get", "remotes.origin"], id="get-text"),
        pytest.param(["config", "get", "remotes.origin", "--json"], id="get-json"),
    ],
)
def test_config_commands_hide_the_credential_of_a_remote_url(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, argv: list[str]
) -> None:
    """Every `pivot config` view of a remote URL hides its credential."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_remote_config(cwd, f"s3://key:{_SECRET}@bucket/prefix")

        result = runner.invoke(cli.cli, argv)

    assert result.exit_code == 0, result.output
    assert _SECRET not in result.output, result.output
    assert "s3://***@bucket/prefix" in result.stdout, result.stdout


def test_config_set_hides_the_credential_it_echoes(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot config set` stores the URL as typed and echoes it redacted."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        url = f"s3://key:{_SECRET}@bucket/prefix"

        result = runner.invoke(cli.cli, ["config", "set", "remotes.origin", url])
        stored = yaml.safe_load((cwd / ".pivot" / "config.yaml").read_text())

    assert result.exit_code == 0, result.output
    assert _SECRET not in result.output, result.output
    assert "s3://***@bucket/prefix" in result.stdout, result.stdout
    assert stored["remotes"]["origin"] == url, "the configured URL must keep its credential"


def test_config_set_refuses_an_invalid_url_without_echoing_its_credential(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The error of a refused remote URL hides the credential it carries."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(
            cli.cli, ["config", "set", "remotes.origin", f"ftp://key:{_SECRET}@bucket/prefix"]
        )

    assert result.exit_code != 0
    assert _SECRET not in result.output, result.output


def test_get_remote_status_hides_the_credential_of_the_remote_url(
    set_project_root: pathlib.Path, mocker: MockerFixture
) -> None:
    """The remote URL that status reports is redacted at its source."""
    url = f"s3://key:{_SECRET}@bucket/prefix"
    remote = mocker.MagicMock()
    remote.url = url
    remote.list_hashes = mocker.AsyncMock(return_value=set[str]())
    mocker.patch.object(remote_config, "list_remotes", return_value={"origin": url})
    mocker.patch.object(transfer, "create_remote_from_name", return_value=(remote, "origin"))
    mocker.patch.object(remote_config, "get_remote_url", return_value=url)

    result = status_mod.get_remote_status(None, set_project_root / ".pivot" / "cache")

    assert result["url"] == "s3://***@bucket/prefix"


def test_doctor_hides_the_credential_of_a_remote_url(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """`pivot doctor --remote` shows a redacted URL in text and in its JSONL stream."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_remote_config(cwd, f"s3://key:{_SECRET}@bucket/prefix")

        text = runner.invoke(cli.cli, ["doctor", "--remote"])
        jsonl = runner.invoke(cli.cli, ["doctor", "--remote", "--json"])

    assert _SECRET not in text.output, text.output
    assert _SECRET not in jsonl.output, jsonl.output
    values = [
        json.loads(line)["value"]
        for line in jsonl.stdout.splitlines()
        if json.loads(line).get("name") == "remote:origin"
    ]
    assert values == ["s3://***@bucket/prefix"], jsonl.stdout


# =============================================================================
# Control characters
# =============================================================================


def test_status_check_imports_escapes_control_characters_in_text_and_the_warning(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """An ESC byte in a `.pvt` revision reaches the terminal as visible text, not as an escape."""
    root = project.get_project_root()
    _write_import_pvt(root, "https://host.example/repo.git", _ESC_REV)
    _fail_import_check(mocker, "https://host.example/repo.git", _ESC_REV)

    result = runner.invoke(cli.cli, ["status", "--check-imports"])

    assert result.exit_code == 0, result.output
    assert _CONTROL.search(result.stdout) is None, repr(result.stdout)
    assert _CONTROL.search(result.stderr) is None, repr(result.stderr)
    assert "main\\x1b[31m" in result.stdout, result.stdout
    assert "main\\x1b[31m" in result.stderr, result.stderr


def test_status_check_imports_keeps_one_import_on_one_line(
    runner: click.testing.CliRunner,
    mock_discovery: object,
    mocker: MockerFixture,
) -> None:
    """A newline inside a `.pvt` value cannot forge a second line of the Imports section."""
    root = project.get_project_root()
    _write_import_pvt(root, "https://host.example/repo.git", "main\n  \u2713 fake.txt: up to date")
    _fail_import_check(
        mocker, "https://host.example/repo.git", "main\n  \u2713 fake.txt: up to date"
    )

    result = runner.invoke(cli.cli, ["status", "--check-imports"])

    assert result.exit_code == 0, result.output
    forged = [line for line in result.stdout.splitlines() if line.startswith("  \u2713 fake.txt")]
    assert forged == [], result.stdout
    assert "\\x0a" in result.stdout, result.stdout


def test_status_text_escapes_a_control_character_in_a_tracked_file_path(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A tracked path read from a `.pvt` file is escaped in the Tracked Files section."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        data_path = cwd / "data.txt"
        data_path.write_text("x\n")
        track.write_pvt_file(
            track.get_pvt_path(data_path),
            track.PvtData(path="da\x1bta.txt", hash="0123456789abcdef", size=2),
        )

        result = runner.invoke(cli.cli, ["status", "--tracked-only", "--verbose"])

    assert result.exit_code in (0, 1), result.output
    assert _CONTROL.search(result.stdout) is None, repr(result.stdout)
    assert "da\\x1bta.txt" in result.stdout, result.stdout


def test_remote_list_escapes_a_control_character_in_a_remote_name(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A remote name read from `.pivot/config.yaml` is escaped before it is printed."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / ".pivot" / "config.yaml").write_text(
            yaml.safe_dump({"remotes": {"ori\x1b[31mgin": "s3://bucket/prefix"}})
        )

        result = runner.invoke(cli.cli, ["remote", "list"])

    assert result.exit_code == 0, result.output
    assert _CONTROL.search(result.stdout) is None, repr(result.stdout)
    assert "ori\\x1b[31mgin" in result.stdout, result.stdout


def test_config_list_escapes_a_control_character_in_a_remote_url(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A config value is escaped before `pivot config list` prints it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _write_remote_config(cwd, "s3://bucket/pre\x1b[31mfix")

        result = runner.invoke(cli.cli, ["config", "list"])

    assert result.exit_code == 0, result.output
    assert _CONTROL.search(result.stdout) is None, repr(result.stdout)
    assert "pre\\x1b[31mfix" in result.stdout, result.stdout


def test_a_log_line_cannot_carry_a_control_character_to_the_terminal(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """The CLI's log handler escapes every record, whichever module logged it."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "pipeline.py").write_text("raise SystemExit\n")
        _write_remote_config(cwd, "s3://bucket/pre\x1b[31mfix")

        result = runner.invoke(cli.cli, ["--verbose", "config", "list"])

    assert _CONTROL.search(result.stderr) is None, repr(result.stderr)


def test_an_error_message_cannot_carry_a_control_character_to_the_terminal(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A refused command's message is escaped, wherever the value in it came from."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["config", "get", "re\x1b[31mmotes.origin"])

    assert result.exit_code != 0
    assert _CONTROL.search(result.output) is None, repr(result.output)
    assert "re\\x1b[31mmotes.origin" in result.output, result.output


# =============================================================================
# Non-finite floats in --json
# =============================================================================


def _prepare_non_finite_params(
    runner: click.testing.CliRunner, cwd: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record a run whose float parameter is ``inf``, commit it, then set finite values.

    The commit gives ``status --explain``, ``params diff`` and ``metrics diff`` a non-finite
    old value to report.
    """
    init_git_repo(cwd, monkeypatch)
    (cwd / "data" / "raw").mkdir(parents=True)
    (cwd / "data" / "raw" / "input.txt").write_text("hello\n")
    (cwd / "pipeline.py").write_text(_PARAMS_PIPELINE.replace("THRESHOLD", 'float("inf")'))
    (cwd / "metrics.json").write_text('{"loss": Infinity}')

    repro = runner.invoke(cli.cli, ["repro"])
    assert repro.exit_code == 0, repro.output

    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "run"], cwd=cwd, check=True, capture_output=True)

    (cwd / "pipeline.py").write_text(_PARAMS_PIPELINE.replace("THRESHOLD", "0.5"))
    (cwd / "metrics.json").write_text('{"loss": 1.0}')


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        pytest.param(["status", "--json"], 0, id="status"),
        pytest.param(["status", "--explain", "--json"], 0, id="status-explain"),
        pytest.param(["params", "show", "--json"], 0, id="params-show"),
        pytest.param(["params", "diff", "--json"], 0, id="params-diff"),
        pytest.param(["metrics", "show", "metrics.json", "--json"], 0, id="metrics-show"),
        pytest.param(["metrics", "diff", "metrics.json", "--json"], 0, id="metrics-diff"),
        # `plots diff` is left out: it writes only paths, hashes and change types, so no float
        # reaches it, and its encoder is the shared `show/common.format_json`.
        pytest.param(["repro", "--dry-run", "--json"], 0, id="repro-dry-run"),
        # The stage's code changed after the recorded run, so verify reports it and exits 1.
        pytest.param(["verify", "--json"], 1, id="verify"),
        pytest.param(["list", "--json"], 0, id="list"),
        pytest.param(["history", "--json"], 0, id="history"),
        pytest.param(["show", "--json"], 0, id="history-show"),
        pytest.param(["config", "list", "--json"], 0, id="config-list"),
        pytest.param(["config", "get", "core.max_workers", "--json"], 0, id="config-get"),
        # Nothing matches, so check-ignore exits 1; its document is still one strict JSON text.
        pytest.param(["check-ignore", ".pivot/cache", "--json"], 1, id="check-ignore"),
    ],
)
def test_json_output_with_a_non_finite_param_parses_strictly(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    exit_code: int,
) -> None:
    """Every `--json` command writes one document a strict RFC 8259 parser accepts."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _prepare_non_finite_params(runner, cwd, monkeypatch)

        result = runner.invoke(cli.cli, argv)

    assert result.exit_code == exit_code, result.output
    assert _strict_loads(result.stdout) is not None, result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["status", "--explain", "--json"], id="status-explain"),
        pytest.param(["params", "diff", "--json"], id="params-diff"),
        pytest.param(["metrics", "diff", "metrics.json", "--json"], id="metrics-diff"),
    ],
)
def test_json_output_names_the_non_finite_value_as_a_string(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """The recorded ``inf`` reads as the quoted name Python used to write bare."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _prepare_non_finite_params(runner, cwd, monkeypatch)

        result = runner.invoke(cli.cli, argv)

    assert result.exit_code == 0, result.output
    assert '"Infinity"' in result.stdout, result.stdout


def test_status_explain_json_names_the_non_finite_old_value(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`status --explain --json` reports the recorded ``inf`` as the string ``"Infinity"``."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _prepare_non_finite_params(runner, cwd, monkeypatch)

        result = runner.invoke(cli.cli, ["status", "--explain", "--json"])

    assert result.exit_code == 0, result.output
    document = _strict_loads(result.stdout)
    assert isinstance(document, dict)
    changes = [change for stage in document["stages"] for change in stage["param_changes"]]
    assert [(c["key"], c["old_value"], c["new_value"]) for c in changes] == [
        ("threshold", "Infinity", 0.5)
    ], result.stdout


def test_params_show_json_writes_a_non_finite_default_as_a_string(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A stage whose parameter is ``-inf`` in the workspace reads back as ``"-Infinity"``."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "input.txt").write_text("hello\n")
        (cwd / "pipeline.py").write_text(_PARAMS_PIPELINE.replace("THRESHOLD", 'float("-inf")'))

        result = runner.invoke(cli.cli, ["params", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert _strict_loads(result.stdout) == {"demo/train": {"threshold": "-Infinity"}}


def test_metrics_json_writes_a_non_finite_metric_as_a_string(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A metric file holding ``Infinity`` is reported in a form a strict parser accepts."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        (cwd / "data" / "raw").mkdir(parents=True)
        (cwd / "data" / "raw" / "input.txt").write_text("hello\n")
        (cwd / "pipeline.py").write_text(_PARAMS_PIPELINE.replace("THRESHOLD", "0.5"))
        (cwd / "metrics.json").write_text('{"loss": Infinity}')

        result = runner.invoke(cli.cli, ["metrics", "show", "metrics.json", "--json"])

    assert result.exit_code == 0, result.output
    assert _strict_loads(result.stdout) == {"metrics.json": {"loss": "Infinity"}}


def test_doctor_jsonl_parses_strictly(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Every line of the JSONL stream is one strict JSON document."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["doctor", "--json"])

    assert [_strict_loads(line) for line in result.stdout.splitlines()] != []


# =============================================================================
# One encoder behind every JSON writer
# =============================================================================


_OUTPUT_MODULES = ("cli", "show", "engine/agent_rpc.py")


def _json_dumps_calls(path: pathlib.Path) -> list[str]:
    """Every ``json.dumps``/``json.dump`` call in ``path``, as ``file:line``."""
    tree = ast.parse(path.read_text())
    found = list[str]()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"dumps", "dump"}
            and isinstance(func.value, ast.Name)
            and func.value.id == "json"
        ):
            found.append(f"{path}:{node.lineno}")
    return found


def test_no_output_module_calls_json_dumps_directly() -> None:
    """Every writer goes through ``display.to_json``, which has ``allow_nan=False`` behind it.

    A direct ``json.dumps`` would write a bare ``Infinity`` again, which a strict RFC 8259
    parser refuses.
    """
    source_root = pathlib.Path(__file__).resolve().parents[2] / "src" / "pivot"
    offenders = list[str]()
    for name in _OUTPUT_MODULES:
        target = source_root / name
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in files:
            offenders.extend(_json_dumps_calls(path))

    assert offenders == [], f"use display.to_json instead: {offenders}"


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["repro", "--jsonl"], id="repro"),
        pytest.param(["run", "demo/train", "--jsonl"], id="run"),
    ],
)
def test_jsonl_stream_parses_strictly(
    runner: click.testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """Every line of a JSONL stream is one document a strict parser accepts."""
    with isolated_pivot_dir(runner, tmp_path) as cwd:
        _prepare_non_finite_params(runner, cwd, monkeypatch)

        result = runner.invoke(cli.cli, argv)

    assert result.exit_code == 0, result.output
    lines = [_strict_loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert lines != [], result.stdout


def test_agent_rpc_encodes_a_non_finite_param_as_a_string() -> None:
    """The RPC the TUI reads carries a non-finite parameter as a string, not bare."""
    message = {"jsonrpc": "2.0", "id": 1, "result": {"param_changes": [{"old_value": math.inf}]}}

    line = agent_rpc._encode_wire(message)

    assert _strict_loads(line.decode()) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"param_changes": [{"old_value": "Infinity"}]},
    }


# =============================================================================
# The boundaries change nothing else
# =============================================================================


def test_a_multi_line_pivot_error_keeps_its_line_breaks(
    runner: click.testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """A pivot error whose own layout is several lines prints exactly those lines."""
    with isolated_pivot_dir(runner, tmp_path):
        result = runner.invoke(cli.cli, ["list"])

    assert result.exit_code != 0
    assert "Error: No pipeline definition found.\n" in result.output, repr(result.output)
    assert "\nThis command requires a pipeline to be defined in one of:\n" in result.output, repr(
        result.output
    )
    assert "\n  - pipeline.py" in result.output, repr(result.output)


def test_the_log_formatter_keeps_a_record_byte_for_byte(caplog: pytest.LogCaptureFixture) -> None:
    """A record with no control character and no credential formats as the plain formatter does.

    The traceback of an ``exc_info`` record keeps its line breaks too.
    """
    try:
        raise ValueError("a plain failure")
    except ValueError:
        record = logging.LogRecord(
            "pivot.test",
            logging.ERROR,
            __file__,
            1,
            "something failed:\n  on two lines",
            None,
            sys.exc_info(),
        )

    safe = cli._SafeFormatter("%(message)s")
    plain = logging.Formatter("%(message)s")

    assert safe.format(record) == plain.format(record)
    assert "Traceback (most recent call last):" in safe.format(record)
    assert "ValueError: a plain failure" in safe.format(record)


def test_the_log_formatter_hides_a_credential_and_escapes_a_control_character() -> None:
    """The same formatter still rewrites the two things it exists for."""
    record = logging.LogRecord(
        "pivot.test",
        logging.WARNING,
        __file__,
        1,
        "cloning %s at %s",
        (_CREDENTIAL_URL, _ESC_REV),
        None,
    )

    formatted = cli._SafeFormatter("%(message)s").format(record)

    assert formatted == f"cloning {_REDACTED_URL} at main\\x1b[31m"
