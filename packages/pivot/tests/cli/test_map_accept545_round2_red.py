"""Red test from round 2 of the acceptance of sjawhar/pivot#545 (BET-473) at caf288134cc2.

It drives the installed ``pivot repro --watch --serve`` in a real git project and calls the agent
server over its Unix socket, as ``docs/architecture/agent-server.md``'s client example does. It
connects through ``pivot.unix_socket``, as the server binds, so a socket path longer than the
kernel's address limit works. It failed at caf288134cc2.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from typing import TYPE_CHECKING

import anyio
import yaml

from pivot import unix_socket
from tests import helpers

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

_PIPELINE = """\
from typing import Annotated

from pivot import Pipeline, directory, loaders, stage


@stage
def plan(
    entries: dict[str, int],
) -> Annotated[dict[str, dict[str, int]], directory, loaders.YAML[dict[str, int]]()]:
    return {f"{name}.yaml": {"value": value} for name, value in entries.items()}


@stage
def cut(entry: dict[str, int]) -> dict[str, int]:
    return {"cut": entry["value"] * 2}


with Pipeline("p") as pipeline:
    cut(pipeline.map(plan(pipeline.input("plan.yaml", t=dict[str, int]))))
"""


def _ok(proc: subprocess.CompletedProcess[str]) -> str:
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    return out


def _project(tmp_path: pathlib.Path, plan: Mapping[str, int]) -> tuple[pathlib.Path, pathlib.Path]:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    for args in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
    ):
        _ok(helpers.run_cli(root, home, *args))
    _ok(helpers.run_pivot(root, home, "init"))
    (root / "pipeline.py").write_text(_PIPELINE)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "plan.yaml").write_text(yaml.safe_dump(dict(plan)))
    return root, home


async def _call(path: pathlib.Path, method: str, params: dict[str, object]) -> dict[str, object]:
    request = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    with anyio.fail_after(30):
        async with await unix_socket.connect_unix(path) as stream:
            await stream.send(json.dumps(request).encode() + b"\n")
            buffer = b""
            while not buffer.endswith(b"\n"):
                try:
                    buffer += await stream.receive(65536)
                except anyio.EndOfStream:
                    break
    return json.loads(buffer)


def _rpc(path: pathlib.Path, method: str, params: dict[str, object]) -> dict[str, object]:
    """One JSON-RPC call to the agent server; the whole reply."""
    return anyio.run(_call, path, method, params)


def _wait_for_expansion(path: pathlib.Path, deadline: float) -> None:
    """Wait until the server's socket exists and its ``stages`` answer holds p/cut's instances.

    The server binds the socket, then listens (``anyio.create_unix_listener``), so a connection
    in between is refused; no other connection error is retried."""
    while not path.is_socket():
        assert time.monotonic() < deadline, "the watch server never created its socket"
        time.sleep(0.2)
    while True:
        try:
            result = _rpc(path, "stages", {}).get("result")
        except ConnectionRefusedError:
            result = None
        if isinstance(result, dict) and result.get("templates", {}).get("p/cut") == [
            "p/cut@a",
            "p/cut@b",
        ]:
            return
        assert time.monotonic() < deadline, "the watch run never expanded p/cut"
        time.sleep(0.5)


def test_rpc_explain_and_stage_info_of_a_key_no_instance_has_say_it_is_unknown(
    tmp_path: pathlib.Path,
) -> None:
    """With p/cut expanded into p/cut@a and p/cut@b (the ``stages`` answer says so), RPC
    ``explain`` of ``p/cut@zzz`` answered ``will_run: true`` with "Not yet expanded: 'p/cut' adds
    its instances when a run finds 'p/plan' current", and ``stage_info`` refused it as "not yet
    expanded", at caf288134cc2. The CLI refuses the same name as "Unknown stage(s): p/cut@zzz", and
    RPC answers a name no stage has with an error."""
    root, home = _project(tmp_path, {"a": 1, "b": 2})
    sock = root / ".pivot" / "agent.sock"
    watch = subprocess.Popen(
        [helpers.PIVOT, "repro", "--watch", "--serve"],
        cwd=root,
        env=helpers.cli_env(home),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_expansion(sock, time.monotonic() + 90)
        no_stage = _rpc(sock, "explain", {"stage": "p/nosuch"})
        explain = _rpc(sock, "explain", {"stage": "p/cut@zzz"})
        info = _rpc(sock, "stage_info", {"stage": "p/cut@zzz"})
    finally:
        watch.send_signal(signal.SIGINT)
        try:
            watch.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watch.kill()
            watch.wait()

    assert "error" in no_stage, f"a name no stage has is an error: {no_stage}"
    assert "error" in explain, f"explain of a key no instance has: {explain}"
    assert "not yet expanded" not in json.dumps(explain).lower(), explain
    assert "error" in info, info
    assert "not yet expanded" not in json.dumps(info).lower(), info
    assert "p/cut@zzz" in json.dumps(info), info
