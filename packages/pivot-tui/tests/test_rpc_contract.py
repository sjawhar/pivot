from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportAny=false
# pyright: reportUnusedCallResult=false
# pyright: reportUnusedVariable=false
# pyright: reportUntypedFunctionDecorator=false
import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import anyio
import pytest

from pivot.engine.agent_rpc import AgentRpcHandler, AgentRpcSource, EventBuffer
from pivot.engine.engine import Engine
from pivot.engine.types import InputEvent, StageStarted
from pivot.loaders import PathOnly
from pivot.outputs import Dep, Out
from pivot_tui.testing.fake_server import FakeRpcServer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pivot.pipeline import pipeline as pipeline_mod


def _helper_contract_stage(
    data: Annotated[Path, Dep("input.txt", PathOnly())],
) -> Annotated[Path, Out("output.txt", PathOnly())]:
    return data


def _helper_build_request(
    method: str,
    params: dict[str, object] | None = None,
    request_id: int = 1,
) -> dict[str, object]:
    request: dict[str, object] = {"jsonrpc": "2.0", "method": method, "id": request_id}
    if params is not None:
        request["params"] = params
    return request


@contextlib.asynccontextmanager
async def _helper_fake_server(socket_path: Path) -> AsyncGenerator[FakeRpcServer]:
    server = FakeRpcServer()
    await server.start(socket_path)
    try:
        yield server
    finally:
        await server.stop()


async def _helper_send_fake_request(
    socket_path: Path, request: dict[str, object]
) -> dict[str, object]:
    async with await anyio.connect_unix(str(socket_path)) as conn:
        await conn.send(json.dumps(request).encode() + b"\n")
        response_line = await conn.receive(4096)
    return json.loads(response_line.decode())


async def _helper_real_query_response(
    handler: AgentRpcHandler,
    method: str,
    params: dict[str, object] | None,
    request_id: int,
) -> dict[str, object]:
    try:
        result = await handler.handle_query(method, params)
    except ValueError as exc:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32602, "message": f"Invalid params: {exc}"},
            "id": request_id,
        }
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


async def _helper_real_command_response(
    handler: AgentRpcHandler,
    method: str,
    params: dict[str, object] | None,
    request_id: int,
) -> dict[str, object]:
    source = AgentRpcSource(socket_path=Path("/tmp/unused.sock"), handler=handler)
    send, _recv = anyio.create_memory_object_stream[InputEvent](10)
    request = _helper_build_request(method, params, request_id)
    try:
        response = await source._handle_request(request, send)
    finally:
        await send.aclose()
    assert response is not None, "Expected response from AgentRpcSource"
    return dict(response)


def _helper_assert_value_shape_matches(
    fake_value: object, real_value: object, context: str
) -> None:
    assert type(fake_value) is type(real_value), (
        f"Type mismatch for {context}: fake={type(fake_value)}, real={type(real_value)}"
    )
    if isinstance(fake_value, dict) and isinstance(real_value, dict):
        assert set(fake_value.keys()) == set(real_value.keys()), (
            f"Key mismatch for {context}: fake={set(fake_value.keys())}, real={set(real_value.keys())}"
        )
        for key in fake_value:
            _helper_assert_value_shape_matches(
                fake_value[key],
                real_value[key],
                f"{context}.{key}",
            )
    if isinstance(fake_value, list) and isinstance(real_value, list) and fake_value and real_value:
        _helper_assert_value_shape_matches(fake_value[0], real_value[0], f"{context}[0]")


def _helper_assert_response_shape_matches(
    fake_response: dict[str, object],
    real_response: dict[str, object],
    method: str,
) -> None:
    assert set(fake_response.keys()) == set(real_response.keys()), (
        f"Response key mismatch for {method}: fake={set(fake_response.keys())}, "
        f"real={set(real_response.keys())}"
    )
    assert ("result" in fake_response) == ("result" in real_response), (
        f"Shape mismatch for {method}: fake={'result' if 'result' in fake_response else 'error'}, "
        f"real={'result' if 'result' in real_response else 'error'}"
    )
    _helper_assert_value_shape_matches(fake_response, real_response, f"{method}.response")
    if "result" in fake_response and "result" in real_response:
        _helper_assert_value_shape_matches(
            fake_response["result"],
            real_response["result"],
            f"{method}.result",
        )
    if "error" in fake_response and "error" in real_response:
        _helper_assert_value_shape_matches(
            fake_response["error"],
            real_response["error"],
            f"{method}.error",
        )


@pytest.mark.anyio
async def test_rpc_contract_status_and_stages(
    tmp_path: Path,
    test_pipeline: pipeline_mod.Pipeline,
    set_project_root: Path,
) -> None:
    test_pipeline.register(_helper_contract_stage, name="contract_stage")
    (set_project_root / "input.txt").write_text("data")

    socket_path = tmp_path / "agent.sock"
    async with _helper_fake_server(socket_path) as server:
        server.set_status("idle", running=[], pending=[])
        server.set_stages(["contract_stage"])

        async with Engine(pipeline=test_pipeline) as engine:
            handler = AgentRpcHandler(engine=engine, event_buffer=EventBuffer())

            request = _helper_build_request("status")
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(handler, "status", None, 1)
            _helper_assert_response_shape_matches(fake_response, real_response, "status")

            request = _helper_build_request("stages")
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(handler, "stages", None, 1)
            _helper_assert_response_shape_matches(fake_response, real_response, "stages")


@pytest.mark.anyio
async def test_rpc_contract_stage_info_and_explain_success(
    tmp_path: Path,
    test_pipeline: pipeline_mod.Pipeline,
    set_project_root: Path,
) -> None:
    stage_name = "contract_stage"
    test_pipeline.register(_helper_contract_stage, name=stage_name)
    (set_project_root / "input.txt").write_text("data")

    socket_path = tmp_path / "agent.sock"
    async with _helper_fake_server(socket_path) as server:
        server.set_stage_info(stage_name, deps=["input.txt"], outs=["output.txt"])
        server.set_explanation(
            stage_name,
            {
                "stage_name": stage_name,
                "will_run": True,
                "is_forced": False,
                "reason": "No previous run",
                "code_changes": [],
                "param_changes": [],
                "dep_changes": [],
                "upstream_stale": [],
            },
        )

        async with Engine(pipeline=test_pipeline) as engine:
            handler = AgentRpcHandler(engine=engine, event_buffer=EventBuffer())

            request = _helper_build_request("stage_info", {"stage": stage_name})
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(
                handler, "stage_info", {"stage": stage_name}, 1
            )
            _helper_assert_response_shape_matches(fake_response, real_response, "stage_info")

            request = _helper_build_request("explain", {"stage": stage_name})
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(
                handler, "explain", {"stage": stage_name}, 1
            )
            _helper_assert_response_shape_matches(fake_response, real_response, "explain")


@pytest.mark.anyio
async def test_rpc_contract_stage_info_and_explain_unknown(
    tmp_path: Path,
    test_pipeline: pipeline_mod.Pipeline,
    set_project_root: Path,
) -> None:
    test_pipeline.register(_helper_contract_stage, name="contract_stage")
    (set_project_root / "input.txt").write_text("data")

    socket_path = tmp_path / "agent.sock"
    async with _helper_fake_server(socket_path), Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine, event_buffer=EventBuffer())

        request = _helper_build_request("stage_info", {"stage": "unknown"})
        fake_response = await _helper_send_fake_request(socket_path, request)
        real_response = await _helper_real_query_response(
            handler, "stage_info", {"stage": "unknown"}, 1
        )
        _helper_assert_response_shape_matches(fake_response, real_response, "stage_info")

        request = _helper_build_request("explain", {"stage": "unknown"})
        fake_response = await _helper_send_fake_request(socket_path, request)
        real_response = await _helper_real_query_response(
            handler, "explain", {"stage": "unknown"}, 1
        )
        _helper_assert_response_shape_matches(fake_response, real_response, "explain")


@pytest.mark.anyio
async def test_rpc_contract_events_since_valid_and_invalid(
    tmp_path: Path,
    test_pipeline: pipeline_mod.Pipeline,
    set_project_root: Path,
) -> None:
    test_pipeline.register(_helper_contract_stage, name="contract_stage")
    (set_project_root / "input.txt").write_text("data")

    socket_path = tmp_path / "agent.sock"
    async with _helper_fake_server(socket_path) as server:
        server.inject_event(
            dict(
                StageStarted(
                    type="stage_started",
                    stage="contract_stage",
                    index=1,
                    total=1,
                )
            )
        )

        event_buffer = EventBuffer(max_events=10)
        event_buffer.handle_sync(
            StageStarted(type="stage_started", stage="contract_stage", index=1, total=1)
        )

        async with Engine(pipeline=test_pipeline) as engine:
            handler = AgentRpcHandler(engine=engine, event_buffer=event_buffer)

            request = _helper_build_request("events_since", {"version": 0})
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(
                handler, "events_since", {"version": 0}, 1
            )
            _helper_assert_response_shape_matches(fake_response, real_response, "events_since")

            request = _helper_build_request("events_since", {"version": "invalid"})
            fake_response = await _helper_send_fake_request(socket_path, request)
            real_response = await _helper_real_query_response(
                handler, "events_since", {"version": "invalid"}, 1
            )
            _helper_assert_response_shape_matches(
                fake_response, real_response, "events_since_invalid"
            )


@pytest.mark.anyio
async def test_rpc_contract_run_and_cancel(
    tmp_path: Path,
    test_pipeline: pipeline_mod.Pipeline,
    set_project_root: Path,
) -> None:
    test_pipeline.register(_helper_contract_stage, name="contract_stage")
    (set_project_root / "input.txt").write_text("data")

    socket_path = tmp_path / "agent.sock"
    async with _helper_fake_server(socket_path), Engine(pipeline=test_pipeline) as engine:
        handler = AgentRpcHandler(engine=engine, event_buffer=EventBuffer())

        request = _helper_build_request("run")
        fake_response = await _helper_send_fake_request(socket_path, request)
        real_response = await _helper_real_command_response(handler, "run", None, 1)
        _helper_assert_response_shape_matches(fake_response, real_response, "run")

        request = _helper_build_request("cancel")
        fake_response = await _helper_send_fake_request(socket_path, request)
        real_response = await _helper_real_command_response(handler, "cancel", None, 1)
        _helper_assert_response_shape_matches(fake_response, real_response, "cancel")
