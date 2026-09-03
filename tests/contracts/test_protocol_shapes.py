"""Public contract checks for orchestration boundary ports and records."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import AsyncContextManager, ContextManager, get_type_hints

import pytest
from pydantic import JsonValue

from multi_agent_pso.core import (
    AgentEpisode,
    AgentStage,
    ArtifactRef,
    Evaluation,
    EvaluationStatus,
    PersonalBest,
    StageEvent,
)
from multi_agent_pso.protocols import (
    AgentRuntime,
    ArtifactStore,
    CandidateRef,
    EvaluationContext,
    Evaluator,
    IterationTransaction,
    ResourceManager,
    RunStore,
    StageRequest,
    StageResponse,
    TaskAdapter,
    ThreadRef,
    TokenUsage,
    ToolContext,
    ToolProvider,
    ToolRequest,
    ToolResult,
    ToolStatus,
    WikiHit,
    WikiQuery,
    WikiRetriever,
)


HASH = "a" * 64
WORKSPACE = Path("/tmp/multi-agent-pso-contracts")
ARTIFACT = ArtifactRef(
    relative_path="artifact.json",
    sha256=HASH,
    size_bytes=0,
    media_type="application/json",
    committed=True,
)


def _protocol_methods(protocol: type[object]) -> set[str]:
    return {
        name
        for name, value in protocol.__dict__.items()
        if not name.startswith("_") and callable(value)
    }


def _expected_parameters(*names: str) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (name, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty)
        for name in names
    )


def _signature_parameters(callable_: object) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_).parameters.values()
    )


def _assert_signature_and_hints(
    callable_: object,
    parameter_names: tuple[str, ...],
    expected_hints: Mapping[str, object],
) -> None:
    assert _signature_parameters(callable_) == _expected_parameters(*parameter_names)
    assert get_type_hints(callable_) == expected_hints


def _assert_json_value(value: object) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_json_value(item)
        return
    pytest.fail(f"non-JSON-compatible value leaked from boundary record: {value!r}")


def test_public_protocols_are_runtime_checkable() -> None:
    for protocol in (
        AgentRuntime,
        ArtifactStore,
        Evaluator,
        IterationTransaction,
        ResourceManager,
        RunStore,
        TaskAdapter,
        ToolProvider,
        WikiRetriever,
    ):
        assert getattr(protocol, "_is_runtime_protocol", False)


def test_protocol_method_names_and_async_boundaries_are_exact() -> None:
    assert _protocol_methods(AgentRuntime) == {
        "start_thread",
        "run_stage",
        "rotate_thread",
        "close_thread",
    }
    assert _protocol_methods(Evaluator) == {"evaluate"}
    assert _protocol_methods(ToolProvider) == {"execute"}
    assert _protocol_methods(ResourceManager) == {"agent_slot", "evaluation_slot"}
    assert _protocol_methods(IterationTransaction) == {
        "put_particle_json",
        "put_snapshot_json",
        "commit",
        "rollback",
    }
    assert _protocol_methods(RunStore) == {
        "create_run",
        "append_stage_event",
        "get_committed_tool_result",
        "iteration_transaction",
    }
    assert _protocol_methods(ArtifactStore) == {
        "publish_bytes",
        "publish_text",
        "publish_json",
    }
    assert _protocol_methods(TaskAdapter) == {
        "build_stage_request",
        "parse_stage_response",
        "realized_position",
        "evaluated_position",
        "position_adherence",
        "compare",
        "summarize_best",
    }
    assert _protocol_methods(WikiRetriever) == {"search"}

    for protocol, methods in (
        (AgentRuntime, ("start_thread", "run_stage", "rotate_thread", "close_thread")),
        (Evaluator, ("evaluate",)),
        (ToolProvider, ("execute",)),
    ):
        for method in methods:
            assert inspect.iscoroutinefunction(getattr(protocol, method))
    for protocol, methods in (
        (ResourceManager, ("agent_slot", "evaluation_slot")),
        (IterationTransaction, ("put_particle_json", "put_snapshot_json", "commit", "rollback")),
        (RunStore, ("create_run", "append_stage_event", "get_committed_tool_result", "iteration_transaction")),
        (ArtifactStore, ("publish_bytes", "publish_text", "publish_json")),
        (TaskAdapter, tuple(_protocol_methods(TaskAdapter))),
        (WikiRetriever, ("search",)),
    ):
        for method in methods:
            assert not inspect.iscoroutinefunction(getattr(protocol, method))


def test_protocol_and_fake_signatures_and_resolved_hints_match() -> None:
    class RuntimeFake:
        async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
            return ThreadRef("thread", particle_id, 0, workspace)

        async def run_stage(self, thread: ThreadRef, request: StageRequest) -> StageResponse:
            return StageResponse("{}", TokenUsage(0, 0))

        async def rotate_thread(
            self, thread: ThreadRef, checkpoint: Mapping[str, JsonValue]
        ) -> ThreadRef:
            return thread

        async def close_thread(self, thread: ThreadRef) -> None:
            return None

    class EvaluatorFake:
        async def evaluate(
            self, candidate: CandidateRef, context: EvaluationContext
        ) -> Evaluation:
            return Evaluation(
                status=EvaluationStatus.SUCCESS,
                feasible=True,
                fitness=0.0,
            )

    class ToolFake:
        async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
            return ToolResult(ToolStatus.SUCCESS, {})

    class ResourceFake:
        def agent_slot(self) -> AsyncContextManager[None]:
            raise NotImplementedError

        def evaluation_slot(self) -> AsyncContextManager[None]:
            raise NotImplementedError

    class TransactionFake:
        def put_particle_json(self, particle_id: str, payload: Mapping[str, JsonValue]) -> None:
            return None

        def put_snapshot_json(self, payload: Mapping[str, JsonValue]) -> None:
            return None

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    class RunStoreFake:
        def create_run(self, run_id: str, snapshot_hash: str) -> None:
            return None

        def append_stage_event(self, event: StageEvent) -> None:
            return None

        def get_committed_tool_result(self, idempotency_key: str) -> ToolResult | None:
            return None

        def iteration_transaction(
            self, run_id: str, iteration_id: int
        ) -> ContextManager[IterationTransaction]:
            raise NotImplementedError

    class ArtifactStoreFake:
        def publish_bytes(self, relative_path: str, data: bytes, media_type: str) -> ArtifactRef:
            return ARTIFACT

        def publish_text(self, relative_path: str, text: str, media_type: str) -> ArtifactRef:
            return ARTIFACT

        def publish_json(self, relative_path: str, payload: Mapping[str, JsonValue]) -> ArtifactRef:
            return ARTIFACT

    class AdapterFake:
        def build_stage_request(
            self, stage: AgentStage, context: Mapping[str, JsonValue]
        ) -> StageRequest:
            return StageRequest(stage, "prompt")

        def parse_stage_response(
            self, stage: AgentStage, response: StageResponse
        ) -> Mapping[str, JsonValue]:
            return {}

        def realized_position(self, episode: AgentEpisode) -> str | None:
            return None

        def evaluated_position(self, target: str, realized: str | None) -> str:
            return target

        def position_adherence(
            self, target: str, realized: str | None
        ) -> Mapping[str, JsonValue]:
            return {}

        def compare(self, left: Evaluation, right: Evaluation) -> int:
            return 0

        def summarize_best(self, best: PersonalBest | None) -> Mapping[str, JsonValue]:
            return {}

    class WikiFake:
        def search(self, query: WikiQuery) -> tuple[WikiHit, ...]:
            return ()

    # runtime_checkable validates member presence only; these explicit checks
    # protect names, kinds, defaults, and resolved hints for implementers.
    contracts: tuple[
        tuple[type[object], object, Mapping[str, tuple[tuple[str, ...], Mapping[str, object]]]],
        ...,
    ] = (
        (
            AgentRuntime,
            RuntimeFake(),
            {
                "start_thread": (("self", "particle_id", "workspace"), {"particle_id": str, "workspace": Path, "return": ThreadRef}),
                "run_stage": (("self", "thread", "request"), {"thread": ThreadRef, "request": StageRequest, "return": StageResponse}),
                "rotate_thread": (("self", "thread", "checkpoint"), {"thread": ThreadRef, "checkpoint": Mapping[str, JsonValue], "return": ThreadRef}),
                "close_thread": (("self", "thread"), {"thread": ThreadRef, "return": type(None)}),
            },
        ),
        (
            Evaluator,
            EvaluatorFake(),
            {"evaluate": (("self", "candidate", "context"), {"candidate": CandidateRef, "context": EvaluationContext, "return": Evaluation})},
        ),
        (
            ToolProvider,
            ToolFake(),
            {"execute": (("self", "request", "context"), {"request": ToolRequest, "context": ToolContext, "return": ToolResult})},
        ),
        (
            ResourceManager,
            ResourceFake(),
            {
                "agent_slot": (("self",), {"return": AsyncContextManager[None]}),
                "evaluation_slot": (("self",), {"return": AsyncContextManager[None]}),
            },
        ),
        (
            IterationTransaction,
            TransactionFake(),
            {
                "put_particle_json": (("self", "particle_id", "payload"), {"particle_id": str, "payload": Mapping[str, JsonValue], "return": type(None)}),
                "put_snapshot_json": (("self", "payload"), {"payload": Mapping[str, JsonValue], "return": type(None)}),
                "commit": (("self",), {"return": type(None)}),
                "rollback": (("self",), {"return": type(None)}),
            },
        ),
        (
            RunStore,
            RunStoreFake(),
            {
                "create_run": (("self", "run_id", "snapshot_hash"), {"run_id": str, "snapshot_hash": str, "return": type(None)}),
                "append_stage_event": (("self", "event"), {"event": StageEvent, "return": type(None)}),
                "get_committed_tool_result": (("self", "idempotency_key"), {"idempotency_key": str, "return": ToolResult | None}),
                "iteration_transaction": (("self", "run_id", "iteration_id"), {"run_id": str, "iteration_id": int, "return": ContextManager[IterationTransaction]}),
            },
        ),
        (
            ArtifactStore,
            ArtifactStoreFake(),
            {
                "publish_bytes": (("self", "relative_path", "data", "media_type"), {"relative_path": str, "data": bytes, "media_type": str, "return": ArtifactRef}),
                "publish_text": (("self", "relative_path", "text", "media_type"), {"relative_path": str, "text": str, "media_type": str, "return": ArtifactRef}),
                "publish_json": (("self", "relative_path", "payload"), {"relative_path": str, "payload": Mapping[str, JsonValue], "return": ArtifactRef}),
            },
        ),
        (
            WikiRetriever,
            WikiFake(),
            {"search": (("self", "query"), {"query": WikiQuery, "return": tuple[WikiHit, ...]})},
        ),
    )
    for protocol, fake, methods in contracts:
        assert isinstance(fake, protocol)
        for method, (parameter_names, expected_hints) in methods.items():
            _assert_signature_and_hints(getattr(protocol, method), parameter_names, expected_hints)
            _assert_signature_and_hints(
                getattr(type(fake), method), parameter_names, expected_hints
            )

    position_type = TaskAdapter.__parameters__[0]
    _assert_signature_and_hints(
        TaskAdapter.build_stage_request,
        ("self", "stage", "context"),
        {"stage": AgentStage, "context": Mapping[str, JsonValue], "return": StageRequest},
    )
    _assert_signature_and_hints(
        TaskAdapter.parse_stage_response,
        ("self", "stage", "response"),
        {"stage": AgentStage, "response": StageResponse, "return": Mapping[str, JsonValue]},
    )
    _assert_signature_and_hints(
        TaskAdapter.realized_position,
        ("self", "episode"),
        {"episode": AgentEpisode, "return": position_type | None},
    )
    _assert_signature_and_hints(
        TaskAdapter.evaluated_position,
        ("self", "target", "realized"),
        {"target": position_type, "realized": position_type | None, "return": position_type},
    )
    _assert_signature_and_hints(
        TaskAdapter.position_adherence,
        ("self", "target", "realized"),
        {"target": position_type, "realized": position_type | None, "return": Mapping[str, JsonValue]},
    )
    _assert_signature_and_hints(
        TaskAdapter.compare,
        ("self", "left", "right"),
        {"left": Evaluation, "right": Evaluation, "return": int},
    )
    _assert_signature_and_hints(
        TaskAdapter.summarize_best,
        ("self", "best"),
        {"best": PersonalBest | None, "return": Mapping[str, JsonValue]},
    )
    adapter_contracts = {
        "build_stage_request": (("self", "stage", "context"), {"stage": AgentStage, "context": Mapping[str, JsonValue], "return": StageRequest}),
        "parse_stage_response": (("self", "stage", "response"), {"stage": AgentStage, "response": StageResponse, "return": Mapping[str, JsonValue]}),
        "realized_position": (("self", "episode"), {"episode": AgentEpisode, "return": str | None}),
        "evaluated_position": (("self", "target", "realized"), {"target": str, "realized": str | None, "return": str}),
        "position_adherence": (("self", "target", "realized"), {"target": str, "realized": str | None, "return": Mapping[str, JsonValue]}),
        "compare": (("self", "left", "right"), {"left": Evaluation, "right": Evaluation, "return": int}),
        "summarize_best": (("self", "best"), {"best": PersonalBest | None, "return": Mapping[str, JsonValue]}),
    }
    for method, (parameter_names, expected_hints) in adapter_contracts.items():
        _assert_signature_and_hints(
            getattr(AdapterFake, method), parameter_names, expected_hints
        )
    adapter: TaskAdapter[str] = AdapterFake()
    assert isinstance(adapter, TaskAdapter)


def test_records_are_frozen_deeply_immutable_and_json_serializable() -> None:
    schema = {"items": [{"type": "string"}]}
    metadata = {"nested": {"values": [1, 2]}}
    request = StageRequest(AgentStage.HYPOTHESIZING, "propose", schema)
    response = StageResponse("{}", TokenUsage(1, 2, 3), metadata)
    candidate = CandidateRef("candidate", HASH, (ARTIFACT,), metadata)
    tool_result = ToolResult(ToolStatus.SUCCESS, metadata, (ARTIFACT,))

    schema["items"][0]["type"] = "number"
    metadata["nested"]["values"].append(3)

    assert request.to_json()["response_schema"] == {"items": [{"type": "string"}]}
    assert response.to_json()["provider_metadata"] == {"nested": {"values": [1, 2]}}
    assert candidate.to_json()["metadata"] == {"nested": {"values": [1, 2]}}
    assert tool_result.to_json()["payload"] == {"nested": {"values": [1, 2]}}
    with pytest.raises((AttributeError, TypeError)):
        request.response_schema["other"] = True  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        response.provider_metadata["nested"]["values"].append(3)  # type: ignore[index,union-attr]

    for record in (
        TokenUsage(1, 2),
        ThreadRef("thread", "particle", 0, WORKSPACE),
        request,
        response,
        ToolRequest("request", "provider", "operation", {}, "idempotency"),
        ToolContext("run", "particle", 0, AgentStage.EXECUTING, 0, WORKSPACE),
        tool_result,
        candidate,
        EvaluationContext("run", "particle", 0, WORKSPACE, HASH, {}),
        WikiQuery("keywords", 2),
        WikiHit("notes/example.md", 1, 2, "evidence", "content"),
    ):
        assert not hasattr(record, "__dict__")
        serialized = record.to_json()
        assert isinstance(serialized, dict)
        _assert_json_value(serialized)
        json.dumps(serialized, allow_nan=False)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: TokenUsage(-1, 0), "input_tokens"),
        (lambda: TokenUsage(0, -1), "output_tokens"),
        (lambda: TokenUsage(0, 0, -1), "cached_input_tokens"),
        (lambda: ThreadRef("", "particle", 0, WORKSPACE), "logical_id"),
        (lambda: ThreadRef("thread", "", 0, WORKSPACE), "particle_id"),
        (lambda: ThreadRef("thread", "particle", -1, WORKSPACE), "generation"),
        (lambda: ThreadRef("thread", "particle", 0, Path("relative")), "workspace"),
        (lambda: ThreadRef("thread", "particle", 0, WORKSPACE, ""), "provider_id"),
        (lambda: StageRequest(AgentStage.PENDING, ""), "prompt"),
        (lambda: ToolRequest("", "provider", "operation", {}, "key"), "request_id"),
        (lambda: ToolRequest("request", "", "operation", {}, "key"), "provider"),
        (lambda: ToolRequest("request", "provider", "", {}, "key"), "operation"),
        (lambda: ToolRequest("request", "provider", "operation", {}, ""), "idempotency_key"),
        (lambda: ToolContext("run", "particle", -1, AgentStage.EXECUTING, 0, WORKSPACE), "iteration_id"),
        (lambda: ToolContext("run", "particle", 0, AgentStage.EXECUTING, -1, WORKSPACE), "attempt"),
        (lambda: ToolContext("", "particle", 0, AgentStage.EXECUTING, 0, WORKSPACE), "run_id"),
        (lambda: ToolContext("run", "", 0, AgentStage.EXECUTING, 0, WORKSPACE), "particle_id"),
        (lambda: CandidateRef("", HASH), "reference"),
        (lambda: CandidateRef("candidate", "A" * 64), "candidate_hash"),
        (lambda: EvaluationContext("run", "particle", -1, WORKSPACE, HASH), "iteration_id"),
        (lambda: EvaluationContext("run", "particle", 0, WORKSPACE, "g" * 64), "protocol_snapshot_hash"),
        (lambda: EvaluationContext("", "particle", 0, WORKSPACE, HASH), "run_id"),
        (lambda: EvaluationContext("run", "", 0, WORKSPACE, HASH), "particle_id"),
        (lambda: EvaluationContext("run", "particle", 0, Path("relative"), HASH), "workspace"),
        (lambda: WikiQuery("", 1), "text"),
        (lambda: WikiQuery("query", 0), "max_results"),
        (lambda: WikiQuery("query", 101), "max_results"),
        (lambda: WikiHit("", 1, 1, "evidence", "content"), "relative_path"),
        (lambda: WikiHit("/absolute/note.md", 1, 1, "evidence", "content"), "relative_path"),
        (lambda: WikiHit("note.md", 0, 1, "evidence", "content"), "line_start"),
        (lambda: WikiHit("note.md", 2, 1, "evidence", "content"), "line_end"),
        (lambda: WikiHit("note.md", 1, 1, "", "content"), "evidence_layer"),
    ],
)
def test_boundary_records_reject_invalid_values(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()  # type: ignore[operator]


def test_resource_manager_return_annotations_are_async_context_managers() -> None:
    hints = get_type_hints(ResourceManager.agent_slot)
    assert hints["return"] == AsyncContextManager[None]
    hints = get_type_hints(ResourceManager.evaluation_slot)
    assert hints["return"] == AsyncContextManager[None]


def test_tool_status_members_are_stable() -> None:
    assert {member.name: member.value for member in ToolStatus} == {
        "SUCCESS": "SUCCESS",
        "REJECTED": "REJECTED",
        "FAILED": "FAILED",
        "TIMEOUT": "TIMEOUT",
    }


def test_task_adapter_compare_docstring_defines_sign_contract() -> None:
    docstring = TaskAdapter.compare.__doc__ or ""
    assert ">0" in docstring
    assert "left better" in docstring
    assert "<0" in docstring
    assert "right better" in docstring
    assert "0" in docstring
    assert "equivalent" in docstring
