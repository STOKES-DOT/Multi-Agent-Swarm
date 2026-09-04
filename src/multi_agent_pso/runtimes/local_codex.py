"""Local Codex SDK adapter behind the generic :class:`AgentRuntime` port."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from multi_agent_pso.protocols import StageRequest, StageResponse, ThreadRef, TokenUsage


_JSON_LIMIT = 256 * 1024
_JSON_MAX_DEPTH = 32
_JSON_MAX_NODES = 10_000
_JSON_MAX_COLLECTION_ITEMS = 4_096
_SANDBOXES = {"workspace-write", "read-only"}
LOCAL_CODEX_RUNTIME_VERSION = 1


@dataclass(frozen=True)
class CodexUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class CodexTurnResult:
    final_response: str | None
    usage: CodexUsage | None
    turn_id: str | None
    status: str | None
    duration_ms: int | float | None


class CodexThreadPort(Protocol):
    id: str

    async def run(
        self,
        prompt: str,
        *,
        cwd: Path,
        output_schema: Mapping[str, JsonValue] | None,
        sandbox: str,
    ) -> CodexTurnResult: ...


class CodexClientPort(Protocol):
    async def thread_start(
        self, *, model: str, cwd: Path, sandbox: str
    ) -> CodexThreadPort: ...

    async def thread_resume(
        self, thread_id: str, *, model: str, cwd: Path, sandbox: str
    ) -> CodexThreadPort: ...

    async def close(self) -> None: ...


class _SDKThreadAdapter:
    def __init__(self, thread: object, sdk: object) -> None:
        self._thread = thread
        self._sdk = sdk
        self.id = _require_text(getattr(thread, "id", None), "SDK thread id")

    async def run(
        self,
        prompt: str,
        *,
        cwd: Path,
        output_schema: Mapping[str, JsonValue] | None,
        sandbox: str,
    ) -> CodexTurnResult:
        result = await self._thread.run(
            prompt,
            cwd=str(cwd),
            output_schema=output_schema,
            sandbox=self._sandbox(sandbox),
        )
        usage_value = getattr(result, "usage", None)
        breakdown = None if usage_value is None else getattr(usage_value, "last", None)
        usage = (
            None
            if breakdown is None
            else CodexUsage(
                getattr(breakdown, "input_tokens"),
                getattr(breakdown, "output_tokens"),
                getattr(breakdown, "cached_input_tokens"),
            )
        )
        status_value = getattr(result, "status", None)
        status = (
            None
            if status_value is None
            else str(getattr(status_value, "value", status_value))
        )
        return CodexTurnResult(
            getattr(result, "final_response", None),
            usage,
            getattr(result, "id", None),
            status,
            getattr(result, "duration_ms", None),
        )

    def _sandbox(self, value: str):
        return {
            "workspace-write": self._sdk.Sandbox.workspace_write,
            "read-only": self._sdk.Sandbox.read_only,
        }[value]


class OpenAICodexClientAdapter:
    """Lazy lifecycle wrapper for ``openai-codex`` 0.147+ APIs."""

    def __init__(self) -> None:
        try:
            import openai_codex
        except ImportError as error:
            raise RuntimeError(
                "LocalCodexRuntime requires the 'codex' extra: pip install -e '.[codex]'"
            ) from error
        self._sdk = openai_codex
        self._client = openai_codex.AsyncCodex(config=None)
        self._open = False
        self._closed = False
        self._lock = asyncio.Lock()
        self.sdk_version = importlib.metadata.version("openai-codex")

    async def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Codex client is closed")
        if self._open:
            return
        async with self._lock:
            if not self._open:
                await self._client.__aenter__()
                self._open = True

    async def thread_start(self, *, model: str, cwd: Path, sandbox: str):
        await self._ensure_open()
        thread = await self._client.thread_start(
            model=model,
            cwd=str(cwd),
            sandbox=self._sandbox(sandbox),
            approval_mode=self._sdk.ApprovalMode.deny_all,
        )
        return _SDKThreadAdapter(thread, self._sdk)

    async def thread_resume(
        self, thread_id: str, *, model: str, cwd: Path, sandbox: str
    ):
        await self._ensure_open()
        thread = await self._client.thread_resume(
            thread_id,
            model=model,
            cwd=str(cwd),
            sandbox=self._sandbox(sandbox),
            approval_mode=self._sdk.ApprovalMode.deny_all,
        )
        return _SDKThreadAdapter(thread, self._sdk)

    def _sandbox(self, value: str):
        return {
            "workspace-write": self._sdk.Sandbox.workspace_write,
            "read-only": self._sdk.Sandbox.read_only,
        }[value]

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._open:
                await self._client.__aexit__(None, None, None)
                self._open = False


@dataclass
class _ThreadEntry:
    reference: ThreadRef
    provider_thread: CodexThreadPort
    lock: asyncio.Lock


class LocalCodexRuntime:
    """Isolated, bounded local Codex threads with explicit ownership."""

    def __init__(
        self,
        *,
        model: str,
        sandbox: str = "workspace-write",
        client: CodexClientPort | None = None,
        own_client: bool | None = None,
    ) -> None:
        self._model = _require_text(model, "model")
        if sandbox not in _SANDBOXES:
            raise ValueError("sandbox must be 'workspace-write' or 'read-only'")
        self._sandbox = sandbox
        if client is None:
            client = OpenAICodexClientAdapter()
            default_ownership = True
        else:
            default_ownership = False
        self._client = client
        self._own_client = default_ownership if own_client is None else own_client
        if type(self._own_client) is not bool:
            raise TypeError("own_client must be a boolean")
        self._sdk_version = str(getattr(client, "sdk_version", "injected"))
        self._entries: dict[str, _ThreadEntry] = {}
        self._particle_threads: dict[str, str] = {}
        self._state_lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self) -> "LocalCodexRuntime":
        self._require_open()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        await self.close()
        return False

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        particle = _require_text(particle_id, "particle_id")
        cwd = _workspace(workspace)
        self._require_open()
        async with self._state_lock:
            self._require_open()
            if particle in self._particle_threads:
                raise ValueError("particle already has an active Codex thread")
            provider = await self._client.thread_start(
                model=self._model, cwd=cwd, sandbox=self._sandbox
            )
            reference = self._reference(provider, particle, 0, cwd)
            self._insert(reference, provider)
            return reference

    async def restore_thread(
        self, particle_id: str, workspace: Path, checkpoint: Mapping[str, JsonValue]
    ) -> ThreadRef:
        particle = _require_text(particle_id, "particle_id")
        cwd = _workspace(workspace)
        self._require_open()
        reference = _checkpoint_reference(checkpoint)
        if reference.particle_id != particle or reference.workspace != cwd:
            raise ValueError("checkpoint thread identity does not match restore request")
        if reference.provider_id is None:
            raise ValueError("checkpoint thread has no provider_id")
        async with self._state_lock:
            self._require_open()
            if particle in self._particle_threads:
                raise ValueError("particle already has an active Codex thread")
            provider = await self._client.thread_resume(
                reference.provider_id,
                model=self._model,
                cwd=cwd,
                sandbox=self._sandbox,
            )
            restored = self._reference(
                provider, particle, reference.generation, cwd
            )
            if restored != reference:
                raise ValueError("resumed SDK thread identity does not match checkpoint")
            self._insert(restored, provider)
            return restored

    async def run_stage(
        self, thread: ThreadRef, request: StageRequest
    ) -> StageResponse:
        if not isinstance(request, StageRequest):
            raise TypeError("request must be a StageRequest")
        entry = await self._entry(thread)
        async with entry.lock:
            self._require_open()
            schema = request.to_json()["response_schema"]
            result = await entry.provider_thread.run(
                request.prompt,
                cwd=thread.workspace,
                output_schema=schema,
                sandbox=self._sandbox,
            )
            return self._response(result)

    async def rotate_thread(
        self, thread: ThreadRef, checkpoint: Mapping[str, JsonValue]
    ) -> ThreadRef:
        entry = await self._entry(thread)
        bootstrap = _canonical_checkpoint(checkpoint)
        async with entry.lock:
            self._require_open()
            provider = await self._client.thread_start(
                model=self._model, cwd=thread.workspace, sandbox=self._sandbox
            )
            rotated = self._reference(
                provider, thread.particle_id, thread.generation + 1, thread.workspace
            )
            if rotated.provider_id == thread.provider_id:
                raise ValueError("rotation must create a distinct provider thread")
            result = await provider.run(
                bootstrap,
                cwd=thread.workspace,
                output_schema=None,
                sandbox=self._sandbox,
            )
            self._response(result)
            async with self._state_lock:
                current = self._entries.get(thread.logical_id)
                if current is not entry:
                    raise ValueError("thread mapping changed during rotation")
                self._entries.pop(thread.logical_id)
                self._particle_threads[thread.particle_id] = rotated.logical_id
                self._entries[rotated.logical_id] = _ThreadEntry(
                    rotated, provider, asyncio.Lock()
                )
            return rotated

    async def close_thread(self, thread: ThreadRef) -> None:
        entry = await self._entry(thread)
        async with entry.lock:
            async with self._state_lock:
                if self._entries.get(thread.logical_id) is not entry:
                    raise ValueError("unknown Codex thread")
                self._entries.pop(thread.logical_id)
                self._particle_threads.pop(thread.particle_id, None)

    async def close(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._particle_threads.clear()
        for entry in entries:
            async with entry.lock:
                pass
        if self._own_client:
            await self._client.close()

    async def _entry(self, thread: ThreadRef) -> _ThreadEntry:
        if not isinstance(thread, ThreadRef):
            raise TypeError("thread must be a ThreadRef")
        self._require_open()
        async with self._state_lock:
            entry = self._entries.get(thread.logical_id)
            if entry is None:
                raise ValueError("unknown Codex thread")
            if entry.reference != thread:
                raise ValueError("Codex thread identity mismatch")
            return entry

    def _insert(self, reference: ThreadRef, provider: CodexThreadPort) -> None:
        if reference.logical_id in self._entries:
            raise ValueError("Codex logical thread collision")
        if any(
            entry.reference.provider_id == reference.provider_id
            for entry in self._entries.values()
        ):
            raise ValueError("Codex provider thread collision")
        self._entries[reference.logical_id] = _ThreadEntry(
            reference, provider, asyncio.Lock()
        )
        self._particle_threads[reference.particle_id] = reference.logical_id

    @staticmethod
    def _reference(
        provider: CodexThreadPort,
        particle_id: str,
        generation: int,
        workspace: Path,
    ) -> ThreadRef:
        provider_id = _require_text(getattr(provider, "id", None), "provider thread id")
        payload = json.dumps(
            ["local-codex-thread", provider_id, particle_id, generation, str(workspace)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return ThreadRef(
            hashlib.sha256(payload).hexdigest(),
            particle_id,
            generation,
            workspace,
            provider_id,
        )

    def _response(self, result: CodexTurnResult) -> StageResponse:
        if not isinstance(result, CodexTurnResult):
            raise TypeError("Codex thread returned an invalid turn result")
        raw = result.final_response
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Codex turn has no nonempty final response")
        usage_available = result.usage is not None
        usage = (
            TokenUsage(0, 0, 0)
            if result.usage is None
            else TokenUsage(
                _token(result.usage.input_tokens),
                _token(result.usage.output_tokens),
                _token(result.usage.cached_input_tokens),
            )
        )
        metadata: dict[str, JsonValue] = {
            "sdk_version": self._sdk_version,
            "runtime_version": LOCAL_CODEX_RUNTIME_VERSION,
            "usage_available": usage_available,
        }
        if result.turn_id is not None:
            metadata["turn_id"] = _require_text(result.turn_id, "turn_id")
        if result.status is not None:
            metadata["status"] = _require_text(result.status, "status")
        if result.duration_ms is not None:
            if isinstance(result.duration_ms, bool) or not isinstance(
                result.duration_ms, (int, float)
            ) or not math.isfinite(float(result.duration_ms)) or result.duration_ms < 0:
                raise ValueError("duration_ms must be a finite nonnegative number")
            metadata["duration_ms"] = result.duration_ms
        return StageResponse(raw, usage, metadata)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LocalCodexRuntime is closed")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _workspace(value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("workspace must be an absolute Path")
    if not value.is_dir():
        raise ValueError("workspace must be an existing directory")
    return value


def _token(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("token usage values must be nonnegative integers")
    return value


def _checkpoint_reference(checkpoint: object) -> ThreadRef:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    value = checkpoint.get("thread_json")
    if not isinstance(value, Mapping) or set(value) != {
        "logical_id", "particle_id", "generation", "workspace", "provider_id"
    }:
        raise ValueError("checkpoint thread_json is missing or invalid")
    try:
        return ThreadRef(
            value["logical_id"],
            value["particle_id"],
            value["generation"],
            Path(value["workspace"]),
            value["provider_id"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint thread_json is invalid") from error


def _json_value(
    value: object, *, depth: int = 0, nodes: list[int] | None = None
) -> JsonValue:
    if depth > _JSON_MAX_DEPTH:
        raise ValueError("checkpoint exceeds the v1 JSON depth limit")
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > _JSON_MAX_NODES:
        raise ValueError("checkpoint exceeds the v1 JSON node limit")
    if isinstance(value, Mapping):
        if len(value) > _JSON_MAX_COLLECTION_ITEMS:
            raise ValueError("checkpoint exceeds the v1 JSON collection limit")
        result: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint JSON keys must be strings")
            result[key] = _json_value(nested, depth=depth + 1, nodes=nodes)
        return result
    if type(value) in (list, tuple):
        if len(value) > _JSON_MAX_COLLECTION_ITEMS:
            raise ValueError("checkpoint exceeds the v1 JSON collection limit")
        return [
            _json_value(item, depth=depth + 1, nodes=nodes) for item in value
        ]
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("checkpoint must contain finite JSON values")


def _canonical_checkpoint(value: object) -> str:
    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _JSON_LIMIT:
        raise ValueError("checkpoint exceeds the v1 JSON byte limit")
    return encoded.decode("utf-8")


__all__ = [
    "CodexClientPort",
    "CodexThreadPort",
    "CodexTurnResult",
    "CodexUsage",
    "LocalCodexRuntime",
    "LOCAL_CODEX_RUNTIME_VERSION",
    "OpenAICodexClientAdapter",
]
