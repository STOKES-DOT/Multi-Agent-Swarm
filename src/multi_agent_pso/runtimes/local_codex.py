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


class ProviderThreadNotFoundError(RuntimeError):
    """The provider no longer has a thread referenced by a checkpoint."""


class CodexTransportInterruptedError(asyncio.CancelledError):
    """The SDK transport stopped without caller-initiated task cancellation."""


def _raise_if_transport_interrupted(error: BaseException, sdk: object) -> None:
    if isinstance(error, asyncio.CancelledError):
        task = asyncio.current_task()
        if task is None or task.cancelling() == 0:
            raise CodexTransportInterruptedError(
                "Codex SDK operation was interrupted without caller cancellation"
            ) from error
        return
    transport_closed = getattr(sdk, "TransportClosedError", ())
    if isinstance(transport_closed, type) and isinstance(error, transport_closed):
        raise CodexTransportInterruptedError(
            "Codex app-server transport closed"
        ) from error


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
        try:
            result = await self._thread.run(
                prompt,
                cwd=str(cwd),
                output_schema=output_schema,
                sandbox=self._sandbox(sandbox),
            )
        except BaseException as error:
            _raise_if_transport_interrupted(error, self._sdk)
            raise
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
        try:
            thread = await self._client.thread_start(
                model=model,
                cwd=str(cwd),
                sandbox=self._sandbox(sandbox),
                approval_mode=self._sdk.ApprovalMode.deny_all,
            )
        except BaseException as error:
            _raise_if_transport_interrupted(error, self._sdk)
            raise
        return _SDKThreadAdapter(thread, self._sdk)

    async def thread_resume(
        self, thread_id: str, *, model: str, cwd: Path, sandbox: str
    ):
        await self._ensure_open()
        try:
            thread = await self._client.thread_resume(
                thread_id,
                model=model,
                cwd=str(cwd),
                sandbox=self._sandbox(sandbox),
                approval_mode=self._sdk.ApprovalMode.deny_all,
            )
        except BaseException as error:
            _raise_if_transport_interrupted(error, self._sdk)
            if not isinstance(error, Exception):
                raise
            message = str(error).lower()
            not_found = "thread" in message and any(
                phrase in message
                for phrase in ("not found", "unknown", "does not exist")
            )
            expected_errors = tuple(
                candidate
                for candidate in (
                    getattr(self._sdk, "CodexRpcError", None),
                    getattr(self._sdk, "InvalidParamsError", None),
                )
                if isinstance(candidate, type)
            )
            if expected_errors and isinstance(error, expected_errors) and not_found:
                raise ProviderThreadNotFoundError(str(error)) from error
            raise
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
            if self._open:
                await self._client.__aexit__(None, None, None)
                self._open = False
            self._closed = True


@dataclass
class _ThreadEntry:
    reference: ThreadRef
    provider_thread: CodexThreadPort
    lock: asyncio.Lock


@dataclass
class _Reservation:
    particle_id: str
    workspace: Path
    done: asyncio.Event


@dataclass
class _IdentityReservation:
    reference: ThreadRef
    done: asyncio.Event


class LocalCodexRuntime:
    """Isolated, bounded local Codex threads with explicit ownership.

    An injected client is caller-owned by default.  Closing a caller-owned runtime
    only forgets local mappings; remote provider-thread cleanup remains the
    caller's responsibility.
    """

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
        self._workspace_threads: dict[Path, str] = {}
        self._reservations: dict[str, _Reservation] = {}
        self._reserved_workspaces: set[Path] = set()
        self._identity_reservations: dict[str, _IdentityReservation] = {}
        self._provider_reservations: dict[str, _IdentityReservation] = {}
        self._state_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "LocalCodexRuntime":
        self._require_available()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        await self.close()
        return False

    async def start_thread(self, particle_id: str, workspace: Path) -> ThreadRef:
        particle = _require_text(particle_id, "particle_id")
        cwd = _workspace(workspace)
        reservation = await self._reserve(particle, cwd)
        try:
            provider = await self._client.thread_start(
                model=self._model, cwd=cwd, sandbox=self._sandbox
            )
            reference = self._reference(provider, particle, 0, cwd)
            await self._commit_reservation(reservation, reference, provider)
            return reference
        except BaseException:
            await self._rollback_reservation(reservation)
            raise

    async def restore_thread(
        self, particle_id: str, workspace: Path, checkpoint: Mapping[str, JsonValue]
    ) -> ThreadRef:
        particle = _require_text(particle_id, "particle_id")
        cwd = _workspace(workspace)
        self._require_available()
        bootstrap = _canonical_checkpoint(checkpoint)
        reference = _checkpoint_reference(json.loads(bootstrap))
        if reference.particle_id != particle or reference.workspace != cwd:
            raise ValueError("checkpoint thread identity does not match restore request")
        if reference.provider_id is None:
            raise ValueError("checkpoint thread has no provider_id")
        reservation = await self._reserve(particle, cwd)
        try:
            try:
                provider = await self._client.thread_resume(
                    reference.provider_id,
                    model=self._model,
                    cwd=cwd,
                    sandbox=self._sandbox,
                )
            except ProviderThreadNotFoundError:
                provider = await self._client.thread_start(
                    model=self._model, cwd=cwd, sandbox=self._sandbox
                )
                restored = self._reference(
                    provider, particle, reference.generation + 1, cwd
                )
                if (
                    restored.provider_id == reference.provider_id
                    or restored.logical_id == reference.logical_id
                ):
                    raise ValueError(
                        "provider-loss recovery must create a distinct thread"
                    )
                identity_reservation = await self._reserve_identity(restored)
                try:
                    self._response(
                        await provider.run(
                            bootstrap,
                            cwd=cwd,
                            output_schema=None,
                            sandbox=self._sandbox,
                        )
                    )
                    await self._commit_reservation(
                        reservation,
                        restored,
                        provider,
                        identity_reservation=identity_reservation,
                    )
                except BaseException:
                    await self._rollback_identity_reservation(identity_reservation)
                    raise
            else:
                restored = self._reference(
                    provider, particle, reference.generation, cwd
                )
                if restored != reference:
                    raise ValueError(
                        "resumed SDK thread identity does not match checkpoint"
                    )
                await self._commit_reservation(reservation, restored, provider)
            return restored
        except BaseException:
            await self._rollback_reservation(reservation)
            raise

    async def run_stage(
        self, thread: ThreadRef, request: StageRequest
    ) -> StageResponse:
        if not isinstance(request, StageRequest):
            raise TypeError("request must be a StageRequest")
        entry = await self._entry(thread)
        async with entry.lock:
            await self._revalidate_entry(entry, thread)
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
            await self._revalidate_entry(entry, thread)
            provider = await self._client.thread_start(
                model=self._model, cwd=thread.workspace, sandbox=self._sandbox
            )
            rotated = self._reference(
                provider, thread.particle_id, thread.generation + 1, thread.workspace
            )
            identity_reservation = await self._reserve_identity(
                rotated, replacing=entry
            )
            try:
                result = await provider.run(
                    bootstrap,
                    cwd=thread.workspace,
                    output_schema=None,
                    sandbox=self._sandbox,
                )
                self._response(result)
                async with self._state_lock:
                    self._require_available()
                    self._assert_current_locked(entry, thread)
                    self._assert_identity_reservation_locked(identity_reservation)
                    self._insert_locked(
                        rotated,
                        provider,
                        replacing=entry,
                        identity_reservation=identity_reservation,
                    )
                    self._release_identity_reservation_locked(identity_reservation)
            except BaseException:
                await self._rollback_identity_reservation(identity_reservation)
                raise
            return rotated

    async def close_thread(self, thread: ThreadRef) -> None:
        entry = await self._entry(thread)
        async with entry.lock:
            async with self._state_lock:
                self._require_available()
                self._assert_current_locked(entry, thread)
                self._remove_locked(entry)

    async def close(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            task = self._close_task
            if task is None or task.done():
                self._closing = True
                task = asyncio.create_task(self._close_impl())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_impl(self) -> None:
        try:
            async with self._state_lock:
                reservation_events = tuple(
                    reservation.done for reservation in self._reservations.values()
                ) + tuple(
                    reservation.done
                    for reservation in self._identity_reservations.values()
                )
            if reservation_events:
                await asyncio.gather(*(event.wait() for event in reservation_events))
            async with self._state_lock:
                entries = tuple(self._entries.values())
            for entry in entries:
                async with entry.lock:
                    pass
            if self._own_client:
                await self._client.close()
            async with self._state_lock:
                self._entries.clear()
                self._particle_threads.clear()
                self._workspace_threads.clear()
                self._reservations.clear()
                self._reserved_workspaces.clear()
                self._identity_reservations.clear()
                self._provider_reservations.clear()
                self._closed = True
                self._closing = False
        except BaseException:
            async with self._state_lock:
                self._closing = False
                self._close_task = None
            raise

    async def _entry(self, thread: ThreadRef) -> _ThreadEntry:
        if not isinstance(thread, ThreadRef):
            raise TypeError("thread must be a ThreadRef")
        self._require_available()
        async with self._state_lock:
            self._require_available()
            entry = self._entries.get(thread.logical_id)
            if entry is None:
                raise ValueError("unknown Codex thread")
            if entry.reference != thread:
                raise ValueError("Codex thread identity mismatch")
            return entry

    async def _revalidate_entry(
        self, entry: _ThreadEntry, reference: ThreadRef
    ) -> None:
        async with self._state_lock:
            self._require_available()
            self._assert_current_locked(entry, reference)

    def _assert_current_locked(
        self, entry: _ThreadEntry, reference: ThreadRef
    ) -> None:
        if self._entries.get(reference.logical_id) is not entry:
            raise ValueError("stale or unknown Codex thread")
        if entry.reference != reference:
            raise ValueError("Codex thread identity mismatch")

    async def _reserve(self, particle_id: str, workspace: Path) -> _Reservation:
        async with self._state_lock:
            self._require_available()
            if (
                particle_id in self._particle_threads
                or particle_id in self._reservations
            ):
                raise ValueError(
                    "particle already has an active or reserved Codex thread"
                )
            if (
                workspace in self._workspace_threads
                or workspace in self._reserved_workspaces
            ):
                raise ValueError(
                    "workspace already has an active or reserved Codex thread"
                )
            reservation = _Reservation(particle_id, workspace, asyncio.Event())
            self._reservations[particle_id] = reservation
            self._reserved_workspaces.add(workspace)
            return reservation

    async def _rollback_reservation(self, reservation: _Reservation) -> None:
        async with self._state_lock:
            if self._reservations.get(reservation.particle_id) is reservation:
                self._reservations.pop(reservation.particle_id)
                self._reserved_workspaces.discard(reservation.workspace)
            reservation.done.set()

    async def _commit_reservation(
        self,
        reservation: _Reservation,
        reference: ThreadRef,
        provider: CodexThreadPort,
        *,
        identity_reservation: _IdentityReservation | None = None,
    ) -> None:
        async with self._state_lock:
            if self._reservations.get(reservation.particle_id) is not reservation:
                raise ValueError("Codex thread reservation is stale")
            if identity_reservation is not None:
                self._assert_identity_reservation_locked(identity_reservation)
                if identity_reservation.reference != reference:
                    raise ValueError("Codex identity reservations do not match")
            if self._closing or self._closed:
                self._reservations.pop(reservation.particle_id)
                self._reserved_workspaces.discard(reservation.workspace)
                if identity_reservation is not None:
                    self._release_identity_reservation_locked(identity_reservation)
                reservation.done.set()
                raise RuntimeError("LocalCodexRuntime is closing or closed")
            self._insert_locked(
                reference,
                provider,
                identity_reservation=identity_reservation,
            )
            if identity_reservation is not None:
                self._release_identity_reservation_locked(identity_reservation)
            self._reservations.pop(reservation.particle_id)
            self._reserved_workspaces.discard(reservation.workspace)
            reservation.done.set()

    async def _reserve_identity(
        self,
        reference: ThreadRef,
        *,
        replacing: _ThreadEntry | None = None,
    ) -> _IdentityReservation:
        async with self._state_lock:
            self._require_available()
            if replacing is not None:
                self._assert_current_locked(replacing, replacing.reference)
                if (
                    reference.logical_id == replacing.reference.logical_id
                    or reference.provider_id == replacing.reference.provider_id
                ):
                    raise ValueError(
                        "rotation must create distinct logical and provider threads"
                    )
            self._assert_identity_available_locked(reference, replacing=replacing)
            reservation = _IdentityReservation(reference, asyncio.Event())
            self._identity_reservations[reference.logical_id] = reservation
            assert reference.provider_id is not None
            self._provider_reservations[reference.provider_id] = reservation
            return reservation

    async def _rollback_identity_reservation(
        self, reservation: _IdentityReservation
    ) -> None:
        async with self._state_lock:
            self._release_identity_reservation_locked(reservation)

    def _assert_identity_reservation_locked(
        self, reservation: _IdentityReservation
    ) -> None:
        reference = reservation.reference
        if self._identity_reservations.get(reference.logical_id) is not reservation:
            raise ValueError("Codex logical thread reservation is stale")
        if (
            reference.provider_id is None
            or self._provider_reservations.get(reference.provider_id) is not reservation
        ):
            raise ValueError("Codex provider thread reservation is stale")

    def _release_identity_reservation_locked(
        self, reservation: _IdentityReservation
    ) -> None:
        reference = reservation.reference
        if self._identity_reservations.get(reference.logical_id) is reservation:
            self._identity_reservations.pop(reference.logical_id)
        if (
            reference.provider_id is not None
            and self._provider_reservations.get(reference.provider_id) is reservation
        ):
            self._provider_reservations.pop(reference.provider_id)
        reservation.done.set()

    def _assert_identity_available_locked(
        self,
        reference: ThreadRef,
        *,
        replacing: _ThreadEntry | None = None,
        identity_reservation: _IdentityReservation | None = None,
    ) -> None:
        if reference.provider_id is None:
            raise ValueError("Codex provider thread identity is missing")
        logical_entry = self._entries.get(reference.logical_id)
        if logical_entry is not None and logical_entry is not replacing:
            raise ValueError("Codex logical thread collision")
        reserved_logical = self._identity_reservations.get(reference.logical_id)
        if (
            reserved_logical is not None
            and reserved_logical is not identity_reservation
        ):
            raise ValueError("Codex logical thread collision")
        if any(
            entry is not replacing
            and entry.reference.provider_id == reference.provider_id
            for entry in self._entries.values()
        ):
            raise ValueError("Codex provider thread collision")
        reserved_provider = self._provider_reservations.get(reference.provider_id)
        if (
            reserved_provider is not None
            and reserved_provider is not identity_reservation
        ):
            raise ValueError("Codex provider thread collision")

    def _insert_locked(
        self,
        reference: ThreadRef,
        provider: CodexThreadPort,
        *,
        replacing: _ThreadEntry | None = None,
        identity_reservation: _IdentityReservation | None = None,
    ) -> None:
        self._assert_identity_available_locked(
            reference,
            replacing=replacing,
            identity_reservation=identity_reservation,
        )
        if reference.particle_id in self._particle_threads:
            current_logical = self._particle_threads[reference.particle_id]
            if replacing is None or current_logical != replacing.reference.logical_id:
                raise ValueError("particle already has an active Codex thread")
        if reference.workspace in self._workspace_threads:
            current_logical = self._workspace_threads[reference.workspace]
            if replacing is None or current_logical != replacing.reference.logical_id:
                raise ValueError("workspace already has an active Codex thread")
        if replacing is not None:
            self._remove_locked(replacing)
        self._entries[reference.logical_id] = _ThreadEntry(
            reference, provider, asyncio.Lock()
        )
        self._particle_threads[reference.particle_id] = reference.logical_id
        self._workspace_threads[reference.workspace] = reference.logical_id

    def _remove_locked(self, entry: _ThreadEntry) -> None:
        reference = entry.reference
        self._entries.pop(reference.logical_id, None)
        if self._particle_threads.get(reference.particle_id) == reference.logical_id:
            self._particle_threads.pop(reference.particle_id)
        if self._workspace_threads.get(reference.workspace) == reference.logical_id:
            self._workspace_threads.pop(reference.workspace)

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

    def _require_available(self) -> None:
        if self._closed:
            raise RuntimeError("LocalCodexRuntime is closed")
        if self._closing:
            raise RuntimeError("LocalCodexRuntime is closing")


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _workspace(value: object) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("workspace must be an absolute Path")
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("workspace must be an existing directory") from error
    if not resolved.is_dir():
        raise ValueError("workspace must be an existing directory")
    return resolved


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
    "CodexTransportInterruptedError",
    "CodexThreadPort",
    "CodexTurnResult",
    "CodexUsage",
    "LocalCodexRuntime",
    "LOCAL_CODEX_RUNTIME_VERSION",
    "OpenAICodexClientAdapter",
    "ProviderThreadNotFoundError",
]
