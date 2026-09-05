"""Production orchestration of MoleculeEditor and spectrum command boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, MutableMapping
import re
from typing import Protocol

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolContext, ToolRequest, ToolResult, ToolStatus
from multi_agent_pso.tools import JsonCommandProvider, JsonCommandStatus

from .evaluator import EVALUATOR_VERSION
from .inputs import RedAbsorptionRunInputs
from .models import SpectrumResult


CacheKey = tuple[str, str, str, str]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _plain_json(value):
    if isinstance(value, Mapping):
        return {key: _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(nested) for nested in value]
    return value


class MoleculeEditorLike(Protocol):
    async def inspect(
        self, source, *, cwd, geometry=None, artifacts=None, timeout=60
    ): ...
    async def edit(
        self,
        inspection,
        commands,
        *,
        cwd,
        geometry=None,
        artifacts=None,
        timeout=60,
        attempt=1,
    ): ...


def _parent_source(inputs: RedAbsorptionRunInputs) -> dict[str, object]:
    parent = inputs.parent
    if parent.kind == "smiles":
        return {"kind": "smiles", "value": parent.value}
    if parent.kind == "chemical_graph":
        return {"kind": "chemical_graph", "value": parent.value}
    return {"kind": "path", "path": parent.path, "format": parent.format}


class RedAbsorptionWorkflowToolProvider:
    """Bound workflow provider; zero-argument construction remains fail-closed."""

    def __init__(self) -> None:
        self._inputs: RedAbsorptionRunInputs | None = None
        self._editor: MoleculeEditorLike | None = None
        self._spectrum: JsonCommandProvider | None = None
        self._own_spectrum = False
        self._cache: MutableMapping[CacheKey, SpectrumResult] = {}
        self._locks: dict[CacheKey, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._spectrum_slots = asyncio.Semaphore(1)
        self._state_lock = asyncio.Lock()
        self._active: set[asyncio.Event] = set()
        self._closing = False
        self._execution_count = 0
        self._cache_hit_count = 0
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    def bind(
        cls,
        inputs: RedAbsorptionRunInputs,
        molecule_editor: MoleculeEditorLike,
        *,
        spectrum: JsonCommandProvider | None = None,
        cache: MutableMapping[CacheKey, SpectrumResult] | None = None,
        own_spectrum: bool | None = None,
    ) -> "RedAbsorptionWorkflowToolProvider":
        """Bind a run. Injected editor/cache are caller-owned; spectrum ownership is explicit."""
        if not isinstance(inputs, RedAbsorptionRunInputs):
            raise TypeError("inputs must be RedAbsorptionRunInputs")
        if not callable(getattr(molecule_editor, "inspect", None)) or not callable(
            getattr(molecule_editor, "edit", None)
        ):
            raise TypeError("molecule_editor must provide inspect and edit")
        instance = cls()
        instance._inputs = inputs
        instance._editor = molecule_editor
        instance._spectrum_slots = asyncio.Semaphore(inputs.evaluation_concurrency)
        if spectrum is None:
            instance._spectrum = JsonCommandProvider(inputs.spectrum_argv)
            instance._own_spectrum = True
        else:
            if not isinstance(spectrum, JsonCommandProvider):
                raise TypeError("spectrum must be JsonCommandProvider")
            instance._spectrum = spectrum
            instance._own_spectrum = False if own_spectrum is None else own_spectrum
        if own_spectrum is not None and type(own_spectrum) is not bool:
            raise TypeError("own_spectrum must be boolean")
        if own_spectrum is not None and spectrum is None and own_spectrum is not True:
            raise ValueError("internally created spectrum provider must be owned")
        if cache is not None and not isinstance(cache, MutableMapping):
            raise TypeError("cache must be a mutable mapping")
        if cache is not None:
            instance._cache = cache
        return instance

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count

    async def _lock_for(self, key: CacheKey) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(key, asyncio.Lock())

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        active = asyncio.Event()
        async with self._state_lock:
            if self._closing or self._closed:
                raise RuntimeError(
                    "red-absorption workflow provider is closing or closed"
                )
            self._active.add(active)
        try:
            return await self._execute_active(request, context)
        finally:
            async with self._state_lock:
                self._active.discard(active)
            active.set()

    async def _execute_active(
        self, request: ToolRequest, context: ToolContext
    ) -> ToolResult:
        if self._inputs is None or self._editor is None or self._spectrum is None:
            return ToolResult(
                ToolStatus.REJECTED,
                error="red-absorption workflow requires validated run inputs",
            )
        if (
            request.provider != "molecule_editor"
            or request.operation != "edit"
            or context.stage is not AgentStage.EXECUTING
            or context.attempt > 2
        ):
            return ToolResult(
                ToolStatus.REJECTED, error="invalid red-absorption workflow request"
            )
        proposal = context.to_json()["metadata"].get("proposal")
        if (
            not isinstance(proposal, Mapping)
            or proposal.get("provider") != request.provider
            or proposal.get("operation") != request.operation
        ):
            return ToolResult(
                ToolStatus.REJECTED, error="tool context lacks persisted proposal"
            )
        authoritative = proposal.get("tool_payload")
        request_payload = request.to_json()["payload"]
        if not isinstance(authoritative, Mapping) or request_payload != authoritative:
            return ToolResult(
                ToolStatus.REJECTED,
                error="tool request differs from persisted proposal",
            )
        inputs = self._inputs
        geometry = inputs.geometry.model_dump(mode="json")
        try:
            inspection = await self._editor.inspect(
                _parent_source(inputs),
                cwd=context.workspace,
                geometry=geometry,
                timeout=inputs.spectrum_timeout_seconds,
            )
        except TimeoutError:
            return ToolResult(ToolStatus.TIMEOUT, error="parent inspection timed out")
        except Exception as error:
            return ToolResult(
                ToolStatus.FAILED,
                error=f"parent inspection failed: {type(error).__name__}",
            )
        if (
            not inspection.processed
            or inspection.chemical_status != "VALID"
            or inspection.geometry_status != "READY"
            or not inspection.ready_for_evaluator
            or inspection.candidate is None
            or inspection.payload is None
        ):
            return ToolResult(
                ToolStatus.FAILED, error="parent inspection is not evaluator-ready"
            )
        if (
            _plain_json(inspection.candidate)
            != _plain_json(authoritative.get("inspected_graph"))
            or inspection.candidate.get("state_hash")
            != authoritative.get("inspected_source_hash")
            or inspection.payload.get("geometry_hash")
            != authoritative.get("inspected_geometry_hash")
        ):
            return ToolResult(
                ToolStatus.REJECTED,
                error="current inspection differs from persisted proposal",
            )
        try:
            edit = await self._editor.edit(
                inspection,
                authoritative["commands"],
                cwd=context.workspace,
                geometry=geometry,
                timeout=inputs.spectrum_timeout_seconds,
                attempt=context.attempt + 1,
            )
        except ValueError as error:
            return ToolResult(
                ToolStatus.REJECTED, error=f"MoleculeEditor rejected edit: {error}"
            )
        except TimeoutError:
            return ToolResult(ToolStatus.TIMEOUT, error="MoleculeEditor edit timed out")
        except Exception as error:
            return ToolResult(
                ToolStatus.FAILED,
                error=f"MoleculeEditor edit failed: {type(error).__name__}",
            )
        if not edit.processed or edit.chemical_status != "VALID":
            return ToolResult(ToolStatus.REJECTED, error="MoleculeEditor rejected edit")
        if (
            edit.geometry_status != "READY"
            or not edit.ready_for_evaluator
            or edit.candidate is None
            or edit.payload is None
        ):
            return ToolResult(
                ToolStatus.FAILED, error="edited candidate geometry is not ready"
            )
        payload = edit.payload
        chemical_hash = payload.get("chemical_identity_hash")
        state_hash = payload.get("state_hash")
        geometry_hash = payload.get("geometry_hash")
        commands = payload.get("committed_commands")
        if not all(
            isinstance(value, str) and _SHA256.fullmatch(value)
            for value in (chemical_hash, state_hash, geometry_hash)
        ) or _plain_json(commands) != _plain_json(authoritative["commands"]):
            return ToolResult(
                ToolStatus.FAILED, error="edited candidate identity is invalid"
            )
        key: CacheKey = (
            chemical_hash,
            geometry_hash,
            inputs.calculation_protocol.protocol_hash,
            EVALUATOR_VERSION,
        )
        lock = await self._lock_for(key)
        async with lock:
            spectrum_result = self._cache.get(key)
            if spectrum_result is None:
                async with self._spectrum_slots:
                    spectrum_result = self._cache.get(key)
                    if spectrum_result is not None:
                        self._cache_hit_count += 1
                        cache_hit = True
                    else:
                        self._execution_count += 1
                        command_result = await self._spectrum.execute_json(
                            {
                                "candidate": payload,
                                "chemical_identity_hash": chemical_hash,
                                "state_hash": state_hash,
                                "geometry_hash": geometry_hash,
                                "protocol": inputs.calculation_protocol.model_dump(
                                    mode="json"
                                ),
                            },
                            cwd=context.workspace,
                            timeout_seconds=inputs.spectrum_timeout_seconds,
                        )
                        if command_result.status is JsonCommandStatus.TIMEOUT:
                            return ToolResult(
                                ToolStatus.TIMEOUT, error="spectrum command timed out"
                            )
                        if (
                            command_result.status is not JsonCommandStatus.SUCCESS
                            or command_result.stdout_text is None
                        ):
                            return ToolResult(
                                ToolStatus.FAILED,
                                error=f"spectrum command failed: {command_result.status.value}",
                            )
                        try:
                            spectrum_result = SpectrumResult.model_validate_json(
                                command_result.stdout_text
                            )
                        except (TypeError, ValueError) as error:
                            return ToolResult(
                                ToolStatus.FAILED,
                                error=f"invalid spectrum result: {type(error).__name__}",
                            )
                        if (
                            spectrum_result.provenance.protocol
                            != inputs.calculation_protocol
                            or spectrum_result.provenance.geometry_hash != geometry_hash
                        ):
                            return ToolResult(
                                ToolStatus.FAILED, error="spectrum provenance mismatch"
                            )
                        self._cache[key] = spectrum_result
                        cache_hit = False
            else:
                if (
                    not isinstance(spectrum_result, SpectrumResult)
                    or spectrum_result.provenance.protocol
                    != inputs.calculation_protocol
                    or spectrum_result.provenance.geometry_hash != geometry_hash
                ):
                    return ToolResult(
                        ToolStatus.FAILED, error="cached spectrum provenance mismatch"
                    )
                self._cache_hit_count += 1
                cache_hit = True
        result_payload = _plain_json(payload)
        result_payload.update(
            {
                "spectrum_result": spectrum_result.model_dump(mode="json"),
                "cache_key": list(key),
                "cache_hit": cache_hit,
            }
        )
        return ToolResult(ToolStatus.SUCCESS, result_payload)

    async def aclose(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            if self._close_task is None:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_active())
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_active(self) -> None:
        try:
            async with self._state_lock:
                active = tuple(self._active)
            if active:
                await asyncio.gather(*(event.wait() for event in active))
            if self._own_spectrum and self._spectrum is not None:
                await self._spectrum.aclose()
        except BaseException:
            async with self._state_lock:
                self._closing = False
                self._close_task = None
            raise
        async with self._state_lock:
            self._closed = True
            self._closing = False


__all__ = ["CacheKey", "RedAbsorptionWorkflowToolProvider"]
