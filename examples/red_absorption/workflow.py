"""Production orchestration of MoleculeEditor and spectrum command boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, MutableMapping
import re
import threading
from typing import Protocol

from multi_agent_pso.core import AgentStage
from multi_agent_pso.protocols import ToolContext, ToolRequest, ToolResult, ToolStatus
from multi_agent_pso.tools import JsonCommandProvider, JsonCommandStatus
from multi_agent_pso.resources import BudgetClaimStatus, DurableBudgetLedger

from .evaluator import EVALUATOR_VERSION
from .geometry import EvaluatedGeometry, GeometryAtom
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


class RedAbsorptionWorkflowResources:
    """Run-scoped cache, concurrency gate, locks, and counters bound to one event loop."""

    def __init__(
        self,
        inputs: RedAbsorptionRunInputs,
        cache: MutableMapping[CacheKey, SpectrumResult],
        max_new_evaluations: int,
        ledger: DurableBudgetLedger | None,
        run_id: str | None,
    ):
        self._concurrency = inputs.evaluation_concurrency
        self._protocol_hash = inputs.calculation_protocol.protocol_hash
        self._timeout_seconds = inputs.spectrum_timeout_seconds
        self._cache = cache
        self._locks: dict[CacheKey, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._spectrum_slots = asyncio.Semaphore(self._concurrency)
        self._execution_count = 0
        self._cache_hit_count = 0
        self._max_new_evaluations = max_new_evaluations
        self._budget_lock = asyncio.Lock()
        self._ledger = ledger
        self._run_id = run_id
        self._loop = None
        self._loop_guard = threading.Lock()

    @classmethod
    def from_inputs(
        cls,
        inputs: RedAbsorptionRunInputs,
        cache: MutableMapping[CacheKey, SpectrumResult] | None = None,
        *,
        max_new_evaluations: int = 25,
        ledger: DurableBudgetLedger | None = None,
        run_id: str | None = None,
    ) -> "RedAbsorptionWorkflowResources":
        if not isinstance(inputs, RedAbsorptionRunInputs):
            raise TypeError("inputs must be RedAbsorptionRunInputs")
        if cache is not None and not isinstance(cache, MutableMapping):
            raise TypeError("cache must be a mutable mapping")
        if type(max_new_evaluations) is not int or max_new_evaluations <= 0:
            raise ValueError("max_new_evaluations must be a positive integer")
        if (ledger is None) != (run_id is None):
            raise ValueError("ledger and run_id must be supplied together")
        return cls(
            inputs,
            {} if cache is None else cache,
            max_new_evaluations,
            ledger,
            run_id,
        )

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def execution_count(self) -> int:
        return (
            self._execution_count
            if self._ledger is None
            else self._ledger.count(self._run_id)
        )

    @property
    def cache_hit_count(self) -> int:
        return self._cache_hit_count

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._loop_guard:
            if self._loop is None:
                self._loop = loop
            elif self._loop is not loop:
                raise RuntimeError(
                    "workflow resources cannot be used across event loops"
                )

    async def _lock_for(self, key: CacheKey) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(key, asyncio.Lock())

    @staticmethod
    def _ledger_key(key: CacheKey) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(key, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _recover(self, key: CacheKey) -> SpectrumResult | None:
        if self._ledger is None:
            return None
        payload = self._ledger.get(self._run_id, self._ledger_key(key))
        return None if payload is None else SpectrumResult.model_validate(payload)

    async def _recover_async(self, key: CacheKey) -> SpectrumResult | None:
        return await asyncio.to_thread(self._recover, key)

    async def _commit_result(self, key: CacheKey, result: SpectrumResult) -> None:
        if self._ledger is not None:
            await asyncio.to_thread(
                self._ledger.commit,
                self._run_id,
                self._ledger_key(key),
                result.model_dump(mode="json"),
            )

    async def _commit_failure(
        self, key: CacheKey, status: str, message: str
    ) -> None:
        if self._ledger is not None:
            await asyncio.to_thread(
                self._ledger.fail,
                self._run_id,
                self._ledger_key(key),
                status,
                message,
            )

    async def _failure(self, key: CacheKey) -> Mapping[str, object] | None:
        if self._ledger is None:
            return None
        failure = await asyncio.to_thread(
            self._ledger.get_failure, self._run_id, self._ledger_key(key)
        )
        return failure if isinstance(failure, Mapping) else None

    async def _commit_failure_preserving(
        self,
        key: CacheKey,
        status: str,
        message: str,
        primary: BaseException,
    ) -> None:
        task = asyncio.create_task(self._commit_failure(key, status, message))
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                primary.add_note(f"evaluation terminal-write cancellation: {error}")
        try:
            task.result()
        except BaseException as error:
            primary.add_note(
                "evaluation terminal-write failure: "
                f"{type(error).__name__}: {error}"
            )

    async def _claim_execution(self, key: CacheKey) -> BudgetClaimStatus:
        async with self._budget_lock:
            if self._ledger is not None:
                return await asyncio.to_thread(
                    self._ledger.claim,
                    self._run_id,
                    self._ledger_key(key),
                    self._max_new_evaluations,
                )
            if self._execution_count >= self._max_new_evaluations:
                return BudgetClaimStatus.EXHAUSTED
            self._execution_count += 1
            return BudgetClaimStatus.RESERVED

    async def _wait_for_terminal(
        self, key: CacheKey
    ) -> tuple[SpectrumResult | None, Mapping[str, object] | None]:
        if self._ledger is None:
            return None, None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_seconds
        while True:
            recovered = await self._recover_async(key)
            if recovered is not None:
                return recovered, None
            failure = await self._failure(key)
            if failure is not None:
                return None, failure
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None, None
            await asyncio.sleep(min(0.05, remaining))


def _parent_source(inputs: RedAbsorptionRunInputs) -> dict[str, object]:
    parent = inputs.parent
    if parent.kind == "smiles":
        return {"kind": "smiles", "value": parent.value}
    if parent.kind == "chemical_graph":
        return {"kind": "chemical_graph", "value": parent.value}
    return {"kind": "path", "path": parent.path, "format": parent.format}


def _evaluated_geometry_from_payload(
    payload: Mapping[str, object], *, charge: int, multiplicity: int
) -> EvaluatedGeometry:
    order = payload.get("coordinate_order")
    result = payload.get("geometry_result")
    if not isinstance(order, (list, tuple)) or not isinstance(result, Mapping):
        raise ValueError("MoleculeEditor geometry payload is missing")
    selected = result.get("selected_conformer_id")
    conformers = result.get("conformers")
    if type(selected) is not int or not isinstance(conformers, (list, tuple)):
        raise ValueError("MoleculeEditor selected conformer is missing")
    match = next(
        (
            conformer
            for conformer in conformers
            if isinstance(conformer, Mapping)
            and conformer.get("conformer_id") == selected
        ),
        None,
    )
    coordinates = match.get("coordinates") if isinstance(match, Mapping) else None
    if not isinstance(coordinates, (list, tuple)):
        raise ValueError("MoleculeEditor coordinates are missing")
    return EvaluatedGeometry(
        coordinate_order=tuple(order),
        coordinates=tuple(
            GeometryAtom(
                atom_id=coordinate["atom_id"],
                atomic_number=coordinate["atomic_number"],
                x_angstrom=coordinate["x_angstrom"],
                y_angstrom=coordinate["y_angstrom"],
                z_angstrom=coordinate["z_angstrom"],
            )
            for coordinate in coordinates
            if isinstance(coordinate, Mapping)
        ),
        charge=charge,
        multiplicity=multiplicity,
    )


def _spectrum_matches_source_geometry(
    spectrum: SpectrumResult, source_geometry_hash: str
) -> bool:
    provenance = spectrum.provenance
    if provenance.protocol.geometry_workflow == "b3lyp_sto3g_optimized":
        geometry = spectrum.evaluated_geometry
        return (
            provenance.source_geometry_hash == source_geometry_hash
            and geometry is not None
            and provenance.evaluation_geometry_hash == geometry.geometry_hash
            and provenance.geometry_hash == geometry.geometry_hash
        )
    return (
        provenance.geometry_hash == source_geometry_hash
        and provenance.source_geometry_hash in {None, source_geometry_hash}
        and provenance.evaluation_geometry_hash in {None, source_geometry_hash}
    )


def _violates_protection_policy(
    inputs: RedAbsorptionRunInputs, authoritative: Mapping[str, object]
) -> bool:
    if inputs.parent.protected_smarts:
        # SMARTS matching belongs to the CLI authority. Until its inspection
        # envelope exposes matched AtomIds, fail closed instead of guessing.
        return True
    protected = set(inputs.parent.protected_atom_ids)
    if not protected:
        return False
    graph = authoritative.get("inspected_graph")
    commands = authoritative.get("commands")
    if not isinstance(graph, Mapping) or not isinstance(commands, (list, tuple)):
        return True
    if len(commands) > 1 and any(
        isinstance(command, Mapping)
        and command.get("operation") in {"detach_fragment", "substitute_fragment"}
        for command in commands
    ):
        # A multi-command transaction can change the component cut before a
        # fragment removal. Fail closed rather than reimplement the editor.
        return True
    atoms = {
        atom.get("atom_id")
        for atom in graph.get("atoms", [])
        if isinstance(atom, Mapping) and isinstance(atom.get("atom_id"), str)
    }
    adjacency = {atom: set() for atom in atoms}
    bond_atoms = {}
    for bond in graph.get("bonds", []):
        if not isinstance(bond, Mapping):
            return True
        identity = bond.get("bond_id")
        begin = bond.get("begin_atom_id")
        end = bond.get("end_atom_id")
        if (
            not isinstance(identity, str)
            or begin not in adjacency
            or end not in adjacency
        ):
            return True
        bond_atoms[identity] = (begin, end)
        adjacency[begin].add(end)
        adjacency[end].add(begin)
    atom_fields = (
        "atom_id",
        "anchor_atom_id",
        "retained_atom_id",
        "begin",
        "end",
    )
    for command in commands:
        if not isinstance(command, Mapping):
            return True
        operation = command.get("operation")
        touched = {command.get(field) for field in atom_fields}
        endpoints = bond_atoms.get(command.get("bond_id"), ())
        touched.update(endpoints)
        if protected & touched:
            return True
        if operation in {"detach_fragment", "substitute_fragment"}:
            if len(endpoints) != 2:
                return True
            begin, end = endpoints
            retained = command.get("retained_atom_id")
            if retained not in adjacency:
                return True
            adjacency[begin].discard(end)
            adjacency[end].discard(begin)
            retained_component = set()
            pending = [retained]
            while pending:
                atom = pending.pop()
                if atom in retained_component:
                    continue
                retained_component.add(atom)
                pending.extend(adjacency[atom] - retained_component)
            discarded = set(adjacency) - retained_component
            if protected & discarded:
                return True
            for atom in discarded:
                for neighbor in adjacency[atom]:
                    adjacency[neighbor].discard(atom)
                adjacency.pop(atom, None)
            for identity, pair in tuple(bond_atoms.items()):
                if pair[0] in discarded or pair[1] in discarded:
                    bond_atoms.pop(identity, None)
            if operation == "substitute_fragment":
                # Fragment-local IDs are outside the protected parent namespace.
                opaque = f"@fragment-{len(adjacency)}"
                adjacency[retained].add(opaque)
                adjacency[opaque] = {retained}
        elif operation == "remove_atom":
            removed = command.get("atom_id")
            if removed not in adjacency or protected & adjacency[removed]:
                return True
            for neighbor in tuple(adjacency[removed]):
                adjacency[neighbor].discard(removed)
            adjacency.pop(removed)
        elif operation == "remove_bond" and len(endpoints) == 2:
            begin, end = endpoints
            adjacency[begin].discard(end)
            adjacency[end].discard(begin)
            bond_atoms.pop(command.get("bond_id"), None)
        elif operation == "add_bond":
            begin, end = command.get("begin"), command.get("end")
            if begin in adjacency and end in adjacency:
                adjacency[begin].add(end)
                adjacency[end].add(begin)
    return False


class RedAbsorptionWorkflowToolProvider:
    """Bound workflow provider; zero-argument construction remains fail-closed."""

    def __init__(self) -> None:
        self._inputs: RedAbsorptionRunInputs | None = None
        self._editor: MoleculeEditorLike | None = None
        self._spectrum: JsonCommandProvider | None = None
        self._resources: RedAbsorptionWorkflowResources | None = None
        self._own_spectrum = False
        self._state_lock = asyncio.Lock()
        self._active: set[asyncio.Event] = set()
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    def bind(
        cls,
        inputs: RedAbsorptionRunInputs,
        molecule_editor: MoleculeEditorLike,
        resources: RedAbsorptionWorkflowResources,
        *,
        spectrum: JsonCommandProvider | None = None,
        own_spectrum: bool | None = None,
    ) -> "RedAbsorptionWorkflowToolProvider":
        """Bind a run. Injected editor/cache are caller-owned; spectrum ownership is explicit."""
        if not isinstance(inputs, RedAbsorptionRunInputs):
            raise TypeError("inputs must be RedAbsorptionRunInputs")
        if not callable(getattr(molecule_editor, "inspect", None)) or not callable(
            getattr(molecule_editor, "edit", None)
        ):
            raise TypeError("molecule_editor must provide inspect and edit")
        if not isinstance(resources, RedAbsorptionWorkflowResources):
            raise TypeError("resources must be RedAbsorptionWorkflowResources")
        if (
            resources.concurrency != inputs.evaluation_concurrency
            or resources._protocol_hash != inputs.calculation_protocol.protocol_hash
        ):
            raise ValueError("workflow resources do not match run inputs")
        instance = cls()
        instance._inputs = inputs
        instance._editor = molecule_editor
        instance._resources = resources
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
        return instance

    @property
    def execution_count(self) -> int:
        return 0 if self._resources is None else self._resources.execution_count

    @property
    def cache_hit_count(self) -> int:
        return 0 if self._resources is None else self._resources.cache_hit_count

    async def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        if self._resources is not None:
            self._resources._bind_loop()
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
            self._active.discard(active)
            active.set()

    async def _execute_active(
        self, request: ToolRequest, context: ToolContext
    ) -> ToolResult:
        if (
            self._inputs is None
            or self._editor is None
            or self._spectrum is None
            or self._resources is None
        ):
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
        if _violates_protection_policy(inputs, authoritative):
            return ToolResult(
                ToolStatus.REJECTED,
                error="edit violates protected parent structure policy",
            )
        resources = self._resources
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
        lock = await resources._lock_for(key)
        async with lock:
            spectrum_result = resources._cache.get(key)
            if spectrum_result is None:
                spectrum_result = await resources._recover_async(key)
            spectrum_process = None
            if spectrum_result is None:
                async with resources._spectrum_slots:
                    spectrum_result = resources._cache.get(key)
                    if spectrum_result is None:
                        spectrum_result = await resources._recover_async(key)
                    if spectrum_result is not None:
                        resources._cache_hit_count += 1
                        cache_hit = True
                    else:
                        claim = await resources._claim_execution(key)
                        if claim is BudgetClaimStatus.COMPLETED:
                            spectrum_result = await resources._recover_async(key)
                            if spectrum_result is None:
                                return ToolResult(
                                    ToolStatus.FAILED,
                                    {"cache_key": list(key), "cache_hit": False},
                                    error="completed evaluation cache is missing",
                                )
                            resources._cache_hit_count += 1
                            cache_hit = True
                        elif claim is BudgetClaimStatus.FAILED:
                            failure = await resources._failure(key)
                            status = (
                                ToolStatus.TIMEOUT
                                if isinstance(failure, Mapping)
                                and failure.get("status") == "TIMEOUT"
                                else ToolStatus.FAILED
                            )
                            message = (
                                failure.get("message")
                                if isinstance(failure, Mapping)
                                else "previous evaluation failed"
                            )
                            return ToolResult(
                                status,
                                {"cache_key": list(key), "cache_hit": False},
                                error=str(message),
                            )
                        elif claim is BudgetClaimStatus.PENDING:
                            spectrum_result, failure = (
                                await resources._wait_for_terminal(key)
                            )
                            if failure is not None:
                                return ToolResult(
                                    ToolStatus.TIMEOUT
                                    if failure.get("status") == "TIMEOUT"
                                    else ToolStatus.FAILED,
                                    {"cache_key": list(key), "cache_hit": False},
                                    error=str(failure.get("message", "evaluation failed")),
                                )
                            if spectrum_result is None:
                                return ToolResult(
                                    ToolStatus.TIMEOUT,
                                    {"cache_key": list(key), "cache_hit": False},
                                    error="matching evaluation remains pending",
                                )
                            resources._cache_hit_count += 1
                            cache_hit = True
                        elif claim is BudgetClaimStatus.EXHAUSTED:
                            return ToolResult(
                                ToolStatus.REJECTED,
                                {"cache_key": list(key), "cache_hit": False},
                                error="maximum new evaluation budget exhausted",
                            )
                        if claim is BudgetClaimStatus.RESERVED:
                            try:
                                command_payload = {
                                    "candidate": payload,
                                    "chemical_identity_hash": chemical_hash,
                                    "state_hash": state_hash,
                                    "geometry_hash": geometry_hash,
                                    "protocol": inputs.calculation_protocol.model_dump(
                                        mode="json"
                                    ),
                                }
                                if (
                                    inputs.calculation_protocol.geometry_workflow
                                    == "b3lyp_sto3g_optimized"
                                ):
                                    source_geometry = _evaluated_geometry_from_payload(
                                        payload,
                                        charge=inputs.calculation_protocol.charge,
                                        multiplicity=inputs.calculation_protocol.multiplicity,
                                    )
                                    command_payload.update(
                                        {
                                            "source_geometry": source_geometry.model_dump(
                                                mode="json"
                                            ),
                                            "source_geometry_hash": geometry_hash,
                                        }
                                    )
                                command_result = await self._spectrum.execute_json(
                                    command_payload,
                                    cwd=context.workspace,
                                    timeout_seconds=inputs.spectrum_timeout_seconds,
                                )
                            except asyncio.CancelledError as cancellation:
                                await resources._commit_failure_preserving(
                                    key,
                                    "FAILED",
                                    "spectrum command was cancelled",
                                    cancellation,
                                )
                                raise
                            except TimeoutError:
                                await resources._commit_failure(
                                    key, "TIMEOUT", "spectrum command timed out"
                                )
                                return ToolResult(
                                    ToolStatus.TIMEOUT,
                                    {"cache_key": list(key), "cache_hit": False},
                                    error="spectrum command timed out",
                                )
                            except Exception as error:
                                await resources._commit_failure(
                                    key,
                                    "FAILED",
                                    f"spectrum command raised {type(error).__name__}",
                                )
                                return ToolResult(
                                    ToolStatus.FAILED,
                                    {"cache_key": list(key), "cache_hit": False},
                                    error=f"spectrum command failed: {type(error).__name__}",
                                )
                            spectrum_process = {
                                "status": command_result.status.value,
                                "exit_code": getattr(command_result, "exit_code", None),
                                "elapsed_seconds": getattr(
                                    command_result, "elapsed_seconds", None
                                ),
                            }
                            failed_process_payload = {
                                "cache_key": list(key),
                                "cache_hit": False,
                                "spectrum_process": spectrum_process,
                            }
                            if command_result.status is JsonCommandStatus.TIMEOUT:
                                await resources._commit_failure(
                                    key, "TIMEOUT", "spectrum command timed out"
                                )
                                return ToolResult(
                                    ToolStatus.TIMEOUT,
                                    failed_process_payload,
                                    error="spectrum command timed out",
                                )
                            if (
                                command_result.status is not JsonCommandStatus.SUCCESS
                                or command_result.stdout_text is None
                            ):
                                await resources._commit_failure(
                                    key,
                                    "FAILED",
                                    f"spectrum command failed: {command_result.status.value}",
                                )
                                return ToolResult(
                                    ToolStatus.FAILED,
                                    failed_process_payload,
                                    error=f"spectrum command failed: {command_result.status.value}",
                                )
                            try:
                                spectrum_result = SpectrumResult.model_validate_json(
                                    command_result.stdout_text
                                )
                            except (TypeError, ValueError) as error:
                                await resources._commit_failure(
                                    key,
                                    "FAILED",
                                    f"invalid spectrum result: {type(error).__name__}",
                                )
                                return ToolResult(
                                    ToolStatus.FAILED,
                                    failed_process_payload,
                                    error=f"invalid spectrum result: {type(error).__name__}",
                                )
                            if (
                                spectrum_result.provenance.protocol
                                != inputs.calculation_protocol
                                or not _spectrum_matches_source_geometry(
                                    spectrum_result, geometry_hash
                                )
                            ):
                                await resources._commit_failure(
                                    key, "FAILED", "spectrum provenance mismatch"
                                )
                                return ToolResult(
                                    ToolStatus.FAILED,
                                    failed_process_payload,
                                    error="spectrum provenance mismatch",
                                )
                            await resources._commit_result(key, spectrum_result)
                            resources._cache[key] = spectrum_result
                            cache_hit = False
            else:
                if (
                    not isinstance(spectrum_result, SpectrumResult)
                    or spectrum_result.provenance.protocol
                    != inputs.calculation_protocol
                    or not _spectrum_matches_source_geometry(
                        spectrum_result, geometry_hash
                    )
                ):
                    return ToolResult(
                        ToolStatus.FAILED, error="cached spectrum provenance mismatch"
                    )
                resources._cache_hit_count += 1
                cache_hit = True
        result_payload = _plain_json(payload)
        result_payload.update(
            {
                "spectrum_result": spectrum_result.model_dump(mode="json"),
                "cache_key": list(key),
                "cache_hit": cache_hit,
                "spectrum_process": spectrum_process,
            }
        )
        return ToolResult(ToolStatus.SUCCESS, result_payload)

    async def aclose(self) -> None:
        if self._resources is not None:
            self._resources._bind_loop()
        async with self._state_lock:
            if self._closed:
                return
            previous = self._close_task
            retry = previous is None or (
                previous.done()
                and (previous.cancelled() or previous.exception() is not None)
            )
            if retry:
                self._closing = True
                self._close_task = asyncio.create_task(self._close_active())
                self._close_task.add_done_callback(self._observe_close_task)
            close_task = self._close_task
        await asyncio.shield(close_task)

    @staticmethod
    def _observe_close_task(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    async def _close_active(self) -> None:
        try:
            async with self._state_lock:
                active = tuple(self._active)
            if active:
                await asyncio.gather(*(event.wait() for event in active))
            if self._own_spectrum and self._spectrum is not None:
                await self._spectrum.aclose()
        except BaseException:
            self._closing = False
            raise
        async with self._state_lock:
            self._closed = True
            self._closing = False


__all__ = [
    "CacheKey",
    "RedAbsorptionWorkflowResources",
    "RedAbsorptionWorkflowToolProvider",
]
