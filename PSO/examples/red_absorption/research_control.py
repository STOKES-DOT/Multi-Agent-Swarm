"""Molecular context for generic PSO; predictions are not mechanistic proofs."""

from collections.abc import Mapping
import hashlib
import json
import math
import re

from multi_agent_pso.core.topology import DistanceMatrixTopology


def assess_prediction(claim, parent_nm, child_nm, *, rollback=False, tolerance_nm=1.0):
    """Compare a predeclared prediction using a fixed decision deadband in nm.

    The deadband is not model uncertainty. All statuses are conditional on the
    fixed FLAME protocol; no status establishes a causal mechanism.
    """
    result = {
        "status": "INCONCLUSIVE",
        "delta_absorption_nm": None,
        "tolerance_nm": tolerance_nm,
        "evidence_level": "model_prediction",
    }
    if rollback:
        return {**result, "status": "NOT_TESTED", "reason": "rollback_parent"}
    if not isinstance(claim, Mapping) or parent_nm is None:
        return {**result, "reason": "missing_claim_or_parent_prediction"}
    direction = claim.get("direction")
    minimum = claim.get("minimum_change_nm", 0.0)
    for number in (parent_nm, child_nm, minimum, tolerance_nm):
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("prediction comparison requires finite numbers")
    if (
        minimum < 0
        or tolerance_nm < 0
        or direction not in {"red_shift", "blue_shift", "no_change"}
    ):
        raise ValueError("invalid prediction contract")
    delta = child_nm - parent_nm
    margin = (delta if direction == "red_shift" else -delta) - minimum
    if direction == "no_change":
        margin = minimum - abs(delta)
    status = (
        "SUPPORTED"
        if margin > tolerance_nm
        else "REFUTED" if margin < -tolerance_nm else "INCONCLUSIVE"
    )
    return {
        **result,
        "status": status,
        "delta_absorption_nm": delta,
        "direction": direction,
        "minimum_change_nm": minimum,
        "reason": "comparison_with_predeclared_prediction",
    }


def molecular_topology(snapshot, initial_smiles, k=3):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    fingerprints = {}
    for particle in snapshot.particles:
        continuation = particle.continuation_state
        smiles = (
            continuation.get("canonical_isomeric_smiles")
            if isinstance(continuation, Mapping)
            else initial_smiles
        )
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            raise ValueError("cannot define molecular neighbors without a valid parent")
        fingerprints[particle.particle_id] = generator.GetFingerprint(molecule)
    matrix = {
        i: {
            j: 1.0 - DataStructs.TanimotoSimilarity(a, b)
            for j, b in fingerprints.items()
        }
        for i, a in fingerprints.items()
    }
    return DistanceMatrixTopology(matrix, k)


def shared_text(value):
    text = str(value)[:2000]
    return re.sub(
        r"\b[ab]\d{4,}\b|@[A-Za-z0-9][A-Za-z0-9_-]*", "[source-local ID]", text
    )


def _reference(kind, run_id, particle_id, iteration):
    return hashlib.sha256(
        json.dumps(
            [kind, run_id, particle_id, iteration],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def episode_knowledge(store, run_id, particle_id, iteration, best=None):
    events = store.list_stage_events(run_id, particle_id, iteration)
    record = {
        "particle_id": particle_id,
        "iteration_id": iteration,
        "hypothesis_reference": None,
        "reflection_reference": None,
        "hypothesis_outcome": {"status": "NOT_TESTED"},
        "failure_summary": [],
        "mechanism_is_proven": False,
    }
    for stored in events:
        event = stored.event
        payload = event.model_dump(mode="json")["payload"]
        if event.event_type == "completed":
            if event.stage.value == "HYPOTHESIZING":
                output = payload.get("output", {})
                record["hypothesis"] = shared_text(output.get("hypothesis", ""))
                record["mechanism"] = shared_text(
                    output.get("mechanism", "unspecified")
                )
                record["predicted_direction"] = output.get("predicted_direction")
                record["evidence_references"] = output.get("evidence_references", [])[
                    :5
                ]
                record["hypothesis_reference"] = _reference(
                    "hypothesis", run_id, particle_id, iteration
                )
            elif event.stage.value == "REFLECTING":
                record["reflection"] = {
                    key: (
                        [shared_text(item) for item in value[:5]]
                        if isinstance(value, list)
                        else shared_text(value)
                    )
                    for key, value in payload.get("output", {}).items()
                    if key != "authorization_id"
                }
                record["reflection_reference"] = _reference(
                    "reflection", run_id, particle_id, iteration
                )
            elif event.stage.value == "EVALUATING":
                evaluation = payload.get("evaluation", {})
                record["reward"] = evaluation.get("fitness")
                record["hypothesis_outcome"] = evaluation.get("provenance", {}).get(
                    "hypothesis_outcome", {"status": "INCONCLUSIVE"}
                )
        elif event.event_type in {"failed", "invalid"}:
            record["failure_summary"].append(
                {"stage": event.stage.value, "status": event.event_type}
            )
    record["failure_summary"] = record["failure_summary"][-3:]
    if record["reflection_reference"] is None:
        record["reflection"] = {
            "origin": "controller",
            "message": "No completed reflection; retain failure evidence and do not infer a scientific outcome.",
        }
    if best is not None:
        record["candidate_hash"] = best.candidate_hash
        record["reward"] = best.fitness
    return record


def social_packet(store, snapshot, particle_id, topology):
    particles = {p.particle_id: p for p in snapshot.particles}
    neighbors = topology.neighbors(particle_id)
    bests = [p for p in snapshot.particles if p.pbest is not None]

    # FLAME uses feasibility first and then scalar reward. Stable ID breaks ties.
    def key(p):
        return (
            -int(p.pbest.evaluation.feasible),
            -p.pbest.fitness,
            p.pbest.candidate_hash,
            p.particle_id,
        )

    local = sorted(
        (particles[i] for i in (particle_id, *neighbors) if particles[i].pbest), key=key
    )
    global_values = sorted(bests, key=key)

    def knowledge(p):
        return episode_knowledge(
            store, snapshot.run_id, p.particle_id, p.pbest.iteration_id, p.pbest
        )

    previous = snapshot.iteration_id - 1
    trace = getattr(snapshot, "update_traces", {}).get(particle_id)
    behavior = (
        trace.model_dump(mode="json").get("behavior_update")
        if trace is not None
        else None
    )
    influences = (
        velocity_influence(behavior)
        if isinstance(behavior, Mapping) and "inertia_component" in behavior
        else None
    )
    return {
        "schema_version": "social-knowledge:v2",
        "source_iteration": snapshot.iteration_id,
        "previous_controller_update": behavior,
        "preclamp_component_magnitude_shares": influences,
        "neighbor_ids": list(neighbors),
        "distances": dict(topology.distances[particle_id]),
        "own_previous": (
            episode_knowledge(store, snapshot.run_id, particle_id, previous)
            if previous >= 0
            else None
        ),
        "local_best": knowledge(local[0]) if local else None,
        "global_best": knowledge(global_values[0]) if global_values else None,
        "neighbor_recent": (
            [episode_knowledge(store, snapshot.run_id, i, previous) for i in neighbors]
            if previous >= 0
            else []
        ),
        "interpretation": "Transfer model-conditional predictions and hypotheses; inspect your own parent for all AtomId/BondId values.",
    }


def velocity_influence(components):
    """Pre-clamp L1 magnitude shares; these are not causal attribution."""

    def norm(value):
        if isinstance(value, (list, tuple)):
            return sum(norm(v) for v in value)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("component must contain finite numeric values")
        return abs(value)

    norms = {
        key: norm(components[key])
        for key in (
            "inertia_component",
            "personal_component",
            "local_component",
            "global_component",
        )
    }
    total = sum(norms.values())
    return {key: value / total if total else 0.0 for key, value in norms.items()}
