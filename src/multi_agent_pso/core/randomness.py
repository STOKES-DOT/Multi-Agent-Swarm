"""Deterministic random-seed derivation for PSO update components."""

import hashlib


_SEPARATOR = "\x1f"


def derive_seed(run_seed: int, particle_id: str, iteration: int, purpose: str) -> int:
    """Derive a stable 64-bit seed from a uniquely delimited update identity."""
    _validate_nonnegative_int(run_seed, name="run_seed")
    _validate_text(particle_id, name="particle_id")
    _validate_nonnegative_int(iteration, name="iteration")
    _validate_text(purpose, name="purpose")
    payload = f"multi-agent-pso:v1\x1f{run_seed}\x1f{particle_id}\x1f{iteration}\x1f{purpose}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _validate_nonnegative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_text(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if _SEPARATOR in value:
        raise ValueError(f"{name} must not contain U+001F")
