"""Deterministic reward mapping for absorption oscillator-strength spectra."""

from __future__ import annotations

from multi_agent_pso.core import ConstraintResult, Evaluation, EvaluationStatus
from multi_agent_pso.protocols import CandidateRef, EvaluationContext

from .models import ExcitedState, SpectrumResult


SIGNIFICANT_OSCILLATOR_STRENGTH = 0.05
RED_BAND_NM = (620.0, 750.0)
EVALUATOR_VERSION = "red-absorption-evaluator:v2"


class RedAbsorptionEvaluator:
    """Map an externally calculated absorption spectrum to one authoritative reward."""

    async def evaluate(
        self, candidate: CandidateRef, context: EvaluationContext
    ) -> Evaluation:
        try:
            metadata = candidate.to_json()["metadata"]
            spectrum = SpectrumResult.model_validate(metadata["spectrum_result"])
        except (KeyError, TypeError, ValueError) as error:
            return Evaluation(
                status=EvaluationStatus.FAILED,
                feasible=False,
                provenance={
                    "error": f"invalid spectrum result: {type(error).__name__}"
                },
            )
        return self.evaluate_spectrum(spectrum)

    def evaluate_spectrum(self, spectrum: SpectrumResult) -> Evaluation:
        provenance = {
            "spectrum": spectrum.model_dump(mode="json"),
            "spectrum_hash": spectrum.spectrum_hash,
            "protocol": spectrum.provenance.protocol.model_dump(mode="json"),
            "protocol_hash": spectrum.provenance.protocol.protocol_hash,
            "geometry_hash": spectrum.provenance.geometry_hash,
            "source_geometry_hash": (
                spectrum.provenance.source_geometry_hash
                or spectrum.provenance.geometry_hash
            ),
            "evaluation_geometry_hash": (
                spectrum.provenance.evaluation_geometry_hash
                or spectrum.provenance.geometry_hash
            ),
        }
        selected = self._selected_state(spectrum)
        metrics = self._metrics(selected)

        if spectrum.status == "FAILED":
            metrics["spectrum_error"] = spectrum.error.model_dump(mode="json")
            return Evaluation(
                status=EvaluationStatus.FAILED,
                feasible=False,
                metrics=metrics,
                provenance=provenance,
            )

        significant_violation = 0.0
        if selected is None:
            converged_strengths = [
                state.oscillator_strength
                for state in spectrum.states
                if state.converged
            ]
            significant_violation = max(
                0.0,
                SIGNIFICANT_OSCILLATOR_STRENGTH - max(converged_strengths, default=0.0),
            )
        lower, upper = RED_BAND_NM
        in_band = selected is not None and lower <= selected.wavelength_nm <= upper
        band_violation = 0.0
        if selected is not None and not in_band:
            band_violation = min(
                abs(selected.wavelength_nm - lower),
                abs(selected.wavelength_nm - upper),
            )
        constraints = (
            ConstraintResult(
                name="significant_absorption_state",
                satisfied=selected is not None,
                violation=significant_violation,
            ),
            ConstraintResult(
                name="red_absorption_band_nm",
                satisfied=in_band,
                violation=band_violation,
            ),
        )

        if selected is None:
            fitness = -2.0
        elif in_band:
            fitness = 1.0 + selected.oscillator_strength
        else:
            fitness = -band_violation / 130.0 + 0.01 * min(
                selected.oscillator_strength, 1.0
            )
        return Evaluation(
            status=EvaluationStatus.SUCCESS,
            feasible=in_band,
            metrics=metrics,
            constraints=constraints,
            fitness=fitness,
            provenance=provenance,
        )

    @staticmethod
    def _selected_state(spectrum: SpectrumResult) -> ExcitedState | None:
        if spectrum.status != "SUCCESS":
            return None
        ordered = sorted(
            (state for state in spectrum.states if state.converged),
            key=lambda state: (state.energy_ev, state.state_index),
        )
        return next(
            (
                state
                for state in ordered
                if state.oscillator_strength >= SIGNIFICANT_OSCILLATOR_STRENGTH
            ),
            None,
        )

    @staticmethod
    def _metrics(selected: ExcitedState | None) -> dict[str, object]:
        raw = selected.model_dump(mode="json") if selected is not None else None
        return {
            "observable": "absorption_oscillator_strength_proxy",
            "units": {
                "energy": "eV",
                "wavelength": "nm",
                "oscillator_strength": "dimensionless",
            },
            "oscillator_strength_threshold": SIGNIFICANT_OSCILLATOR_STRENGTH,
            "target_wavelength_band_nm": list(RED_BAND_NM),
            "selected_state": raw,
            "selected_state_index": (
                selected.state_index if selected is not None else None
            ),
            "selected_energy_ev": selected.energy_ev if selected is not None else None,
            "selected_wavelength_nm": (
                selected.wavelength_nm if selected is not None else None
            ),
            "selected_oscillator_strength": (
                selected.oscillator_strength if selected is not None else None
            ),
            "selected_converged": selected.converged if selected is not None else None,
            "selected_root_character": (
                selected.root_character if selected is not None else None
            ),
        }


__all__ = [
    "EVALUATOR_VERSION",
    "RED_BAND_NM",
    "SIGNIFICANT_OSCILLATOR_STRENGTH",
    "RedAbsorptionEvaluator",
]
