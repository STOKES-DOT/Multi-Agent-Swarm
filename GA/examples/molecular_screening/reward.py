"""Configurable absorption target with weak PLQY and log-epsilon terms."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpectralObjective:
    lower_nm: float = 620.0
    upper_nm: float = 750.0
    secondary_weight: float = 0.01

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.lower_nm, self.upper_nm, self.secondary_weight)):
            raise ValueError('objective must be finite')
        if not 0 < self.lower_nm < self.upper_nm or not 0 <= self.secondary_weight <= .01:
            raise ValueError('invalid band or dominating secondary weight')

    def evaluate(self, prediction):
        wavelength = prediction['absorption_nm']
        plqy, epsilon = prediction['plqy'], prediction['epsilon_m1_cm1']
        if not all(math.isfinite(x) for x in (wavelength, plqy, epsilon)) or wavelength <= 0:
            raise ValueError('invalid spectral prediction')
        distance = max(self.lower_nm-wavelength, wavelength-self.upper_nm, 0.0)
        q = plqy if 0 <= plqy <= 1 else 0.
        e = min(1., max(0., (math.log10(epsilon)-3.)/3.)) if epsilon > 0 else 0.
        secondary = self.secondary_weight * (q+e)/2
        score = 1.+secondary if distance == 0 else -distance/(self.upper_nm-self.lower_nm)+secondary
        return {'fitness': score, 'feasible': distance == 0, 'target_distance': distance,
                'secondary_contribution': secondary, 'absorption_nm': wavelength,
                'log10_epsilon': math.log10(epsilon) if epsilon > 0 else None}
