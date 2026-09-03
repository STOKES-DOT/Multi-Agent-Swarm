"""Domain-independent position and velocity spaces for PSO updates."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Generic, Iterable, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

P = TypeVar("P")
V = TypeVar("V")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Projection(Generic[P]):
    """A projected position and the dimensions changed by that projection."""

    position: P
    changed_dimensions: tuple[int, ...]


class PositionSpace(Protocol[P, V]):
    """Operations required by a domain-specific PSO position representation."""

    def sample_position(self, rng: np.random.Generator) -> P: ...

    def zero_velocity(self) -> V: ...

    def difference(self, target: P, origin: P) -> V: ...

    def scale_velocity(self, velocity: V, scalar: float) -> V: ...

    def random_scale(
        self, velocity: V, upper: float, rng: np.random.Generator
    ) -> V: ...

    def add_velocities(self, parts: Iterable[V]) -> V: ...

    def clamp_velocity(self, velocity: V, fraction: float) -> V: ...

    def advance(self, position: P, velocity: V) -> P: ...

    def project(self, position: P) -> Projection[P]: ...

    def distance(self, left: P, right: P) -> float: ...

    def serialize_position(self, position: P) -> object: ...

    def deserialize_position(self, value: object) -> P: ...

    def serialize_velocity(self, velocity: V) -> object: ...

    def deserialize_velocity(self, value: object) -> V: ...


class ContinuousBoxPositionSpace(PositionSpace[FloatArray, FloatArray]):
    """A continuous float64 position space bounded by an axis-aligned box."""

    def __init__(self, lower: object, upper: object) -> None:
        lower_array = self._coerce_array(lower, name="lower")
        upper_array = self._coerce_array(upper, name="upper")
        if lower_array.shape != upper_array.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if lower_array.size == 0:
            raise ValueError("bounds must not be empty")
        if np.any(lower_array >= upper_array):
            raise ValueError("each lower bound must be strictly less than its upper bound")
        self._lower = self._readonly_copy(lower_array)
        self._upper = self._readonly_copy(upper_array)

    @property
    def lower(self) -> FloatArray:
        """A fresh, immutable copy of the lower box bounds."""
        return self._readonly_copy(self._lower)

    @property
    def upper(self) -> FloatArray:
        """A fresh, immutable copy of the upper box bounds."""
        return self._readonly_copy(self._upper)

    def sample_position(self, rng: np.random.Generator) -> FloatArray:
        factors = rng.random(self._dimension)
        sample = (1.0 - factors) * self._lower + factors * self._upper
        sample = np.maximum(sample, self._lower)
        upper_inside = np.nextafter(self._upper, self._lower)
        sample = np.where(sample >= self._upper, upper_inside, sample)
        return self._result(sample)

    def zero_velocity(self) -> FloatArray:
        return self._readonly_copy(np.zeros(self._dimension, dtype=np.float64))

    def difference(self, target: FloatArray, origin: FloatArray) -> FloatArray:
        return self._result(self._array(target, name="target") - self._array(origin, name="origin"))

    def scale_velocity(self, velocity: FloatArray, scalar: float) -> FloatArray:
        return self._result(self._array(velocity, name="velocity") * self._scalar(scalar, name="scalar"))

    def random_scale(
        self, velocity: FloatArray, upper: float, rng: np.random.Generator
    ) -> FloatArray:
        upper_value = self._scalar(upper, name="upper")
        if upper_value < 0.0:
            raise ValueError("upper must be nonnegative")
        velocity_values = self._array(velocity, name="velocity")
        factors = rng.uniform(0.0, upper_value, size=self._dimension)
        return self._result(velocity_values * factors)

    def add_velocities(self, parts: Iterable[FloatArray]) -> FloatArray:
        total = np.zeros(self._dimension, dtype=np.float64)
        for part in parts:
            total += self._array(part, name="velocity")
        return self._result(total)

    def clamp_velocity(self, velocity: FloatArray, fraction: float) -> FloatArray:
        fraction_value = self._scalar(fraction, name="fraction")
        if not 0.0 < fraction_value <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        with np.errstate(over="ignore", invalid="ignore"):
            raw_width = self._upper - self._lower
            finite_width = np.isfinite(raw_width)
            limit = np.empty_like(raw_width)
            limit[finite_width] = fraction_value * raw_width[finite_width]
            overflow_width = ~finite_width
            if np.any(overflow_width):
                limit[overflow_width] = (
                    fraction_value * self._upper[overflow_width]
                    - fraction_value * self._lower[overflow_width]
                )
        return self._result(np.clip(self._array(velocity, name="velocity"), -limit, limit))

    def advance(self, position: FloatArray, velocity: FloatArray) -> FloatArray:
        return self._result(self._array(position, name="position") + self._array(velocity, name="velocity"))

    def project(self, position: FloatArray) -> Projection[FloatArray]:
        source = self._array(position, name="position")
        projected = np.clip(source, self._lower, self._upper)
        changed = tuple(int(index) for index in np.flatnonzero(projected != source))
        return Projection(position=self._result(projected), changed_dimensions=changed)

    def distance(self, left: FloatArray, right: FloatArray) -> float:
        left_values = self._array(left, name="left")
        right_values = self._array(right, name="right")
        distance = math.dist(left_values.tolist(), right_values.tolist())
        if not math.isfinite(distance):
            raise ValueError("distance must be finite")
        return distance

    def serialize_position(self, position: FloatArray) -> object:
        return self._array(position, name="position").tolist()

    def deserialize_position(self, value: object) -> FloatArray:
        if not isinstance(value, list):
            raise TypeError("position serialization must be a Python list")
        return self._array(value, name="position")

    def serialize_velocity(self, velocity: FloatArray) -> object:
        return self._array(velocity, name="velocity").tolist()

    def deserialize_velocity(self, value: object) -> FloatArray:
        if not isinstance(value, list):
            raise TypeError("velocity serialization must be a Python list")
        return self._array(value, name="velocity")

    @property
    def _dimension(self) -> int:
        return int(self._lower.size)

    def _array(self, value: object, *, name: str) -> FloatArray:
        array = self._coerce_array(value, name=name)
        if array.shape != self._lower.shape:
            raise ValueError(f"{name} must have shape {self._lower.shape}")
        return self._readonly_copy(array)

    @staticmethod
    def _coerce_array(value: object, *, name: str) -> FloatArray:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a one-dimensional numeric array") from error
        if raw.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        try:
            array = np.array(raw, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain real numeric values") from error
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _readonly_copy(array: FloatArray) -> FloatArray:
        result = np.array(array, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def _result(self, value: FloatArray) -> FloatArray:
        if not np.all(np.isfinite(value)):
            raise ValueError("operation produced non-finite values")
        return self._readonly_copy(value)

    @staticmethod
    def _scalar(value: object, *, name: str) -> float:
        if isinstance(value, np.ndarray) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite real scalar")
        if not isinstance(value, (Real, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite real scalar") from error
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be a finite real scalar")
        return scalar
