"""Strict symmetric-positive-definite linear algebra.

There is deliberately no algorithmic covariance clipping here.  A negative
eigenvalue is repaired only when it is within an explicit floating-point
roundoff threshold; every such repair is returned to the caller for logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

FloatArray = NDArray[np.float64]


class SPDValidationError(ValueError):
    """Raised when a matrix is genuinely not positive definite."""


@dataclass(frozen=True)
class RoundoffRepair:
    minimum_eigenvalue: float
    tolerance: float
    replacement: float


def symmetrize(matrix: FloatArray) -> FloatArray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.asarray((matrix + matrix.T) * 0.5, dtype=np.float64)


def _eigh(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    return np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)


def ensure_spd(
    matrix: FloatArray,
    *,
    roundoff_factor: float = 100.0,
) -> tuple[FloatArray, RoundoffRepair | None]:
    """Validate SPD, allowing only a documented roundoff-scale repair."""

    matrix = symmetrize(matrix)
    values, vectors = _eigh(matrix)
    scale = max(float(np.max(np.abs(values))), 1.0)
    tolerance = roundoff_factor * np.finfo(np.float64).eps * scale
    minimum = float(values[0])
    if minimum > 0.0:
        return matrix, None
    if minimum < -tolerance:
        raise SPDValidationError(
            f"covariance lost positive definiteness: lambda_min={minimum:.6e}, "
            f"roundoff_tolerance={tolerance:.6e}"
        )
    replacement = tolerance
    repaired_values = np.maximum(values, replacement)
    repaired = symmetrize((vectors * repaired_values) @ vectors.T)
    return repaired, RoundoffRepair(minimum, tolerance, replacement)


def spectral_function(matrix: FloatArray, function: Callable[[FloatArray], FloatArray]) -> FloatArray:
    values, vectors = _eigh(matrix)
    return symmetrize((vectors * function(values)) @ vectors.T)


def spd_sqrt(matrix: FloatArray) -> FloatArray:
    matrix, _ = ensure_spd(matrix)
    return spectral_function(matrix, np.sqrt)


def spd_inv_sqrt(matrix: FloatArray) -> FloatArray:
    matrix, _ = ensure_spd(matrix)
    return spectral_function(matrix, lambda x: 1.0 / np.sqrt(x))


def spd_log(matrix: FloatArray) -> FloatArray:
    matrix, _ = ensure_spd(matrix)
    return spectral_function(matrix, np.log)


def spd_exp(matrix: FloatArray) -> FloatArray:
    return spectral_function(symmetrize(matrix), np.exp)


def spd_solve(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    matrix, _ = ensure_spd(matrix)
    factor = cho_factor(matrix, lower=True, check_finite=True)
    return np.asarray(cho_solve(factor, rhs, check_finite=True), dtype=np.float64)


def spd_inverse(matrix: FloatArray) -> FloatArray:
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    return symmetrize(spd_solve(matrix, identity))


def logdet_spd(matrix: FloatArray) -> float:
    matrix, _ = ensure_spd(matrix)
    factor = np.linalg.cholesky(matrix)
    return float(2.0 * np.log(np.diag(factor)).sum())


def jko_entropy_eigenvalue_map(values: FloatArray, step_size: float) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < 0.0):
        raise SPDValidationError("the FB--GVI forward covariance is not positive semidefinite")
    return 0.5 * (values + 2.0 * step_size + np.sqrt(values * (values + 4.0 * step_size)))

