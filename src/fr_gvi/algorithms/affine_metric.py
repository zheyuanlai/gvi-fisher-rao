"""General affine-invariant Gaussian metrics of manuscript Section 5.

The classification theorem states that every affine-invariant Riemannian metric
on ``R^N x SPD(N)`` has the form

    g_a((u,X),(v,Y)) = eta u^T C^{-1} v
                       + omega Tr(C^{-1} X C^{-1} Y)
                       + tau Tr(C^{-1} X) Tr(C^{-1} Y),

with ``eta > 0``, ``omega > 0`` and ``tau > -omega / N``.  The Fisher--Rao metric
is the member ``(eta, omega, tau) = (1, 1/2, 0)``.

This module implements the corresponding gradient flow and its Riemannian
retraction discretization exactly as written in the manuscript, and the two
predicted modal decay rates for the standard Gaussian target.  It is a
Section 5 verification tool and is deliberately kept out of the six-algorithm
main comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.core import AlgorithmFailure, GaussianState
from fr_gvi.expectations.core import ExpectationEngine
from fr_gvi.linear_algebra.spd import (
    SPDValidationError,
    ensure_spd,
    spd_exp,
    spd_sqrt,
    symmetrize,
)
from fr_gvi.targets.core import Target

FloatArray = NDArray[np.float64]

FISHER_RAO_OMEGA = 0.5
FISHER_RAO_TAU = 0.0


@dataclass(frozen=True)
class AffineMetric:
    """A member of the affine-invariant family, parameterized by (omega, tau)."""

    omega: float
    tau: float
    dimension: int

    def __post_init__(self) -> None:
        if self.omega <= 0.0:
            raise ValueError("omega must be positive")
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.tau <= -self.omega / self.dimension:
            raise ValueError(
                f"tau={self.tau} violates positive definiteness: tau > -omega/N = "
                f"{-self.omega / self.dimension}"
            )

    @property
    def is_fisher_rao(self) -> bool:
        return bool(
            np.isclose(self.omega, FISHER_RAO_OMEGA) and np.isclose(self.tau, FISHER_RAO_TAU)
        )

    @property
    def traceless_rate(self) -> float:
        """Predicted linearized decay rate of the traceless covariance mode."""

        return 1.0 / (2.0 * self.omega)

    @property
    def trace_rate(self) -> float:
        """Predicted linearized decay rate of the trace covariance mode."""

        return 1.0 / (2.0 * (self.omega + self.tau * self.dimension))

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "omega": float(self.omega),
            "tau": float(self.tau),
            "dimension": int(self.dimension),
            "predicted_traceless_rate": float(self.traceless_rate),
            "predicted_trace_rate": float(self.trace_rate),
            "is_fisher_rao": bool(self.is_fisher_rao),
        }


def covariance_velocity(
    metric: AffineMetric,
    covariance: FloatArray,
    hessian: FloatArray,
) -> FloatArray:
    """Right-hand side of the covariance ODE of manuscript equation (5.2).

    ``hessian`` is ``H(a) = E[grad grad V]``, i.e. minus the expected Hessian of
    ``log rho_post``.
    """

    covariance = symmetrize(np.asarray(covariance, dtype=np.float64))
    hessian = symmetrize(np.asarray(hessian, dtype=np.float64))
    isotropic_trace = float(np.trace(np.eye(metric.dimension) - covariance @ hessian))
    primary = symmetrize(covariance - covariance @ hessian @ covariance) / (2.0 * metric.omega)
    correction = (
        metric.tau
        / (2.0 * metric.omega * (metric.omega + metric.tau * metric.dimension))
        * isotropic_trace
        * covariance
    )
    return symmetrize(primary - correction)


def retraction_step(
    metric: AffineMetric,
    target: Target,
    state: GaussianState,
    step_size: float,
    *,
    engine: ExpectationEngine,
) -> GaussianState:
    """One step of the Riemannian retraction scheme of manuscript (5.3).

    For ``(omega, tau) = (1/2, 0)`` this reduces exactly to the Fisher--Rao
    retraction scheme ``upd:Riem``.
    """

    if step_size <= 0.0:
        raise ValueError("step size must be positive")
    expectation = engine.evaluate(target, state.mean, state.covariance)
    hessian = symmetrize(np.asarray(expectation.hessian, dtype=np.float64))
    score = -np.asarray(expectation.grad, dtype=np.float64)

    mean_new = state.mean + step_size * state.covariance @ score

    root = spd_sqrt(state.covariance)
    whitened_hessian = symmetrize(root @ hessian @ root)
    isotropic = (
        metric.omega + metric.tau * float(np.trace(state.covariance @ hessian))
    ) / (metric.omega + metric.dimension * metric.tau)
    exponent = (step_size / (2.0 * metric.omega)) * (
        -whitened_hessian + isotropic * np.eye(metric.dimension)
    )
    try:
        covariance_new = symmetrize(root @ spd_exp(exponent) @ root)
        covariance_new, _ = ensure_spd(covariance_new)
    except (SPDValidationError, np.linalg.LinAlgError, ValueError) as exc:
        raise AlgorithmFailure(f"affine-metric retraction failed: {exc}") from exc
    if not np.all(np.isfinite(mean_new)) or not np.all(np.isfinite(covariance_new)):
        raise AlgorithmFailure("NaN or Inf encountered in affine-metric iterate")
    return GaussianState(mean_new, covariance_new)


def modal_decomposition(covariance: FloatArray) -> tuple[float, float]:
    """Split ``X = C - I`` into its traceless Frobenius norm and its trace."""

    covariance = symmetrize(np.asarray(covariance, dtype=np.float64))
    dimension = covariance.shape[0]
    deviation = covariance - np.eye(dimension)
    trace = float(np.trace(deviation))
    traceless = deviation - (trace / dimension) * np.eye(dimension)
    return float(np.linalg.norm(traceless, ord="fro")), trace
