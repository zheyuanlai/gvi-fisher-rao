"""The six algorithms admitted by the scientific protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from fr_gvi.expectations.core import ExpectationEngine
from fr_gvi.linear_algebra.spd import (
    SPDValidationError,
    ensure_spd,
    jko_entropy_eigenvalue_map,
    spd_exp,
    spd_inverse,
    spd_solve,
    spd_sqrt,
    symmetrize,
)
from fr_gvi.targets.core import Target, mean_hessian
from fr_gvi.utils.accounting import OperationCounts

FloatArray = NDArray[np.float64]


class AlgorithmFailure(RuntimeError):
    """Expected wrapper for a genuine algorithmic numerical failure."""


class Method(StrEnum):
    FR_R = "FR--R"
    FR_KL = "FR--KL"
    FR_R_STL = "FR--R--STL"
    FR_KL_STL = "FR--KL--STL"
    FB_GVI = "FB--GVI"
    S_FB_GVI = "S--FB--GVI"
    LAPLACE = "Laplace"

    @property
    def stochastic(self) -> bool:
        return self in {self.FR_R_STL, self.FR_KL_STL, self.S_FB_GVI}


@dataclass(frozen=True)
class GaussianState:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance, _ = ensure_spd(np.asarray(self.covariance, dtype=np.float64))
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance shape does not match mean")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class StepDiagnostics:
    repair: dict[str, float] | None
    gradient: FloatArray
    hessian: FloatArray


def _validated_state(
    mean: FloatArray,
    covariance: FloatArray,
    counts: OperationCounts,
) -> tuple[GaussianState, dict[str, float] | None]:
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise AlgorithmFailure("NaN or Inf encountered in iterate")
    try:
        covariance, repair = ensure_spd(covariance)
    except SPDValidationError as exc:
        raise AlgorithmFailure(str(exc)) from exc
    repair_dict: dict[str, float] | None = None
    if repair is not None:
        counts.roundoff_repairs += 1
        repair_dict = {
            "minimum_eigenvalue": repair.minimum_eigenvalue,
            "tolerance": repair.tolerance,
            "replacement": repair.replacement,
        }
    return GaussianState(mean, covariance), repair_dict


def quadratic_rescue(target: Target, input_mean: FloatArray) -> GaussianState:
    gradient = np.asarray(target.grad(input_mean), dtype=np.float64)
    hessian = np.asarray(target.hessian(input_mean), dtype=np.float64)
    mean = input_mean - spd_solve(hessian, gradient)
    covariance = spd_inverse(hessian)
    return GaussianState(mean, covariance)


def _population(
    target: Target,
    state: GaussianState,
    engine: ExpectationEngine,
    counts: OperationCounts,
) -> tuple[FloatArray, FloatArray]:
    result = engine.evaluate(target, state.mean, state.covariance)
    counts.expectation_evaluations += 1
    counts.gradient_evaluations += 1
    counts.hessian_evaluations += 1
    counts.oracle_pairs += 1
    return np.asarray(result.grad, dtype=np.float64), np.asarray(result.hessian, dtype=np.float64)


def _sampled(
    target: Target,
    state: GaussianState,
    rng: np.random.Generator,
    batch_size: int,
    *,
    stl: bool,
    counts: OperationCounts,
) -> tuple[FloatArray, FloatArray]:
    root = spd_sqrt(state.covariance)
    counts.matrix_square_roots += 1
    counts.eigendecompositions += 1
    normals = rng.standard_normal((batch_size, state.mean.size), dtype=np.float64)
    samples = state.mean + normals @ root.T
    gradients = np.asarray(target.grad(samples), dtype=np.float64)
    averaged_hessian = mean_hessian(target, samples)
    counts.gradient_evaluations += batch_size
    counts.hessian_evaluations += batch_size
    counts.oracle_pairs += batch_size
    if stl:
        centered = samples - state.mean
        score = spd_solve(state.covariance, centered.T).T
        counts.cholesky_factorizations += 1
        counts.cholesky_solves += 1
        mean_direction = np.mean(-gradients + score, axis=0)
    else:
        mean_direction = np.mean(gradients, axis=0)
    return np.asarray(mean_direction), averaged_hessian


def step(
    method: Method,
    target: Target,
    state: GaussianState,
    step_size: float,
    *,
    engine: ExpectationEngine | None,
    rng: np.random.Generator,
    batch_size: int,
    counts: OperationCounts,
    raw_mean_ablation: bool = False,
) -> tuple[GaussianState, StepDiagnostics]:
    if step_size <= 0.0:
        raise ValueError("step size must be positive")
    covariance = state.covariance
    identity = np.eye(state.mean.size, dtype=np.float64)

    if method in {Method.FR_R, Method.FR_KL, Method.FB_GVI}:
        if engine is None:
            raise ValueError("deterministic method requires an expectation engine")
        expected_gradient, hessian = _population(target, state, engine, counts)
        if method in {Method.FR_R, Method.FR_KL}:
            mean_direction = -expected_gradient
        else:
            mean_direction = expected_gradient
    elif method in {Method.FR_R_STL, Method.FR_KL_STL}:
        mean_direction, hessian = _sampled(
            target,
            state,
            rng,
            batch_size,
            stl=not raw_mean_ablation,
            counts=counts,
        )
        if raw_mean_ablation:
            mean_direction = -mean_direction
        expected_gradient = -mean_direction
    elif method == Method.S_FB_GVI:
        mean_direction, hessian = _sampled(
            target, state, rng, batch_size, stl=False, counts=counts
        )
        expected_gradient = mean_direction
    else:
        raise ValueError(f"{method} does not have an iterative step")

    try:
        if method in {Method.FR_R, Method.FR_R_STL}:
            mean_new = state.mean + step_size * covariance @ mean_direction
            root = spd_sqrt(covariance)
            tangent = symmetrize(identity - root @ hessian @ root)
            covariance_new = symmetrize(root @ spd_exp(step_size * tangent) @ root)
            counts.matrix_square_roots += 1
            counts.matrix_exponentials += 1
            counts.eigendecompositions += 2
        elif method in {Method.FR_KL, Method.FR_KL_STL}:
            mean_new = state.mean + step_size * covariance @ mean_direction
            precision = spd_inverse(covariance)
            covariance_new = (1.0 + step_size) * spd_solve(
                precision + step_size * hessian, identity
            )
            covariance_new = symmetrize(covariance_new)
            counts.cholesky_factorizations += 2
            counts.cholesky_solves += 2
        else:
            mean_new = state.mean - step_size * mean_direction
            forward_map = identity - step_size * hessian
            half_covariance = symmetrize(forward_map @ covariance @ forward_map)
            values, vectors = np.linalg.eigh(half_covariance)
            scale = max(float(np.max(np.abs(values))), 1.0)
            tolerance = 100.0 * np.finfo(np.float64).eps * scale
            if float(values[0]) < -tolerance:
                raise SPDValidationError(
                    f"FB--GVI forward covariance invalid: lambda_min={values[0]:.6e}"
                )
            values = np.maximum(values, 0.0)
            updated_values = jko_entropy_eigenvalue_map(values, step_size)
            covariance_new = symmetrize((vectors * updated_values) @ vectors.T)
            counts.eigendecompositions += 1
            counts.matrix_square_roots += 1
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, SPDValidationError) as exc:
        raise AlgorithmFailure(f"{method} update failed: {exc}") from exc

    counts.iterations += 1
    next_state, repair = _validated_state(mean_new, covariance_new, counts)
    return next_state, StepDiagnostics(repair, expected_gradient, hessian)

