from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.expectations.core import ExpectationEngine, ExpectationResult
from fr_gvi.linear_algebra.spd import logdet_spd, spd_inv_sqrt, spd_solve, spd_sqrt
from fr_gvi.targets.core import GaussianTarget, Target

FloatArray = NDArray[np.float64]


def objective(
    target: Target,
    state: GaussianState,
    engine: ExpectationEngine,
) -> tuple[float, ExpectationResult]:
    expectation = engine.evaluate(target, state.mean, state.covariance)
    return expectation.value - 0.5 * logdet_spd(state.covariance), expectation


def gaussian_kl_gap(target: GaussianTarget, state: GaussianState) -> float:
    delta = state.mean - target.mean
    product = target.precision @ state.covariance
    sign, logdet = np.linalg.slogdet(product)
    if sign <= 0.0:
        raise ValueError("H C must have positive determinant")
    dimension = state.mean.size
    return 0.5 * (
        float(delta @ target.precision @ delta)
        + float(np.trace(product))
        - dimension
        - float(logdet)
    )


def gaussian_w2_squared(first: GaussianState, second: GaussianState) -> float:
    root_second = spd_sqrt(second.covariance)
    middle = spd_sqrt(root_second @ first.covariance @ root_second)
    mean_term = float(np.sum((first.mean - second.mean) ** 2))
    covariance_term = float(np.trace(first.covariance + second.covariance - 2.0 * middle))
    return max(mean_term + covariance_term, 0.0)


@dataclass(frozen=True)
class RelativeDiagnostics:
    mean_norm: float
    covariance_frobenius: float
    covariance_minimum_eigenvalue: float
    covariance_maximum_eigenvalue: float


def optimizer_relative(state: GaussianState, optimum: GaussianState) -> RelativeDiagnostics:
    inverse_root = spd_inv_sqrt(optimum.covariance)
    relative_mean = inverse_root @ (state.mean - optimum.mean)
    relative_covariance = inverse_root @ state.covariance @ inverse_root
    values = np.linalg.eigvalsh(relative_covariance)
    return RelativeDiagnostics(
        float(np.linalg.norm(relative_mean)),
        float(np.linalg.norm(relative_covariance - np.eye(state.mean.size), ord="fro")),
        float(values[0]),
        float(values[-1]),
    )


@dataclass(frozen=True)
class ResidualDiagnostics:
    fisher_rao_squared: float
    bures_wasserstein_squared: float


def residuals(state: GaussianState, expectation: ExpectationResult) -> ResidualDiagnostics:
    g = -expectation.grad
    hessian = expectation.hessian
    root = spd_sqrt(state.covariance)
    fr_covariance = np.eye(state.mean.size) - root @ hessian @ root
    fr = float(g @ state.covariance @ g + 0.5 * np.linalg.norm(fr_covariance, ord="fro") ** 2)
    precision_minus_hessian = spd_solve(
        state.covariance, np.eye(state.mean.size)
    ) - hessian
    bw = float(
        g @ g
        + np.trace(
            precision_minus_hessian
            @ state.covariance
            @ precision_minus_hessian
        )
    )
    return ResidualDiagnostics(fr, bw)

