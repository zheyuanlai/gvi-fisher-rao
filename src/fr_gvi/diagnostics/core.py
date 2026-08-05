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
    """Exact KL(N(m,C) || N(mu, H^{-1})) for a Gaussian target.

    ``log det(H C)`` is split into ``log det H + log det C`` and each factor is
    taken through its Cholesky factorization.  Forming ``H C`` and calling
    ``slogdet`` on the product loses positivity once both factors are severely
    ill-conditioned, which happens on the Diao benchmark where each has
    condition number ``1e9``.
    """

    lower = np.linalg.cholesky(target.precision)
    delta = lower.T @ (state.mean - target.mean)
    congruent = lower.T @ state.covariance @ lower
    eigenvalues = np.linalg.eigvalsh((congruent + congruent.T) / 2.0)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("L^T C L must be positive definite")
    shifted = eigenvalues - 1.0
    return 0.5 * float(delta @ delta + np.sum(shifted - np.log1p(shifted)))


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


def gaussian_core_rate(state: GaussianState, optimum: GaussianState) -> float:
    """The exact Gaussian-core dissipation rate ``q_G`` of Lemma 2.24.

    In optimizer-whitened coordinates ``a_star = (0, I)`` and

        <xi, F_G(a)>_star = -(1/2) q_G(a) D(a),
        q_G(a) = [2 m^T C m + Tr(C (C - I)^2)] / [|m|^2 + (1/2)|C - I|_F^2],

    so ``q_G`` is the instantaneous decay rate of ``|a - a_star|_star^2`` along the
    continuous-time flow.  The bound ``q_G >= 2 lambda_min(C)`` of the same lemma
    is what Corollary 2.26 turns into the uniform sublevel rate.
    """

    inverse_root = spd_inv_sqrt(optimum.covariance)
    mean = inverse_root @ (state.mean - optimum.mean)
    covariance = inverse_root @ state.covariance @ inverse_root
    covariance = (covariance + covariance.T) / 2.0
    deviation = covariance - np.eye(mean.size)
    denominator = float(mean @ mean) + 0.5 * float(np.linalg.norm(deviation, ord="fro") ** 2)
    if denominator <= 0.0:
        return float("nan")
    numerator = 2.0 * float(mean @ covariance @ mean) + float(
        np.trace(covariance @ deviation @ deviation)
    )
    return numerator / denominator


def gaussian_sharp_threshold(rate: float) -> float:
    """The sharp Gaussian energy sublevel of Corollary 2.26, ``Delta_G^sharp(rho)``."""

    half = rate / 2.0
    return 0.5 * (half - 1.0 - float(np.log(half)))


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

