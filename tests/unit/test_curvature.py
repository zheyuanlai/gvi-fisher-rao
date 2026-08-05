from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.diagnostics.curvature import curvature_constants, whitened_initialization
from fr_gvi.experiments.reference import solve_reference
from fr_gvi.targets.core import GaussianTarget, ShiftedLogCoshTarget


def test_gaussian_whitened_curvature_is_the_identity() -> None:
    rng = np.random.default_rng(5)
    dimension = 6
    root = rng.standard_normal((dimension, dimension))
    precision = root @ root.T + dimension * np.eye(dimension)
    target = GaussianTarget(rng.standard_normal(dimension), precision)
    optimum = GaussianState(target.mean, target.covariance)
    constants = curvature_constants(target, optimum)
    assert constants.alpha_star == pytest.approx(1.0)
    assert constants.beta_star == pytest.approx(1.0)
    assert constants.kappa_star == pytest.approx(1.0)


def _logcosh(dimension: int, transform: np.ndarray, shift: np.ndarray) -> ShiftedLogCoshTarget:
    return ShiftedLogCoshTarget(
        np.geomspace(0.1, 1.0, dimension),
        1.3,
        np.linspace(-0.5, 0.5, dimension),
        transform,
        shift,
    )


def test_logcosh_whitened_constants_match_brute_force() -> None:
    """alpha_star and beta_star must equal the sampled whitened Hessian extremes."""

    dimension = 4
    target = _logcosh(dimension, np.eye(dimension), np.zeros(dimension))
    reference = solve_reference(target, GaussianState(np.zeros(dimension), np.eye(dimension)),
                                points=512, seed=3)
    constants = curvature_constants(target, reference.state)

    root = np.linalg.cholesky(reference.state.covariance)
    rng = np.random.default_rng(0)
    lowest, highest = np.inf, -np.inf
    for _ in range(4000):
        point = 6.0 * rng.standard_normal(dimension)
        whitened = root.T @ np.asarray(target.hessian(point)) @ root
        values = np.linalg.eigvalsh(whitened)
        lowest = min(lowest, float(values[0]))
        highest = max(highest, float(values[-1]))

    assert constants.alpha_star <= lowest + 1e-9
    assert constants.beta_star >= highest - 1e-9
    assert lowest == pytest.approx(constants.alpha_star, rel=0.05)
    assert highest == pytest.approx(constants.beta_star, rel=0.05)


def test_whitened_constants_are_affine_invariant() -> None:
    """kappa_star must not change under an invertible change of variables."""

    dimension = 4
    rng = np.random.default_rng(21)
    base = _logcosh(dimension, np.eye(dimension), np.zeros(dimension))
    left, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    transform = left * np.geomspace(1.0, 500.0, dimension)
    shift = rng.standard_normal(dimension)
    transformed = _logcosh(dimension, transform, shift)

    initial = GaussianState(np.zeros(dimension), np.eye(dimension))
    base_constants = curvature_constants(
        base, solve_reference(base, initial, points=512, seed=1).state
    )
    transformed_constants = curvature_constants(
        transformed, solve_reference(transformed, initial, points=512, seed=1).state
    )

    assert transformed_constants.alpha_star == pytest.approx(base_constants.alpha_star, rel=1e-9)
    assert transformed_constants.beta_star == pytest.approx(base_constants.beta_star, rel=1e-9)
    assert transformed_constants.kappa_star == pytest.approx(base_constants.kappa_star, rel=1e-9)
    # The original-coordinate condition number is not invariant, which is the
    # whole point of working with kappa_star.
    assert transformed_constants.condition > 100.0 * base_constants.condition


def test_whitened_initialization_bounds() -> None:
    dimension = 3
    target = GaussianTarget(np.zeros(dimension), np.diag([1.0, 4.0, 9.0]))
    optimum = GaussianState(target.mean, target.covariance)
    initial = GaussianState(np.zeros(dimension), 0.5 * np.eye(dimension))
    whitened = whitened_initialization(initial, optimum, 1.0)
    # C_star = diag(1, 1/4, 1/9), so C_star^{-1/2} (0.5 I) C_star^{-1/2}
    # has eigenvalues 0.5, 2, 4.5.
    assert whitened.lambda_0_star == pytest.approx(0.5)
    assert whitened.lambda_0_star_max == pytest.approx(4.5)
    assert whitened.lambda_max_star == pytest.approx(4.5)
