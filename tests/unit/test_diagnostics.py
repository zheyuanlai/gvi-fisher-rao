from __future__ import annotations

import numpy as np

from fr_gvi.algorithms import GaussianState
from fr_gvi.diagnostics import gaussian_kl_gap, gaussian_w2_squared
from fr_gvi.targets import GaussianTarget


def test_exact_gaussian_kl_gap() -> None:
    target = GaussianTarget(np.zeros(2), np.diag([2.0, 0.5]))
    state = GaussianState(np.asarray([1.0, -2.0]), np.diag([0.7, 3.0]))
    product = target.precision @ state.covariance
    expected = 0.5 * (
        state.mean @ target.precision @ state.mean
        + np.trace(product)
        - 2
        - np.linalg.slogdet(product)[1]
    )
    assert abs(gaussian_kl_gap(target, state) - expected) < 1e-14
    assert gaussian_kl_gap(target, GaussianState(target.mean, target.covariance)) < 1e-14


def test_gaussian_wasserstein_commuting_case() -> None:
    first = GaussianState(np.asarray([1.0, 2.0]), np.diag([1.0, 4.0]))
    second = GaussianState(np.asarray([-1.0, 1.0]), np.diag([9.0, 1.0]))
    expected = np.sum((first.mean - second.mean) ** 2) + (1.0 - 3.0) ** 2 + (2.0 - 1.0) ** 2
    assert abs(gaussian_w2_squared(first, second) - expected) < 1e-12
    assert gaussian_w2_squared(first, first) < 1e-12

