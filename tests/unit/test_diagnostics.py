from __future__ import annotations

import pytest
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



def test_gaussian_kl_gap_is_stable_on_the_diao_spectrum() -> None:
    """The exact gap must vanish at the optimum even at condition number 1e9."""

    import numpy as np

    from fr_gvi.algorithms.core import GaussianState
    from fr_gvi.diagnostics.core import gaussian_kl_gap
    from fr_gvi.targets.core import GaussianTarget

    rng = np.random.default_rng(0)
    dimension = 10
    rotation, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    precision = (rotation * np.geomspace(1e-9, 1.0, dimension)) @ rotation.T
    target = GaussianTarget(rng.random(dimension), precision)

    at_optimum = gaussian_kl_gap(target, GaussianState(target.mean, target.covariance))
    assert abs(at_optimum) < 1e-14

    for scale in (1.3, 2.0, 0.5):
        gap = gaussian_kl_gap(
            target, GaussianState(target.mean, target.covariance * scale)
        )
        analytic = 0.5 * dimension * (scale - 1.0 - np.log(scale))
        # The tolerance is set by the accuracy of inverting a precision matrix
        # of condition number 1e9, not by the gap formula itself.
        assert gap == pytest.approx(analytic, rel=1e-7)
