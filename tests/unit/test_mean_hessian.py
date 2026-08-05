from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.targets.core import (
    GaussianTarget,
    LogisticRegressionTarget,
    ShiftedLogCoshTarget,
    mean_hessian,
)


def _targets() -> list[object]:
    rng = np.random.default_rng(3)
    dimension = 5
    root = rng.standard_normal((dimension, dimension))
    gaussian = GaussianTarget(
        rng.standard_normal(dimension), root @ root.T + dimension * np.eye(dimension)
    )
    logcosh = ShiftedLogCoshTarget(
        np.linspace(0.4, 1.6, dimension),
        0.7,
        np.linspace(-0.4, 0.4, dimension),
        np.eye(dimension) + 0.2 * rng.standard_normal((dimension, dimension)),
        rng.standard_normal(dimension),
    )
    features = rng.standard_normal((40, dimension))
    labels = rng.integers(0, 2, 40).astype(float)
    logistic = LogisticRegressionTarget(features, labels, 0.5)
    return [gaussian, logcosh, logistic]


@pytest.mark.parametrize("target", _targets())
def test_fast_weighted_hessian_matches_per_sample_average(target: object) -> None:
    """The O(n d^2) fast path must reproduce the generic per-sample average."""

    rng = np.random.default_rng(19)
    samples = rng.standard_normal((64, target.dimension))
    weights = rng.random(64)
    weights /= weights.sum()

    fast = mean_hessian(target, samples, weights)
    reference = np.einsum(
        "s,sij->ij", weights, np.asarray(target.hessian(samples), dtype=np.float64)
    )
    assert np.allclose(fast, reference, rtol=1e-12, atol=1e-12)
    assert np.allclose(fast, fast.T, atol=1e-13)
