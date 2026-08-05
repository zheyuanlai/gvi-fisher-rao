from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.targets import GaussianTarget, LogisticRegressionTarget, ShiftedLogCoshTarget


def finite_gradient(target: object, x: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
    result = np.zeros_like(x)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = epsilon
        result[index] = (target.value(x + direction) - target.value(x - direction)) / (2 * epsilon)
    return result


def finite_hessian(target: object, x: np.ndarray, epsilon: float = 2.0e-5) -> np.ndarray:
    result = np.zeros((x.size, x.size))
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = epsilon
        result[:, index] = (target.grad(x + direction) - target.grad(x - direction)) / (2 * epsilon)
    return result


@pytest.mark.parametrize("kind", ["gaussian", "logcosh", "logistic"])
def test_target_derivatives(kind: str) -> None:
    rng = np.random.default_rng(12)
    if kind == "gaussian":
        target = GaussianTarget(np.asarray([0.2, -0.1]), np.asarray([[2.0, 0.3], [0.3, 1.0]]))
    elif kind == "logcosh":
        target = ShiftedLogCoshTarget(
            np.asarray([0.4, 1.2]),
            0.7,
            np.asarray([-0.3, 0.2]),
            np.asarray([[1.1, 0.2], [-0.1, 0.8]]),
            np.asarray([0.2, -0.4]),
        )
    else:
        features = rng.standard_normal((7, 2))
        labels = rng.integers(0, 2, size=7).astype(float)
        target = LogisticRegressionTarget(features, labels, 0.6)
    point = np.asarray([0.17, -0.28])
    np.testing.assert_allclose(target.grad(point), finite_gradient(target, point), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(target.hessian(point), finite_hessian(target, point), rtol=2e-5, atol=2e-6)


def test_vectorized_target_shapes() -> None:
    target = ShiftedLogCoshTarget.base(3, nu=0.5, rho=1.0)
    points = np.zeros((4, 3))
    assert target.value(points).shape == (4,)
    assert target.grad(points).shape == (4, 3)
    assert target.hessian(points).shape == (4, 3, 3)

