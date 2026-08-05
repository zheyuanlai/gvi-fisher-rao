from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms import GaussianState, Method, step
from fr_gvi.expectations import ExactGaussianExpectation
from fr_gvi.targets import GaussianTarget
from fr_gvi.utils.accounting import OperationCounts


@pytest.mark.parametrize("method", [Method.FR_R, Method.FR_KL])
def test_fisher_rao_iterates_are_affine_equivariant(method: Method) -> None:
    rng = np.random.default_rng(91)
    base_target = GaussianTarget(np.asarray([0.2, -0.3, 0.4]), np.diag([0.7, 1.0, 1.4]))
    base_state = GaussianState(
        np.asarray([0.8, 0.1, -0.7]),
        np.asarray([[1.2, 0.1, 0.0], [0.1, 0.8, 0.05], [0.0, 0.05, 1.5]]),
    )
    transform = rng.standard_normal((3, 3))
    transform += 2.0 * np.eye(3)
    shift = rng.standard_normal(3)
    inverse = np.linalg.solve(transform, np.eye(3))
    transformed_target = GaussianTarget(
        transform @ base_target.mean + shift,
        inverse.T @ base_target.precision @ inverse,
    )
    transformed_state = GaussianState(
        transform @ base_state.mean + shift,
        transform @ base_state.covariance @ transform.T,
    )
    for _ in range(5):
        base_state, _ = step(
            method,
            base_target,
            base_state,
            0.1,
            engine=ExactGaussianExpectation(),
            rng=np.random.default_rng(1),
            batch_size=1,
            counts=OperationCounts(),
        )
        transformed_state, _ = step(
            method,
            transformed_target,
            transformed_state,
            0.1,
            engine=ExactGaussianExpectation(),
            rng=np.random.default_rng(1),
            batch_size=1,
            counts=OperationCounts(),
        )
        np.testing.assert_allclose(transformed_state.mean, transform @ base_state.mean + shift, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(transformed_state.covariance, transform @ base_state.covariance @ transform.T, rtol=2e-11, atol=2e-11)

