from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms import AlgorithmFailure, GaussianState, Method, quadratic_rescue, step
from fr_gvi.algorithms.core import _sampled, _validated_state
from fr_gvi.expectations import ExactGaussianExpectation
from fr_gvi.linear_algebra.spd import jko_entropy_eigenvalue_map
from fr_gvi.targets import GaussianTarget
from fr_gvi.utils.accounting import OperationCounts


def target_state() -> tuple[GaussianTarget, GaussianState]:
    target = GaussianTarget(np.asarray([0.3, -0.4]), np.asarray([[1.5, 0.2], [0.2, 0.8]]))
    return target, GaussianState(target.mean, target.covariance)


@pytest.mark.parametrize("method", [Method.FR_R, Method.FR_KL, Method.FB_GVI])
def test_deterministic_gaussian_optimizer_is_fixed(method: Method) -> None:
    target, optimum = target_state()
    next_state, _ = step(
        method,
        target,
        optimum,
        0.1,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(2),
        batch_size=1,
        counts=OperationCounts(),
    )
    np.testing.assert_allclose(next_state.mean, optimum.mean, atol=2e-14)
    np.testing.assert_allclose(next_state.covariance, optimum.covariance, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("method", [Method.FR_R_STL, Method.FR_KL_STL])
def test_stl_gaussian_optimizer_is_fixed_pathwise(method: Method) -> None:
    target, optimum = target_state()
    next_state, _ = step(
        method,
        target,
        optimum,
        0.1,
        engine=None,
        rng=np.random.default_rng(33),
        batch_size=1,
        counts=OperationCounts(),
    )
    np.testing.assert_allclose(next_state.mean, optimum.mean, atol=2e-14)
    np.testing.assert_allclose(next_state.covariance, optimum.covariance, rtol=3e-13, atol=3e-13)


def test_quadratic_rescue_exactly_recovers_gaussian() -> None:
    target, optimum = target_state()
    rescued = quadratic_rescue(target, np.asarray([10.0, -7.0]))
    np.testing.assert_allclose(rescued.mean, optimum.mean, atol=3e-15)
    np.testing.assert_allclose(rescued.covariance, optimum.covariance, atol=3e-15)


@pytest.mark.parametrize("method", [Method.FR_R, Method.FR_KL])
def test_fisher_rao_updates_match_flow_to_first_order(method: Method) -> None:
    target = GaussianTarget(np.asarray([0.2, -0.1]), np.asarray([[1.2, 0.1], [0.1, 0.7]]))
    state = GaussianState(np.asarray([0.8, -0.6]), np.asarray([[0.9, 0.2], [0.2, 1.4]]))
    expectation = ExactGaussianExpectation().evaluate(target, state.mean, state.covariance)
    g = -expectation.grad
    covariance_velocity = state.covariance - state.covariance @ expectation.hessian @ state.covariance
    h = 1.0e-7
    updated, _ = step(
        method,
        target,
        state,
        h,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(3),
        batch_size=1,
        counts=OperationCounts(),
    )
    np.testing.assert_allclose((updated.mean - state.mean) / h, state.covariance @ g, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose((updated.covariance - state.covariance) / h, covariance_velocity, rtol=3e-7, atol=3e-7)


def test_fb_gvi_matches_algorithm_one_formula() -> None:
    target = GaussianTarget(np.asarray([0.1, 0.2]), np.diag([0.8, 1.4]))
    state = GaussianState(np.asarray([0.6, -0.3]), np.diag([0.7, 1.2]))
    h = 0.2
    updated, _ = step(
        Method.FB_GVI,
        target,
        state,
        h,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(1),
        batch_size=1,
        counts=OperationCounts(),
    )
    gradient = target.precision @ (state.mean - target.mean)
    forward = np.eye(2) - h * target.precision
    half = forward @ state.covariance @ forward
    expected_covariance = np.diag(jko_entropy_eigenvalue_map(np.diag(half), h))
    np.testing.assert_allclose(updated.mean, state.mean - h * gradient)
    np.testing.assert_allclose(updated.covariance, expected_covariance)


def test_gaussian_stl_mean_noise_vanishes_at_matched_covariance() -> None:
    target = GaussianTarget(np.asarray([0.2, -0.5]), np.asarray([[1.3, 0.2], [0.2, 0.9]]))
    state = GaussianState(np.asarray([1.0, 0.7]), target.covariance)
    directions = []
    for seed in range(5):
        direction, _ = _sampled(
            target,
            state,
            np.random.default_rng(seed),
            1,
            stl=True,
            counts=OperationCounts(),
        )
        directions.append(direction)
    for direction in directions[1:]:
        np.testing.assert_allclose(direction, directions[0], atol=4e-15)


def test_sampled_estimators_are_empirically_unbiased() -> None:
    target = GaussianTarget(np.asarray([0.2, -0.5]), np.asarray([[1.3, 0.2], [0.2, 0.9]]))
    state = GaussianState(np.asarray([0.7, 0.3]), np.asarray([[0.8, 0.1], [0.1, 1.1]]))
    rng = np.random.default_rng(123)
    estimates = []
    for _ in range(3000):
        estimate, hessian = _sampled(target, state, rng, 1, stl=True, counts=OperationCounts())
        estimates.append(estimate)
        np.testing.assert_allclose(hessian, target.precision)
    expected = -target.precision @ (state.mean - target.mean)
    np.testing.assert_allclose(np.mean(estimates, axis=0), expected, atol=0.035)


def test_common_random_number_pairing_is_deterministic() -> None:
    target = GaussianTarget(np.zeros(2), np.eye(2))
    state = GaussianState(np.ones(2), 2.0 * np.eye(2))
    outputs = []
    for _ in range(2):
        outputs.append(
            step(
                Method.FR_R_STL,
                target,
                state,
                0.1,
                engine=None,
                rng=np.random.default_rng(88),
                batch_size=3,
                counts=OperationCounts(),
            )[0]
        )
    np.testing.assert_array_equal(outputs[0].mean, outputs[1].mean)
    np.testing.assert_array_equal(outputs[0].covariance, outputs[1].covariance)


def test_no_silent_covariance_clipping() -> None:
    with pytest.raises(AlgorithmFailure, match="lost positive definiteness"):
        _validated_state(np.zeros(2), np.diag([1.0, -0.01]), OperationCounts())

