from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms.affine_metric import (
    AffineMetric,
    covariance_velocity,
    modal_decomposition,
    retraction_step,
)
from fr_gvi.algorithms.core import GaussianState, Method, step
from fr_gvi.expectations.core import ExactGaussianExpectation
from fr_gvi.targets.core import GaussianTarget
from fr_gvi.utils.accounting import OperationCounts


def _problem() -> tuple[GaussianTarget, GaussianState]:
    rng = np.random.default_rng(7)
    dimension = 4
    root = rng.standard_normal((dimension, dimension))
    precision = root @ root.T + dimension * np.eye(dimension)
    target = GaussianTarget(rng.standard_normal(dimension), precision)
    perturbation = rng.standard_normal((dimension, dimension))
    covariance = np.eye(dimension) + 0.1 * (perturbation + perturbation.T) / 2.0
    return target, GaussianState(rng.standard_normal(dimension), covariance)


def test_fisher_rao_is_the_member_omega_half_tau_zero() -> None:
    """(omega, tau) = (1/2, 0) must reproduce the Fisher--Rao retraction exactly."""

    target, state = _problem()
    metric = AffineMetric(0.5, 0.0, target.dimension)
    assert metric.is_fisher_rao

    general = retraction_step(metric, target, state, 0.13, engine=ExactGaussianExpectation())
    fisher_rao, _ = step(
        Method.FR_R,
        target,
        state,
        0.13,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(0),
        batch_size=1,
        counts=OperationCounts(),
    )
    assert np.allclose(general.mean, fisher_rao.mean, rtol=0.0, atol=1e-13)
    assert np.allclose(general.covariance, fisher_rao.covariance, rtol=0.0, atol=1e-13)


def test_covariance_velocity_reduces_to_fisher_rao() -> None:
    target, state = _problem()
    metric = AffineMetric(0.5, 0.0, target.dimension)
    velocity = covariance_velocity(metric, state.covariance, target.precision)
    expected = state.covariance - state.covariance @ target.precision @ state.covariance
    assert np.allclose(velocity, expected, atol=1e-13)


def test_metric_rejects_indefinite_parameters() -> None:
    with pytest.raises(ValueError, match="tau"):
        AffineMetric(1.0, -1.0 / 3.0, 3)
    with pytest.raises(ValueError, match="omega"):
        AffineMetric(0.0, 0.0, 3)


@pytest.mark.parametrize(
    ("omega", "tau"),
    [(0.5, 0.0), (1.0, 0.0), (0.25, 0.5), (2.0, -0.1), (0.75, 1.5)],
)
def test_predicted_modal_rates_are_attained(omega: float, tau: float) -> None:
    """The linearized traceless and trace modes decay at 1/(2w) and 1/(2(w+tN))."""

    dimension = 4
    target = GaussianTarget(np.zeros(dimension), np.eye(dimension))
    metric = AffineMetric(omega, tau, dimension)
    engine = ExactGaussianExpectation()

    rng = np.random.default_rng(11)
    raw = rng.standard_normal((dimension, dimension))
    traceless = (raw + raw.T) / 2.0
    traceless -= np.trace(traceless) / dimension * np.eye(dimension)
    traceless /= np.linalg.norm(traceless, ord="fro")
    epsilon = 1e-4
    covariance = np.eye(dimension) + epsilon * (traceless + np.eye(dimension) / np.sqrt(dimension))

    state = GaussianState(np.zeros(dimension), covariance)
    step_size = 0.01
    iterations = 200
    history = []
    for iteration in range(iterations + 1):
        history.append((iteration * step_size, *modal_decomposition(state.covariance)))
        if iteration < iterations:
            state = retraction_step(metric, target, state, step_size, engine=engine)

    array = np.asarray(history)
    traceless_rate = -np.polyfit(array[:, 0], np.log(array[:, 1]), 1)[0]
    trace_rate = -np.polyfit(array[:, 0], np.log(np.abs(array[:, 2])), 1)[0]

    assert traceless_rate == pytest.approx(metric.traceless_rate, rel=0.02)
    assert trace_rate == pytest.approx(metric.trace_rate, rel=0.02)
