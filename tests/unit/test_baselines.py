"""The three external square-root baselines, against their published form.

Each test pins one property that would silently break the comparison if it were
wrong: the fixed point (which decides whether a method converges to the same
optimizer at all), the gradient convention (which decides whether the covariance
moves in the right direction), and the small-step limit (which decides whether
Sq--NGVI really is a different discretization of the same flow rather than a
different flow).
"""

from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms import AlgorithmFailure, GaussianState, Method, step
from fr_gvi.algorithms.baselines import (
    FactorBreakdown,
    _entropy_proximal_diagonal,
    sticking_the_landing_parameter_gradient,
)
from fr_gvi.expectations import ExactGaussianExpectation
from fr_gvi.targets import GaussianTarget

from fr_gvi.utils.accounting import OperationCounts

BASELINES = [Method.SQ_NGVI, Method.PRICE_BBVI, Method.BBVI_STL]


def target_state() -> tuple[GaussianTarget, GaussianState]:
    target = GaussianTarget(np.asarray([0.3, -0.4]), np.asarray([[1.5, 0.2], [0.2, 0.8]]))
    return target, GaussianState(target.mean, target.covariance)


def run(method: Method, target, state: GaussianState, step_size: float, *, seed: int = 5, batch: int = 1):
    return step(
        method,
        target,
        state,
        step_size,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(seed),
        batch_size=batch,
        counts=OperationCounts(),
        projection_floor=1.0 / np.sqrt(float(np.linalg.eigvalsh(target.precision)[-1]))
        if method is Method.BBVI_STL
        else None,
    )


@pytest.mark.parametrize("method", BASELINES)
def test_gaussian_optimizer_covariance_is_fixed_pathwise(method: Method) -> None:
    """Every baseline leaves the optimizing covariance where it is, pathwise.

    For Sq--NGVI this checks the ``tril`` convention with the halved diagonal;
    for Price--BBVI it checks that the Price energy gradient and the entropy
    proximal step cancel exactly, which they only do if both are right; for
    BBVI--STL it is the covariance half of the estimator's cancellation.
    """

    target, optimum = target_state()
    next_state, _ = run(method, target, optimum, 0.1)
    np.testing.assert_allclose(next_state.covariance, optimum.covariance, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", [Method.SQ_NGVI, Method.BBVI_STL])
def test_landing_methods_fix_the_optimizing_mean_pathwise(method: Method) -> None:
    target, optimum = target_state()
    next_state, _ = run(method, target, optimum, 0.1)
    np.testing.assert_allclose(next_state.mean, optimum.mean, atol=2e-14)


def test_price_bbvi_mean_is_unbiased_but_not_pathwise_fixed() -> None:
    """Price--BBVI uses the raw Bonnet mean estimator, so it does not land.

    This is not a defect: it is the property the stochastic figure measures.
    ``FR--R--STL``, ``FR--KL--STL`` and ``BBVI--STL`` subtract the score and
    their mean noise vanishes pathwise once the covariance is matched, while the
    Bonnet mean estimator retains ``O(1)`` across-seed dispersion at the
    optimizer. Only its expectation vanishes.
    """

    target, optimum = target_state()
    displacements = []
    for seed in range(400):
        next_state, _ = run(Method.PRICE_BBVI, target, optimum, 0.1, seed=seed)
        displacements.append(next_state.mean - optimum.mean)
    displacements = np.asarray(displacements)
    assert np.linalg.norm(displacements, axis=1).min() > 1e-6
    np.testing.assert_allclose(displacements.mean(axis=0), np.zeros(2), atol=6e-3)


def test_price_bbvi_fixed_point_holds_across_step_sizes() -> None:
    """The energy/entropy cancellation is exact, not a small-step accident."""

    target, optimum = target_state()
    for step_size in (1e-3, 1e-2, 0.1, 0.5, 1.0):
        next_state, _ = run(Method.PRICE_BBVI, target, optimum, step_size)
        np.testing.assert_allclose(next_state.covariance, optimum.covariance, rtol=1e-11, atol=1e-11)


def test_entropy_proximal_solves_its_own_optimality_condition() -> None:
    factor = np.asarray([[0.7, 0.0], [-0.3, 0.2]])
    for step_size in (1e-3, 0.25, 3.0):
        updated = _entropy_proximal_diagonal(factor, step_size)
        diagonal = np.diag(updated)
        # c' - c - gamma / c' = 0, and the off-diagonal is untouched.
        np.testing.assert_allclose(
            diagonal - np.diag(factor) - step_size / diagonal, 0.0, atol=1e-14
        )
        np.testing.assert_allclose(np.tril(updated, -1), np.tril(factor, -1), atol=0.0)
        assert np.all(diagonal > 0.0)


def test_square_root_natural_gradient_matches_the_flow_to_first_order() -> None:
    """Sq--NGVI and FR--R discretize the same natural-gradient flow.

    Both are consistent one-step approximations of the same vector field, so
    their disagreement after one step must be ``O(h^2)``.  Halving the step must
    therefore quarter the discrepancy; a wrong ``tril`` convention or a stray
    factor of two would leave an ``O(h)`` term and the ratio would go to two.
    """

    target = GaussianTarget(np.asarray([0.2, -0.1]), np.asarray([[2.0, 0.3], [0.3, 1.1]]))
    state = GaussianState(np.asarray([1.0, -0.5]), np.asarray([[0.6, 0.1], [0.1, 0.4]]))
    discrepancies = []
    steps = [4e-3, 2e-3, 1e-3]
    for step_size in steps:
        riemannian, _ = run(Method.FR_R, target, state, step_size)
        square_root, _ = run(Method.SQ_NGVI, target, state, step_size)
        np.testing.assert_allclose(riemannian.mean, square_root.mean, atol=1e-15)
        discrepancies.append(
            float(np.linalg.norm(riemannian.covariance - square_root.covariance, ord="fro"))
        )
    ratios = [discrepancies[i] / discrepancies[i + 1] for i in range(len(steps) - 1)]
    assert all(3.6 < ratio < 4.4 for ratio in ratios), ratios


def test_price_energy_gradient_matches_finite_differences() -> None:
    """``grad_C E[V(m + C eps)] = E[grad^2 V] C`` on the triangular parameters.

    Checked against a central difference of the exact Gaussian energy, so the
    Price identity and the restriction to the lower triangle are both tested.
    """

    target = GaussianTarget(np.asarray([0.4, -0.2]), np.asarray([[1.3, 0.25], [0.25, 0.9]]))
    factor = np.asarray([[0.8, 0.0], [0.35, 0.6]])
    mean = np.asarray([0.7, -0.3])

    def energy(matrix: np.ndarray) -> float:
        covariance = matrix @ matrix.T
        delta = mean - target.mean
        return float(
            0.5 * delta @ target.precision @ delta
            + 0.5 * np.trace(target.precision @ covariance)
        )

    analytic = np.tril(target.precision @ factor)
    numeric = np.zeros_like(factor)
    epsilon = 1e-6
    for i in range(2):
        for j in range(i + 1):
            perturbed = factor.copy()
            perturbed[i, j] += epsilon
            plus = energy(perturbed)
            perturbed[i, j] -= 2.0 * epsilon
            numeric[i, j] = (plus - energy(perturbed)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_sticking_the_landing_estimator_is_unbiased() -> None:
    """The STL parameter gradient averages to the exact reparameterization one.

    On a Gaussian target the exact energy gradients are ``Q(m - mu)`` and
    ``tril(Q C)``, and the entropy contributes ``-diag(1/C_ii)`` to the factor.
    """

    precision = np.asarray([[1.4, 0.3], [0.3, 0.9]])
    target = GaussianTarget(np.asarray([0.1, -0.25]), precision)
    factor = np.asarray([[0.9, 0.0], [0.2, 0.7]])
    mean = np.asarray([0.5, 0.2])
    rng = np.random.default_rng(11)
    batch = 400_000
    normals = rng.standard_normal((batch, 2))
    samples = mean + normals @ factor.T
    gradients = target.grad(samples)
    mean_gradient, factor_gradient = sticking_the_landing_parameter_gradient(
        factor, normals, gradients
    )
    expected_mean = precision @ (mean - target.mean)
    expected_factor = np.tril(precision @ factor)
    np.fill_diagonal(expected_factor, np.diag(expected_factor) - 1.0 / np.diag(factor))
    np.testing.assert_allclose(mean_gradient, expected_mean, rtol=0.0, atol=6e-3)
    np.testing.assert_allclose(factor_gradient, expected_factor, rtol=0.0, atol=1e-2)


def test_square_root_factor_breakdown_is_a_failure_not_a_repair() -> None:
    """An overlong Sq--NGVI step must be recorded, never silently continued.

    The covariance ``C C^T`` stays positive definite when a diagonal entry of the
    factor changes sign, so nothing downstream would notice.
    """

    target = GaussianTarget(np.zeros(2), np.diag([50.0, 50.0]))
    state = GaussianState(np.zeros(2), np.eye(2))
    with pytest.raises(AlgorithmFailure, match="positive diagonal"):
        run(Method.SQ_NGVI, target, state, 1.0)


def test_bbvi_stl_projection_is_counted() -> None:
    """The published projection is applied and its activations are recorded."""

    target = GaussianTarget(np.zeros(2), np.diag([4.0, 4.0]))
    state = GaussianState(np.asarray([3.0, 3.0]), np.eye(2))
    counts = OperationCounts()
    _, diagnostics = step(
        Method.BBVI_STL,
        target,
        state,
        0.6,
        engine=None,
        rng=np.random.default_rng(3),
        batch_size=1,
        counts=counts,
        projection_floor=1.0 / np.sqrt(4.0),
    )
    assert diagnostics.projection_activations == counts.projection_activations
    assert counts.projection_activations >= 1


def test_bbvi_stl_requires_its_projection_floor() -> None:
    target, optimum = target_state()
    with pytest.raises(ValueError, match="projection floor"):
        step(
            Method.BBVI_STL,
            target,
            optimum,
            0.1,
            engine=None,
            rng=np.random.default_rng(1),
            batch_size=1,
            counts=OperationCounts(),
        )


@pytest.mark.parametrize("method", BASELINES)
def test_cost_accounting_splits_oracle_from_algebra(method: Method) -> None:
    target = GaussianTarget(np.zeros(4), np.diag([1.0, 1.4, 0.8, 1.1]))
    state = GaussianState(np.ones(4), np.eye(4))
    counts = OperationCounts()
    _, diagnostics = step(
        method,
        target,
        state,
        0.05,
        engine=ExactGaussianExpectation(),
        rng=np.random.default_rng(7),
        batch_size=4,
        counts=counts,
        projection_floor=0.5 if method is Method.BBVI_STL else None,
    )
    assert diagnostics.oracle_seconds > 0.0
    assert diagnostics.linear_algebra_seconds >= 0.0
    assert counts.oracle_seconds == pytest.approx(diagnostics.oracle_seconds)


def test_bbvi_stl_uses_no_hessian() -> None:
    """The gradient-only arm must not be charged, or credited, with Hessians."""

    target, optimum = target_state()
    counts = OperationCounts()
    step(
        Method.BBVI_STL,
        target,
        optimum,
        0.05,
        engine=None,
        rng=np.random.default_rng(4),
        batch_size=8,
        counts=counts,
        projection_floor=0.5,
    )
    assert counts.gradient_evaluations == 8
    assert counts.hessian_evaluations == 0
    assert counts.oracle_pairs == 0


def test_cholesky_round_trip_is_exact_on_the_admissible_set() -> None:
    """Carrying ``(m, Sigma)`` loses nothing while the factor stays admissible."""

    rng = np.random.default_rng(19)
    raw = np.tril(rng.standard_normal((6, 6)))
    np.fill_diagonal(raw, np.abs(np.diag(raw)) + 0.2)
    recovered = np.linalg.cholesky(raw @ raw.T)
    np.testing.assert_allclose(recovered, raw, rtol=1e-12, atol=1e-12)


def test_factor_breakdown_is_a_value_error_subclass() -> None:
    """So the campaign's existing failure handling catches it unchanged."""

    assert issubclass(FactorBreakdown, ValueError)
