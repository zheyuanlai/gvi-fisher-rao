from __future__ import annotations

import numpy as np

from fr_gvi.expectations import ExactGaussianExpectation, FixedNormalExpectation, GaussHermiteLogCoshExpectation
from fr_gvi.targets import GaussianTarget, ShiftedLogCoshTarget


def test_exact_gaussian_expectation() -> None:
    target = GaussianTarget(np.asarray([0.1, -0.2]), np.asarray([[2.0, 0.2], [0.2, 1.0]]))
    mean = np.asarray([0.5, 0.3])
    covariance = np.asarray([[0.8, 0.1], [0.1, 0.6]])
    result = ExactGaussianExpectation().evaluate(target, mean, covariance)
    delta = mean - target.mean
    assert abs(result.value - 0.5 * (delta @ target.precision @ delta + np.trace(target.precision @ covariance))) < 1e-14
    np.testing.assert_allclose(result.grad, target.precision @ delta)
    np.testing.assert_allclose(result.hessian, target.precision)


def test_fixed_qmc_seed_reproducibility() -> None:
    first = FixedNormalExpectation.qmc(3, 100, 19)
    second = FixedNormalExpectation.qmc(3, 100, 19)
    np.testing.assert_array_equal(first.normals, second.normals)
    assert first.backend == "scrambled_sobol"


def test_logcosh_gauss_hermite_matches_large_qmc() -> None:
    target = ShiftedLogCoshTarget.base(2, nu=np.asarray([0.4, 1.0]), rho=0.7)
    mean = np.asarray([0.2, -0.1])
    covariance = np.asarray([[0.8, 0.25], [0.25, 1.3]])
    quadrature = GaussHermiteLogCoshExpectation(60).evaluate(target, mean, covariance)
    qmc = FixedNormalExpectation.qmc(2, 2**16, 101).evaluate(target, mean, covariance)
    np.testing.assert_allclose(quadrature.grad, qmc.grad, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(quadrature.hessian, qmc.hessian, atol=2e-4, rtol=2e-4)

