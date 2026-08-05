from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from fr_gvi.expectations.core import GaussHermiteLogCoshExpectation
from fr_gvi.targets.core import ShiftedLogCoshTarget


def _adaptive_expected_hessian(nu: float, rho: float, offset: float, mean: float, variance: float) -> float:
    deviation = np.sqrt(variance)

    def integrand(z: float) -> float:
        curvature = nu + rho * (1.0 - np.tanh(z - offset) ** 2)
        density = np.exp(-0.5 * ((z - mean) / deviation) ** 2) / (deviation * np.sqrt(2.0 * np.pi))
        return curvature * density

    value, _ = quad(integrand, mean - 40 * deviation, mean + 40 * deviation, limit=800)
    return float(value)


@pytest.mark.parametrize("variance", [0.5, 5.0, 56.8, 400.0])
def test_separable_quadrature_resolves_a_narrow_peak_in_a_wide_marginal(variance: float) -> None:
    """The sech^2 peak has unit width while the marginal can be far wider.

    Gauss--Hermite spaces its nodes proportionally to the marginal width and is
    still wrong by 5e-4 at order 200 on the widest case here, so the engine uses
    truncated Gauss--Legendre instead.  This pins that accuracy down against an
    adaptive integrator.
    """

    nu, rho, offset, mean = 0.01, 0.1, 0.5, 0.3
    target = ShiftedLogCoshTarget(
        np.asarray([nu]), rho, np.asarray([offset]), np.eye(1), np.zeros(1)
    )
    result = GaussHermiteLogCoshExpectation(80).evaluate(
        target, np.asarray([mean]), np.asarray([[variance]])
    )
    expected = _adaptive_expected_hessian(nu, rho, offset, mean, variance)
    assert float(result.hessian[0, 0]) == pytest.approx(expected, rel=1e-10)


def test_quadrature_is_converged_in_its_order() -> None:
    """Doubling the node count must not move the answer."""

    rng = np.random.default_rng(4)
    dimension = 6
    target = ShiftedLogCoshTarget(
        np.geomspace(0.01, 1.0, dimension),
        0.1,
        np.linspace(-0.5, 0.5, dimension),
        np.eye(dimension) + 0.1 * rng.standard_normal((dimension, dimension)),
        rng.standard_normal(dimension),
    )
    mean = rng.standard_normal(dimension)
    covariance = np.diag(np.geomspace(0.5, 60.0, dimension))

    coarse = GaussHermiteLogCoshExpectation(80).evaluate(target, mean, covariance)
    fine = GaussHermiteLogCoshExpectation(160).evaluate(target, mean, covariance)
    assert np.allclose(coarse.hessian, fine.hessian, rtol=1e-11, atol=1e-14)
    assert np.allclose(coarse.grad, fine.grad, rtol=1e-11, atol=1e-14)
    assert coarse.value == pytest.approx(fine.value, rel=1e-11)
