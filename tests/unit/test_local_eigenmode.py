"""The local-rate initialization and the affine-transform guard."""

from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.diagnostics.local_operator import assemble_local_operator, symmetric_star_basis
from fr_gvi.experiments.campaign import _local_eigenmode_state
from fr_gvi.expectations.core import FixedNormalExpectation
from fr_gvi.linear_algebra.spd import spd_inv_sqrt
from fr_gvi.targets.core import GaussianTarget, ShiftedLogCoshTarget


def _star_distance(state: GaussianState, optimum: GaussianState) -> float:
    """``||a - a_star||_star`` to first order, the quantity the local panels report."""

    inverse_root = spd_inv_sqrt(optimum.covariance)
    mean = inverse_root @ (state.mean - optimum.mean)
    covariance = inverse_root @ state.covariance @ inverse_root - np.eye(state.mean.size)
    return float(np.sqrt(mean @ mean + 0.5 * np.linalg.norm(covariance, ord="fro") ** 2))


@pytest.mark.parametrize("radius", [1.0e-1, 5.0e-2, 1.0e-2])
def test_eigenmode_initialization_sits_at_the_requested_radius(radius: float) -> None:
    dimension = 4
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    precision = rotation @ np.diag(np.linspace(0.7, 2.0, dimension)) @ rotation.T
    target = GaussianTarget(np.zeros(dimension), precision)
    optimum = GaussianState(target.mean, target.covariance)
    normals = FixedNormalExpectation.qmc(dimension, 1024, 3).normals
    operator = assemble_local_operator(target, optimum, normals)

    state, spectrum = _local_eigenmode_state(operator, optimum, radius)
    # The perturbation is applied at unit star-norm, so the initial distance is the
    # radius up to the second-order term of the covariance parameterization.
    assert _star_distance(state, optimum) == pytest.approx(radius, rel=0.05)
    assert float(np.linalg.eigvalsh(state.covariance)[0]) > 0.0
    assert spectrum[0] <= spectrum[-1]


def test_eigenmode_direction_is_the_slowest_mode() -> None:
    """The perturbation lies along the eigenvector of the smallest eigenvalue."""

    dimension = 3
    target = GaussianTarget(np.zeros(dimension), np.eye(dimension))
    optimum = GaussianState(target.mean, target.covariance)
    normals = FixedNormalExpectation.qmc(dimension, 2048, 5).normals
    operator = assemble_local_operator(target, optimum, normals)
    values, vectors = np.linalg.eigh(operator)

    radius = 1.0e-3
    state, _ = _local_eigenmode_state(operator, optimum, radius)
    basis = symmetric_star_basis(dimension)
    direction = np.concatenate(
        [
            (state.mean - optimum.mean) / radius,
            [
                0.5 * np.trace(matrix @ (state.covariance - optimum.covariance)) / radius
                for matrix in basis
            ],
        ]
    )
    expected = vectors[:, 0]
    alignment = abs(float(direction @ expected) / float(np.linalg.norm(direction)))
    assert alignment == pytest.approx(1.0, abs=1e-6)
    assert values[0] == pytest.approx(np.min(values))


def test_sign_convention_is_reproducible() -> None:
    dimension = 3
    target = GaussianTarget(np.zeros(dimension), np.diag([1.0, 2.0, 3.0]))
    optimum = GaussianState(target.mean, target.covariance)
    normals = FixedNormalExpectation.qmc(dimension, 512, 1).normals
    operator = assemble_local_operator(target, optimum, normals)
    first, _ = _local_eigenmode_state(operator, optimum, 0.05)
    second, _ = _local_eigenmode_state(operator, optimum, 0.05)
    assert np.allclose(first.mean, second.mean)
    assert np.allclose(first.covariance, second.covariance)


def test_invertibility_guard_accepts_a_well_conditioned_map_with_tiny_determinant() -> None:
    """Invertibility is about conditioning, not scale.

    At d = 50 a map with condition number 7 has |det| near 1e-22, far below eps, so
    a determinant test rejects it while the singular-value test accepts it.
    """

    dimension = 50
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    scale = np.linspace(0.97, 6.8, dimension)
    transform = rotation / scale
    assert abs(float(np.linalg.det(transform))) < np.finfo(np.float64).eps
    assert np.linalg.cond(transform) < 10.0

    target = ShiftedLogCoshTarget(
        np.ones(dimension), 1.0, np.zeros(dimension), transform, np.zeros(dimension)
    )
    assert target.dimension == dimension


def test_invertibility_guard_still_rejects_a_singular_map() -> None:
    transform = np.eye(4)
    transform[3, 3] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        ShiftedLogCoshTarget(np.ones(4), 1.0, np.zeros(4), transform, np.zeros(4))
