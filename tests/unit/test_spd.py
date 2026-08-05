from __future__ import annotations

import numpy as np
import pytest

from fr_gvi.linear_algebra.spd import (
    SPDValidationError,
    ensure_spd,
    jko_entropy_eigenvalue_map,
    spd_exp,
    spd_inv_sqrt,
    spd_log,
    spd_sqrt,
)


def random_spd(seed: int = 4, dimension: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dimension, dimension))
    return matrix @ matrix.T + 0.4 * np.eye(dimension)


def test_spd_spectral_functions_round_trip() -> None:
    matrix = random_spd()
    root = spd_sqrt(matrix)
    inverse_root = spd_inv_sqrt(matrix)
    np.testing.assert_allclose(root @ root, matrix, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(inverse_root @ matrix @ inverse_root, np.eye(5), rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(spd_exp(spd_log(matrix)), matrix, rtol=5e-13, atol=5e-13)


def test_jko_eigenvalue_map_matches_scalar_formula() -> None:
    values = np.asarray([0.0, 0.2, 4.0])
    step_size = 0.3
    expected = 0.5 * (values + 2 * step_size + np.sqrt(values * (values + 4 * step_size)))
    np.testing.assert_allclose(jko_entropy_eigenvalue_map(values, step_size), expected)


def test_only_roundoff_scale_repair_is_allowed() -> None:
    tiny = np.diag([1.0, -10.0 * np.finfo(np.float64).eps])
    repaired, record = ensure_spd(tiny)
    assert record is not None
    assert np.linalg.eigvalsh(repaired)[0] > 0.0
    with pytest.raises(SPDValidationError, match="lost positive definiteness"):
        ensure_spd(np.diag([1.0, -1.0e-4]))

