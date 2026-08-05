from __future__ import annotations

import numpy as np

from fr_gvi.experiments.factories import build_problem
from fr_gvi.targets import GaussianTarget


def test_burnin_precision_spectrum_spans_one_to_kappa() -> None:
    problem = build_problem(
        {
            "kind": "gaussian",
            "dimension": 6,
            "condition": 1000.0,
            "spectrum_scale": "unit_min",
            "rotation": True,
            "mean_scale": 0.0,
        },
        seed=71,
        experiment="A",
    )
    assert isinstance(problem.target, GaussianTarget)
    eigenvalues = np.linalg.eigvalsh(problem.target.precision)
    np.testing.assert_allclose([eigenvalues[0], eigenvalues[-1]], [1.0, 1000.0], rtol=2e-13)


def test_affine_parameter_k_is_condition_of_a_a_transpose() -> None:
    requested = 1.0e4
    problem = build_problem(
        {"kind": "gaussian", "dimension": 5, "affine_condition": requested},
        seed=72,
        experiment="B",
    )
    transform = np.asarray(problem.metadata["transform"], dtype=np.float64)
    observed = np.linalg.cond(transform @ transform.T)
    np.testing.assert_allclose(observed, requested, rtol=2e-9)

