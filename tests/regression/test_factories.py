from __future__ import annotations

import numpy as np

from fr_gvi.experiments.factories import build_problem
from fr_gvi.targets import GaussianTarget


def test_gaussian_initial_covariance_can_match_target() -> None:
    problem = build_problem(
        {
            "kind": "gaussian",
            "dimension": 4,
            "condition": 100.0,
            "rotation": True,
            "initial_covariance": "target",
        },
        seed=13,
        experiment="H",
    )
    assert isinstance(problem.target, GaussianTarget)
    np.testing.assert_allclose(problem.initial_state.covariance, problem.target.covariance)


def test_gaussian_initial_covariance_diagonal_is_honored() -> None:
    diagonal = np.asarray([0.25, 1.0])
    problem = build_problem(
        {
            "kind": "gaussian",
            "dimension": 2,
            "condition": 1.0,
            "rotation": False,
            "initial_covariance_diagonal": diagonal.tolist(),
        },
        seed=14,
        experiment="F",
    )
    np.testing.assert_allclose(problem.initial_state.covariance, np.diag(diagonal))

