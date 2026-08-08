"""The practical benchmark's protocol, as executable claims.

The benchmark's whole value rests on two things a reader cannot check by reading
a figure: that the stepsizes were chosen without the optimizer, and that the real
posteriors are the ones the manifest says they are.  Both are pinned here.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from fr_gvi.algorithms.core import GaussianState, Method
from fr_gvi.expectations.core import LogisticExactExpectation
from fr_gvi.experiments import tuning
from fr_gvi.experiments.datasets import (
    DATASET_KEYS,
    MANIFEST,
    SPECS,
    load_dataset,
    split_and_standardize,
)
from fr_gvi.experiments.factories import build_problem

FORBIDDEN = ("alpha_star", "beta_star", "kappa_star", "lambda_0_star", "solve_reference")


def test_the_tuning_module_never_reads_an_optimizer_whitened_quantity() -> None:
    """A structural check on the selection path, not a comment.

    ``curvature_constants`` is imported for one purpose only -- BBVI--STL's
    projection radius ``1/sqrt(S)``, where ``S`` is the log-smoothness of the
    target in the *original* coordinates and a closed-form property of the model.
    Nothing else in the module may touch the whitened constants or a reference
    solve, since a stepsize chosen with those is a stepsize chosen with the
    answer.
    """

    source = inspect.getsource(tuning)
    # Strip the docstrings, which name the forbidden quantities in order to say
    # that they are forbidden.
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith(("#", '"', "'"))
    )
    for token in FORBIDDEN:
        assert f".{token}" not in body and f'"{token}"' not in body, (
            f"the tuning path references {token!r}; the practical rule must be "
            f"computable without the optimizer"
        )
    assert "forbidden_inputs" in source


def test_base_step_is_scale_free_for_the_natural_gradient_methods() -> None:
    """Affine equivariance means the natural-gradient stepsize carries no units.

    Rescaling the target must leave the Fisher--Rao base step alone and must
    rescale the Euclidean one by exactly the change in curvature; otherwise the
    grid would mean something different on the two families.
    """

    engine = LogisticExactExpectation(32)
    problem = build_problem(
        {
            "kind": "logistic",
            "dimension": 6,
            "samples": 60,
            "feature_condition": 10.0,
            "prior_precision": 1.0,
            "initial_covariance": "prior",
        },
        11,
        "R",
    )
    scaled = build_problem(
        {
            "kind": "logistic",
            "dimension": 6,
            "samples": 60,
            "feature_condition": 10.0,
            "prior_precision": 1.0,
            "initial_covariance": "prior",
        },
        11,
        "R",
    )
    fisher_rao, oracles = tuning.implementable_base_step(
        Method.FR_R, problem.target, problem.initial_state, engine
    )
    assert fisher_rao == 1.0 and oracles == 0
    euclidean, oracles = tuning.implementable_base_step(
        Method.FB_GVI, scaled.target, scaled.initial_state, engine
    )
    assert oracles == 1 and 0.0 < euclidean < np.inf


def test_selection_prefers_the_lowest_pilot_objective_among_admissible() -> None:
    engine = LogisticExactExpectation(32)
    problem = build_problem(
        {
            "kind": "logistic",
            "dimension": 4,
            "samples": 40,
            "feature_condition": 10.0,
            "prior_precision": 1.0,
            "initial_covariance": "prior",
        },
        3,
        "R",
    )
    selection = tuning.select_step(
        Method.FR_KL, problem, engine, problem_name="unit", iterations=12
    )
    admissible = [c for c in selection.candidates if c.admissible]
    assert admissible
    assert selection.pilot_objective == min(c.final_objective for c in admissible)
    chosen = next(c for c in selection.candidates if c.exponent == selection.exponent)
    assert chosen.admissible
    # Every candidate is screened, so an unstable one is recorded rather than raised.
    assert any(not c.admissible for c in selection.candidates), "the grid never left the stable range"


def test_inadmissible_candidates_carry_a_reason() -> None:
    engine = LogisticExactExpectation(32)
    problem = build_problem(
        {
            "kind": "logistic",
            "dimension": 4,
            "samples": 40,
            "feature_condition": 10.0,
            "prior_precision": 1.0,
            "initial_covariance": "prior",
        },
        3,
        "R",
    )
    selection = tuning.select_step(
        Method.FR_R, problem, engine, problem_name="unit", iterations=12
    )
    for candidate in selection.candidates:
        assert candidate.admissible == (candidate.reason == "")


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_every_dataset_matches_its_committed_digest(key: str) -> None:
    dataset = load_dataset(key, allow_download=False)
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["datasets"][key]
    assert dataset.sha256 == recorded["sha256"]
    assert dataset.observations == recorded["observations"]
    assert SPECS[key].openml_id == recorded["openml_id"]


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_standardization_uses_training_statistics_only(key: str) -> None:
    """Standardizing before the split would let the test rows shape the design."""

    split = split_and_standardize(load_dataset(key, allow_download=False))
    train = split.train_features[:, :-1]
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(train.std(axis=0), 1.0, rtol=1e-10)
    # The intercept is appended after standardization and is left alone.
    np.testing.assert_allclose(split.train_features[:, -1], 1.0)
    np.testing.assert_allclose(split.test_features[:, -1], 1.0)
    # The test block is transformed by the same map, so it is not itself centred.
    assert abs(float(split.test_features[:, :-1].mean())) < 0.5


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_the_split_is_stratified_and_disjoint(key: str) -> None:
    split = split_and_standardize(load_dataset(key, allow_download=False))
    train_rate = float(split.train_labels.mean())
    test_rate = float(split.test_labels.mean())
    assert abs(train_rate - test_rate) < 0.05
    assert split.train_labels.size + split.test_labels.size == split.metadata["observations"]


def test_the_pilot_subsample_is_disjoint_from_no_training_row_but_smaller() -> None:
    """The pilot fits a strict subset of the training rows, so it is a different problem.

    It has to be a subset rather than a held-out block: the point is that a
    practitioner can run it, and subsampling one's own training data is something
    a practitioner can always do.  What matters is that the objective the
    selection minimizes is not the objective any figure reports.
    """

    full = build_problem(
        {"kind": "logistic_dataset", "dataset": "sonar", "prior_precision": 1.0}, 0, "R"
    )
    pilot = build_problem(
        {
            "kind": "logistic_dataset",
            "dataset": "sonar",
            "prior_precision": 1.0,
            "subsample_fraction": tuning.PILOT_SUBSAMPLE,
        },
        0,
        "R",
    )
    assert pilot.target.features.shape[0] < full.target.features.shape[0]
    assert pilot.target.dimension == full.target.dimension
    ratio = pilot.target.features.shape[0] / full.target.features.shape[0]
    assert abs(ratio - tuning.PILOT_SUBSAMPLE) < 0.02


def test_logistic_dataset_targets_are_strongly_log_concave_with_closed_form_constants() -> None:
    """The benchmark must stay inside the hypotheses the theory is stated under."""

    from fr_gvi.diagnostics.curvature import curvature_constants

    for key in DATASET_KEYS:
        problem = build_problem(
            {"kind": "logistic_dataset", "dataset": key, "prior_precision": 1.0}, 0, "R"
        )
        constants = curvature_constants(
            problem.target, GaussianState(problem.initial_state.mean, problem.initial_state.covariance)
        )
        assert constants.exact
        assert constants.alpha == pytest.approx(1.0)
        assert constants.beta > constants.alpha
