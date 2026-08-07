"""The manuscript protocol is preregistered, so its shape is asserted in tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fr_gvi.experiments import manuscript
from fr_gvi.experiments.factories import build_problem
from fr_gvi.experiments.reference import solve_reference
from fr_gvi.plotting.style import load_experiment
from fr_gvi.targets.core import ShiftedLogCoshTarget

ALLOWED = {*manuscript.ITERATIVE, "Laplace"}


def _all_configs() -> list[dict]:
    configs: list[dict] = []
    for name in manuscript.GROUPS:
        directory = manuscript.CONFIG_ROOT / name
        if not directory.exists():
            pytest.skip(f"{directory} not generated")
        configs.extend(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        )
    return configs


def test_campaign_declares_the_preregistered_trajectory_count() -> None:
    configs = _all_configs()
    assert manuscript.trajectory_count(configs) == 131


def test_only_protocol_methods_appear() -> None:
    for config in _all_configs():
        for specification in config["methods"]:
            assert specification["name"] in ALLOWED


def test_configs_use_the_frozen_multipliers() -> None:
    """Every practical step in the campaign is the one the pilot recorded."""

    if not manuscript.SELECTED_STEPS.exists():
        pytest.skip("selected_steps.json not generated")
    frozen = manuscript.selected_multipliers()
    seen = False
    for config in _all_configs():
        for specification in config["methods"]:
            if "step_scale" not in specification:
                continue
            seen = True
            name = specification["name"]
            assert specification["normalized_step_size"] == pytest.approx(frozen[name])
            assert specification["step_size"] == pytest.approx(
                frozen[name] * specification["step_scale"]
            )
    assert seen, "no frozen practical steps found in the campaign"


def test_theorem_panels_stay_inside_their_certified_window() -> None:
    """Panels that verify a theorem must not exceed the step it admits."""

    for config in _all_configs():
        for specification in config["methods"]:
            if "step_scale" in specification or "certified_step_size" not in specification:
                continue
            assert specification["step_size"] <= specification["certified_step_size"] * (
                1.0 + 1e-12
            )


@pytest.mark.parametrize(("dimension", "condition", "rho"), [(6, 10.0, 1.0), (12, 100.0, 0.1)])
def test_whitened_logcosh_has_identity_optimizer(
    dimension: int, condition: float, rho: float
) -> None:
    """The whitened construction puts C_star at the identity, so C_0 = I is in the band."""

    problem = build_problem(
        manuscript._logcosh_target(dimension, condition, rho), 4242, "D"
    )
    assert isinstance(problem.target, ShiftedLogCoshTarget)
    reference = solve_reference(
        problem.target, problem.initial_state, points=1024, seed=11
    )
    deviation = np.linalg.norm(reference.state.covariance - np.eye(dimension))
    assert deviation < 1e-9
    # And the initialization is the displaced mean with an identity covariance.
    assert problem.initial_state.mean[0] == pytest.approx(2.0)
    assert np.allclose(problem.initial_state.mean[1:], 0.0)
    assert np.allclose(problem.initial_state.covariance, np.eye(dimension))


def test_logcosh_reference_residual_is_uniformly_small() -> None:
    """The panelled marginal rule must hold on wide marginals too.

    The small-curvature coordinates of the ill-conditioned cells have marginals
    several units wide around a unit-width sech^2 feature; a rule whose nodes scale
    with the marginal misses it and degrades the reference by orders of magnitude.
    """

    for dimension, condition, rho in ((10, 100.0, 0.1), (10, 1.0, 1.0)):
        problem = build_problem(
            manuscript._logcosh_target(dimension, condition, rho), 4242, "D"
        )
        reference = solve_reference(
            problem.target, problem.initial_state, points=2048, seed=7
        )
        assert np.sqrt(reference.fisher_rao_residual_squared) < 1e-11


def test_affine_grid_stays_inside_double_precision() -> None:
    """No cell may put the transported covariance past the float64 limit.

    The change of variables carries the covariance by ``C -> S C S^T``, whose
    condition number is ``cond(S)^2``.  A cell with ``cond(S)^2 >= 1/eps`` cannot
    be measured at all, so the generator refuses to emit one.
    """

    limit = 1.0 / np.finfo(np.float64).eps
    for config in manuscript.figure1_affine_equivariance():
        condition = float(config["grid"]["transform_condition"])
        assert condition**2 < limit


def test_logistic_uses_exact_quadrature_not_a_sampling_design() -> None:
    """The logistic cells must carry quadrature orders, never point counts."""

    if not (manuscript.CONFIG_ROOT / "figure3_logistic").exists():
        pytest.skip("figure3_logistic not generated")
    for config in _all_configs():
        if config["experiment"] != "L":
            continue
        assert "quadrature_order" in config
        assert config["evaluation_quadrature_order"] > config["quadrature_order"]
        for key in ("update_points", "evaluation_points", "reference_points"):
            assert key not in config


def test_logistic_expectations_match_an_independent_rule() -> None:
    """The exact engine must agree with a much finer rule far below any reported gap."""

    from fr_gvi.expectations.core import LogisticExactExpectation

    problem = build_problem(
        {
            "kind": "logistic",
            "dimension": 12,
            "samples": 60,
            "test_samples": 20,
            "feature_condition": 100.0,
            "prior_precision": 1.0,
            "initial_covariance": "prior",
        },
        4242,
        "L",
    )
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((12, 12))
    mean = 0.3 * rng.standard_normal(12)
    covariance = 0.2 * (factor @ factor.T) / 12 + 0.05 * np.eye(12)
    coarse = LogisticExactExpectation(48).evaluate(problem.target, mean, covariance)
    fine = LogisticExactExpectation(160).evaluate(problem.target, mean, covariance)
    assert abs(coarse.value - fine.value) < 1e-9
    assert np.linalg.norm(coarse.grad - fine.grad) < 1e-9
    assert np.linalg.norm(coarse.hessian - fine.hessian) < 1e-8


def test_certified_gap_bound_dominates_the_measured_gap() -> None:
    """The proved gradient-domination bound must hold wherever the gap is meaningful."""

    from fr_gvi.diagnostics.core import certified_gap_bound

    frame = load_experiment("D", "manuscript")
    if not frame.empty:
        frame = frame[~frame["job_id"].astype(str).str.startswith("pilot")].copy()
    if frame.empty or "certified_gap" not in frame.columns:
        pytest.skip("no manuscript campaign results yet")
    frame["initial"] = frame.groupby(["job_id", "method"])["objective_gap"].transform("first")
    meaningful = frame[frame["objective_gap"] > 1e-10 * frame["initial"]]
    assert len(meaningful) > 100
    for row in meaningful.itertuples():
        bound = certified_gap_bound(
            fisher_rao_squared=row.fisher_rao_residual_squared,
            bures_wasserstein_squared=row.bures_wasserstein_residual_squared,
            alpha_star=row.alpha_star,
            covariance_min_eigenvalue=row.covariance_min_eigenvalue,
        )
        assert bound >= row.objective_gap


def test_every_protocol_amendment_is_decided_and_matches_the_code() -> None:
    """A deviation from the brief must be recorded, decided, and still accurate.

    The audit owns the full check; this keeps it in the fast suite so an
    implementation change that silently departs from an approved amendment fails
    before anyone runs a campaign.
    """

    from fr_gvi.experiments.manuscript_audit import check_amendments

    errors, _, summary = check_amendments()
    assert not errors, errors
    assert set(summary["decisions"]) == {"A", "B", "C", "D"}
    assert set(summary["decisions"].values()) == {"approved"}
