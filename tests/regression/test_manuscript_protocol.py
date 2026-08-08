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

ALLOWED = {
    *manuscript.ITERATIVE,
    "Laplace",
    "FR--R--STL",
    "FR--KL--STL",
    "S--FB--GVI",
    *manuscript.DETERMINISTIC_BENCHMARK,
    *manuscript.STOCHASTIC_BENCHMARK,
}

# The preregistered shape of the campaign, group by group.  Asserting one total
# would hide a group silently losing its cells to another gaining them.
EXPECTED_TRAJECTORIES = {
    "figure1_gaussian_burnin": 8,
    "figure1_affine_equivariance": 12,
    "figure1_anisotropic_gaussian": 3,
    "figure2_logcosh_global": 36,
    "figure2_logcosh_local": 12,
    "figure3_logistic": 60,
    "figure3_stochastic_cancellation": 150,
    "figure3_stochastic_floor": 1050,
    "figure3_stochastic_decreasing": 120,
    "figure4_real_datasets": 150,
    "appendix_scaling": 100,
}


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
    assert set(EXPECTED_TRAJECTORIES) == set(manuscript.GROUPS)
    for name, expected in EXPECTED_TRAJECTORIES.items():
        directory = manuscript.CONFIG_ROOT / name
        if not directory.exists():
            pytest.skip(f"{directory} not generated")
        configs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]
        assert manuscript.trajectory_count(configs) == expected, name
    assert manuscript.trajectory_count(_all_configs()) == sum(EXPECTED_TRAJECTORIES.values())


def test_every_specification_declares_which_stepsize_rule_it_used() -> None:
    """Three regimes coexist, so each must say which one it belongs to.

    Certified where a panel verifies a theorem, pilot-frozen on the earlier
    practical panels, and the implementable dyadic grid throughout the benchmark.
    A specification that declares nothing would be checked by no gate.
    """

    from fr_gvi.experiments.manuscript_audit import STEP_RULES

    for config in _all_configs():
        for specification in config["methods"]:
            assert specification.get("step_rule") in STEP_RULES, (
                config["id"],
                specification.get("name"),
            )


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
            if specification.get("step_rule") != "pilot_frozen":
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

    seen = False
    for config in _all_configs():
        for specification in config["methods"]:
            if specification.get("step_rule") != "certified":
                continue
            seen = True
            assert specification["step_size"] <= specification["certified_step_size"] * (
                1.0 + 1e-12
            )
    assert seen, "no certified-step panels found in the campaign"


def test_benchmark_steps_reproduce_from_the_recorded_selection() -> None:
    """Every implementable-grid step is its frozen multiplier times its base scale."""

    from fr_gvi.experiments.tuning import PRACTICAL_STEPS, load_selected

    if not PRACTICAL_STEPS.exists():
        pytest.skip("practical_steps.json not generated")
    selections = load_selected()
    seen = 0
    for config in _all_configs():
        for specification in config["methods"]:
            if specification.get("step_rule") != "implementable_grid":
                continue
            seen += 1
            assert specification["step_size"] == pytest.approx(
                specification["normalized_step_size"] * specification["step_scale"], rel=1e-12
            )
            # The multiplier must be one the tuner actually recorded, not a number
            # written into the generator by hand.
            recorded = {
                float(value["multiplier"])
                for key, value in selections.items()
                if key.endswith(f":{specification['name']}")
            }
            assert specification["normalized_step_size"] in recorded
    assert seen >= 50, seen


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
    assert set(summary["decisions"]) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(summary["decisions"].values()) == {"approved"}
