"""The reduced, preregistered configuration set behind the manuscript figures.

The exploratory campaign in ``fr_gvi.experiments.grids`` sweeps stepsizes and
replicates; it exists to locate stable operating points and to check that the
implementation behaves as the theory says.  The manuscript itself needs far less:
three deterministic experiment groups, three figures, and roughly one hundred and
thirty trajectories.  This module emits exactly those configs.

The protocol is fixed before the final runs:

* only ``FR--R``, ``FR--KL`` and ``FB--GVI`` iterate, with ``Laplace`` as a
  non-iterative reference on the logistic problem;
* stepsizes are either the certified ones admitted by the theorems, used wherever
  a panel verifies a theorem, or the frozen practical multipliers selected once on
  a designated pilot target and recorded in ``configs/manuscript/selected_steps.json``;
* the non-Gaussian targets are built in optimizer-whitened coordinates, so a
  global comparison starting from ``C_0 = I`` is not confounded by a covariance
  burn-in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fr_gvi.diagnostics.curvature import (
    certified_step_sizes,
    curvature_constants,
    whitened_initialization,
)
from fr_gvi.experiments.factories import build_problem
from fr_gvi.experiments.reference import solve_reference

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "configs" / "manuscript"
SELECTED_STEPS = CONFIG_ROOT / "selected_steps.json"

TIER = "manuscript"
ITERATIVE = ("FR--R", "FR--KL", "FB--GVI")

# Multipliers offered to the pilot.
PILOT_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

PILOT_STEP_SCALE = "inverse_beta_star_times_initial_ceiling"


def practical_step_scale(constants: dict[str, Any]) -> float:
    """The scale a frozen practical multiplier is applied to.

    What has to stay of order one for any of the three schemes to be stable is
    ``h beta_star lambda_max(C_star^{-1/2} C_n C_star^{-1/2})``: the Riemannian
    retraction exponentiates ``h (I - R H R)``, and ``R H R`` is bounded by
    ``beta_star`` times the whitened covariance.  The whitened covariance is largest
    at the initialization whenever ``C_0`` overshoots ``C_star``, so the transferable
    scale is

        1 / (beta_star * max(lambda_{0,star}^max, 1)).

    Two other candidates fail, each on one family.  A multiple of the certified step
    ``1/(2 beta_star lambda_max_star)`` carries the worst-case growth allowance
    ``lambda_max_star >= 1/alpha_star``, which is never attained on the whitened
    log-cosh cells; there its conservatism grows like ``kappa_star`` and a multiplier
    chosen at ``kappa_star = 11`` diverges at ``kappa_star = 2``.  A multiple of
    ``1/beta_star`` alone ignores the initialization and diverges on the logistic
    posteriors, where ``C_0 = lambda_prior^{-1} I`` overshoots ``C_star`` by two
    orders of magnitude.  The scale above is the certified step with the
    non-attained growth allowance removed and the initialization kept.
    """

    ceiling = max(float(constants["lambda_0_star_max"]), 1.0)
    return 1.0 / (float(constants["beta_star"]) * ceiling)


# Two pilot cells, one per target family, because a single cell cannot calibrate
# both regimes: the whitened log-cosh cell starts with its covariance already in the
# band, the logistic cell starts two orders of magnitude above it.  Each has the
# nominal parameters of a cell that appears in the figures but a different master
# seed, hence a different target instance, so no trajectory the stepsizes were
# selected on is shown in any panel.  The frozen multiplier is the smallest of the
# per-pilot choices, so it is admissible on both.
PILOT_SEED = 20260401
PILOT_CELL = {"dimension": 10, "condition": 10.0, "rho": 1.0}
PILOT_LOGISTIC_SEED = 20260402
PILOT_LOGISTIC_CELL = {"dimension": 50, "feature_condition": 100.0}


def _seed_for(master_seed: int, stream: int, repeat: int = 0) -> int:
    sequence = np.random.SeedSequence(master_seed, spawn_key=(stream, repeat))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def cell_constants(
    target_config: dict[str, Any],
    master_seed: int,
    experiment: str,
    *,
    reference_points: int = 2048,
) -> dict[str, Any]:
    """Curvature constants and certified stepsizes of one cell.

    The problem is instantiated with the same deterministic seeds the runner will
    use, so the constants recorded in the config are the ones the run sees.
    """

    problem = build_problem(target_config, _seed_for(master_seed, 0, 0), experiment)
    reference = solve_reference(
        problem.target,
        problem.initial_state,
        points=reference_points,
        seed=_seed_for(master_seed, 3, 0),
        certify=False,
    )
    constants = curvature_constants(problem.target, reference.state)
    whitened = whitened_initialization(
        problem.initial_state, reference.state, constants.alpha_star
    )
    return {
        **constants.to_dict(),
        **whitened.to_dict(),
        "certified_step_sizes": certified_step_sizes(constants, whitened),
    }


def _logcosh_target(dimension: int, condition: float, rho: float) -> dict[str, Any]:
    """One cell of the optimizer-whitened log-cosh family.

    ``m_0 = 2 e_1`` and ``C_0 = I``: the covariance already sits in the band, so
    what the trajectory measures is the non-Gaussian localization alone.
    """

    return {
        "kind": "logcosh",
        "dimension": dimension,
        "condition": condition,
        "rho": rho,
        "whiten_optimizer": True,
        "shift": 0.0,
        "initial_mean_axis_scale": 2.0,
        "initial_covariance_scale": 1.0,
    }


# ---------------------------------------------------------------------------
# Stepsize pilot
# ---------------------------------------------------------------------------


def _pilot_methods(constants: dict[str, Any], name: str) -> list[dict[str, Any]]:
    certified = constants["certified_step_sizes"]
    scale = practical_step_scale(constants)
    return [
        {
            "name": name,
            "step_size": float(scale * multiplier),
            "normalized_step_size": float(multiplier),
            "step_scale": scale,
            "certified_step_size": float(certified[name]),
            "tag": f"x{multiplier:g}".replace(".", "p"),
        }
        for multiplier in PILOT_MULTIPLIERS
    ]


def pilot_configs() -> list[dict[str, Any]]:
    """The designated cells on which the practical stepsizes are chosen.

    One config per (family, method) so the sweep parallelizes over workers; the
    reference of a cell is solved once per config.
    """

    configs: list[dict[str, Any]] = []

    target = _logcosh_target(**dict(PILOT_CELL))
    constants = cell_constants(target, PILOT_SEED, "D")
    for name in ITERATIVE:
        slug = name.lower().replace("--", "-")
        configs.append(
            {
                "id": f"pilot_logcosh_d10_k10_rho1_{slug}",
                "experiment": "D",
                "tier": TIER,
                "master_seed": PILOT_SEED,
                "grid": {**PILOT_CELL, "role": "stepsize_pilot", "family": "logcosh"},
                "target": target,
                "iterations": 200,
                "curvature": constants,
                "certified_step_sizes": constants["certified_step_sizes"],
                "methods": _pilot_methods(constants, name),
            }
        )

    logistic_target = {
        "kind": "logistic",
        "dimension": PILOT_LOGISTIC_CELL["dimension"],
        "samples": LOGISTIC_TRAIN,
        "test_samples": LOGISTIC_TEST,
        "feature_condition": PILOT_LOGISTIC_CELL["feature_condition"],
        "prior_precision": LOGISTIC_PRIOR,
        "initial_covariance": "prior",
    }
    logistic_constants = cell_constants(logistic_target, PILOT_LOGISTIC_SEED, "L")
    for name in ITERATIVE:
        slug = name.lower().replace("--", "-")
        configs.append(
            {
                "id": f"pilot_logistic_d50_kX1e2_{slug}",
                "experiment": "L",
                "tier": TIER,
                "master_seed": PILOT_LOGISTIC_SEED,
                "grid": {
                    **PILOT_LOGISTIC_CELL,
                    "role": "stepsize_pilot",
                    "family": "logistic",
                },
                "target": logistic_target,
                "quadrature_order": LOGISTIC_UPDATE_ORDER,
                "evaluation_quadrature_order": LOGISTIC_EVALUATION_ORDER,
                "iterations": 150,
                "record_every": 1,
                "curvature": logistic_constants,
                "certified_step_sizes": logistic_constants["certified_step_sizes"],
                "methods": _pilot_methods(logistic_constants, name),
            }
        )
    return configs


def selected_multipliers() -> dict[str, float]:
    if not SELECTED_STEPS.exists():
        raise FileNotFoundError(
            f"{SELECTED_STEPS} is missing; run the stepsize pilot first "
            "(make manuscript-pilot)"
        )
    payload = json.loads(SELECTED_STEPS.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in payload["multipliers"].items()}


def _frozen_methods(
    constants: dict[str, Any], multipliers: dict[str, float]
) -> list[dict[str, Any]]:
    """Each method at its frozen multiple of the cell's practical step scale."""

    certified = constants["certified_step_sizes"]
    scale = practical_step_scale(constants)
    return [
        {
            "name": name,
            "step_size": float(scale * multipliers[name]),
            "normalized_step_size": float(multipliers[name]),
            "step_scale": scale,
            "certified_step_size": float(certified[name]),
        }
        for name in ITERATIVE
    ]


# ---------------------------------------------------------------------------
# Figure 1: Gaussian structure and affine invariance
# ---------------------------------------------------------------------------


def figure1_gaussian_burnin() -> list[dict[str, Any]]:
    """Fisher--Rao covariance entry time against ``log(1/lambda_0)``.

    The target is the standard Gaussian in ``d = 20``, so ``C_star = I``,
    ``beta_star = 1``, and the manuscript's whitened entry criterion
    ``lambda_min(C_star^{-1/2} C_n C_star^{-1/2}) >= 1/(2 beta_star)`` is exactly
    ``lambda_min(C_n) >= 1/2``.  The step ``0.1`` lies inside both certified
    windows (``1/2`` for FR--R and ``1`` for FR--KL) and is common to the two
    methods and to every initialization, so ``N_ent h`` is directly comparable.
    """

    dimension = 20
    step = 0.1
    configs: list[dict[str, Any]] = []
    for exponent in (2, 4, 6, 8):
        scale = 10.0 ** (-exponent)
        master_seed = 20260401 + exponent
        target = {
            "kind": "gaussian",
            "dimension": dimension,
            "condition": 1.0,
            "rotation": False,
            "mean_scale": 0.0,
            "initial_covariance_scale": scale,
        }
        certified = cell_constants(target, master_seed, "A")["certified_step_sizes"]
        iterations = int(np.ceil((np.log(1.0 / scale) + 6.0) / step))
        configs.append(
            {
                "id": f"F1burnin_d{dimension}_lam1e-{exponent}",
                "experiment": "A",
                "tier": TIER,
                "master_seed": master_seed,
                "grid": {"dimension": dimension, "lambda0": scale, "figure": "1a"},
                "target": target,
                "iterations": iterations,
                "certified_step_sizes": certified,
                "methods": [
                    {
                        "name": name,
                        "step_size": step,
                        "normalized_step_size": step,
                        "certified_step_size": float(certified[name]),
                    }
                    for name in ("FR--R", "FR--KL")
                ],
            }
        )
    return configs


def figure1_affine_equivariance() -> list[dict[str, Any]]:
    """Equivariance under non-orthogonal changes of variables ``x -> S x + b``.

    The grid stops at ``cond(S) = 10^6``.  The change of variables carries the
    covariance by the congruence ``C -> S C S^T``, whose condition number is
    ``cond(S)^2``, so at ``cond(S) = 10^8`` the transported state has condition
    number ``1/eps`` and is not representable in double precision: the iteration
    breaks down there and no method can be measured.  Plotting a broken trajectory
    beside complete ones would not be a comparison, so the panel stops one decade
    short and marks the floating-point boundary instead.
    """

    dimension = 10
    configs: list[dict[str, Any]] = []
    for exponent in (0, 2, 4, 6):
        transform_condition = 10.0**exponent
        master_seed = 20260420 + exponent
        target = {
            "kind": "gaussian",
            "dimension": dimension,
            "transform_condition": transform_condition,
        }
        certified = cell_constants(target, master_seed, "B")["certified_step_sizes"]
        config = {
                "id": f"F1affine_d{dimension}_S1e{exponent}",
                "experiment": "B",
                "tier": TIER,
                "master_seed": master_seed,
                "grid": {
                    "dimension": dimension,
                    "transform_condition": transform_condition,
                    "figure": "1b",
                },
                "target": target,
                "iterations": 60,
                "certified_step_sizes": certified,
                "methods": [
                    {
                        "name": name,
                        "step_size": float(0.5 * certified[name]),
                        "normalized_step_size": 0.5,
                        "certified_step_size": float(certified[name]),
                    }
                    for name in ITERATIVE
                ],
            }
        if transform_condition**2 >= 1.0 / np.finfo(np.float64).eps:
            raise ValueError(
                f"cond(S) = {transform_condition:g} puts the transported covariance at "
                "the limit of double precision; every cell in the panel must complete "
                "cleanly, so the grid must stop below this point"
            )
        configs.append(config)
    return configs


def figure1_anisotropic_gaussian() -> list[dict[str, Any]]:
    """The anisotropic Gaussian with ``kappa = 1e9`` but ``kappa_star = 1``."""

    master_seed = 20260430
    target = {
        "kind": "gaussian",
        "dimension": 10,
        "diao_spectrum": True,
        "rotation": True,
        "mean_distribution": "uniform",
        "initial_covariance_scale": 1.0,
    }
    constants = cell_constants(target, master_seed, "C")
    certified = constants["certified_step_sizes"]
    return [
        {
            "id": "F1aniso_d10_kappa1e9",
            "experiment": "C",
            "tier": TIER,
            "master_seed": master_seed,
            "grid": {"dimension": 10, "condition": 1.0e9, "figure": "1c"},
            "target": target,
            "iterations": 500,
            "curvature": constants,
            "certified_step_sizes": certified,
            "methods": [
                {
                    "name": name,
                    "step_size": float(certified[name]),
                    "normalized_step_size": 1.0,
                    "certified_step_size": float(certified[name]),
                }
                for name in ITERATIVE
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Figure 2: global-to-local deterministic convergence
# ---------------------------------------------------------------------------

GLOBAL_CELLS = [
    (dimension, condition, rho)
    for dimension in (10, 50)
    for condition in (1.0, 10.0, 100.0)
    for rho in (0.1, 1.0)
]

REPRESENTATIVE_CELLS = ((10, 1.0, 1.0), (10, 10.0, 1.0), (50, 100.0, 1.0))

GLOBAL_BUDGET = 400
GLOBAL_TOLERANCE = 1.0e-6

LOCAL_CELLS = ((10, 10.0, 0.1), (10, 10.0, 1.0))
LOCAL_RADII = (1.0e-1, 5.0e-2, 1.0e-2)


def figure2_logcosh_global() -> list[dict[str, Any]]:
    """One fixed instance per cell of the reduced log-cosh grid."""

    multipliers = selected_multipliers()
    configs: list[dict[str, Any]] = []
    for dimension, condition, rho in GLOBAL_CELLS:
        master_seed = 20260500 + dimension * 100 + int(np.log10(condition)) * 10 + int(rho * 10)
        target = _logcosh_target(dimension, condition, rho)
        constants = cell_constants(target, master_seed, "D")
        certified = constants["certified_step_sizes"]
        configs.append(
            {
                "id": f"F2global_d{dimension}_k{condition:g}_rho{rho:g}",
                "experiment": "D",
                "tier": TIER,
                "master_seed": master_seed,
                "grid": {
                    "dimension": dimension,
                    "condition": condition,
                    "rho": rho,
                    "representative": (dimension, condition, rho) in REPRESENTATIVE_CELLS,
                    "figure": "2a-2c",
                },
                "target": target,
                # A common budget for every cell, so panel (c) reports
                # iterations-to-tolerance at a genuinely fixed budget.  It is set
                # by the slowest cell: the certified Fisher--Rao step is
                # 1/(2 kappa_star), and kappa_star reaches 101 here.
                "iterations": GLOBAL_BUDGET,
                "record_every": 1,
                "curvature": constants,
                "certified_step_sizes": certified,
                "methods": _frozen_methods(constants, multipliers),
            }
        )
    return configs


def figure2_logcosh_local() -> list[dict[str, Any]]:
    """Local spectral rate from the slowest eigenmode of ``L_star``.

    Each cell is initialized at ``a_star + r v_min`` and run at its own certified
    step, because the panel compares the measured contraction with the prediction
    ``1 - h gamma_star`` that the theorem states at admissible steps.
    """

    configs: list[dict[str, Any]] = []
    for dimension, condition, rho in LOCAL_CELLS:
        master_seed = 20260600 + dimension * 10 + int(rho * 10)
        target = _logcosh_target(dimension, condition, rho)
        constants = cell_constants(target, master_seed, "G")
        certified = constants["certified_step_sizes"]
        for radius in LOCAL_RADII:
            configs.append(
                {
                    "id": (
                        f"F2local_d{dimension}_k{condition:g}_rho{rho:g}"
                        f"_r{radius:g}".replace(".", "p")
                    ),
                    "experiment": "G",
                    "tier": TIER,
                    "master_seed": master_seed,
                    "grid": {
                        "dimension": dimension,
                        "condition": condition,
                        "rho": rho,
                        "radius": radius,
                        "figure": "2d",
                    },
                    "target": target,
                    "local_operator_points": 8192,
                    "local_initialization": {"radius": radius},
                    "iterations": 2000,
                    "record_every": 1,
                    "curvature": constants,
                    "certified_step_sizes": certified,
                    "methods": [
                        {
                            "name": name,
                            "step_size": float(certified[name]),
                            "normalized_step_size": 1.0,
                            "certified_step_size": float(certified[name]),
                        }
                        for name in ("FR--R", "FR--KL")
                    ],
                }
            )
    return configs


# ---------------------------------------------------------------------------
# Figure 3: deterministic Bayesian logistic regression
# ---------------------------------------------------------------------------

LOGISTIC_DIMENSION = 50
LOGISTIC_TRAIN = 500
LOGISTIC_TEST = 5000
LOGISTIC_PRIOR = 1.0
LOGISTIC_DATASETS = 5
LOGISTIC_BUDGET = 2000
# The logistic expectations are computed exactly, by one panelled Gauss-Legendre
# rule per linear predictor, so there is no sampling design.  The updates use
# order 48 and the objective evaluation and the reference solve use order 96: the
# evaluation rule is strictly finer than the update rule, and the two agree to
# about 1e-11, so the "independent finer evaluation" requirement is met with a
# transfer error twelve orders below the gaps the figures resolve rather than the
# 1e-3 a 4096-point quasi-Monte-Carlo design would leave.
LOGISTIC_UPDATE_ORDER = 48
LOGISTIC_EVALUATION_ORDER = 96


def figure3_logistic() -> list[dict[str, Any]]:
    """Synthetic Bayesian logistic regression over three feature conditionings.

    One config per (conditioning, dataset, method).  The campaign parallelizes over
    config files and runs the methods inside one file sequentially, so emitting a
    single config per conditioning would run sixty trajectories twenty deep on
    whatever machine is available; split this way they are all independent.  Each
    dataset gets its own master seed, hence its own regression problem, and the
    five replicates are five problems rather than five runs of one.
    """

    multipliers = selected_multipliers()
    configs: list[dict[str, Any]] = []
    for exponent in (0, 2, 4):
        feature_condition = 10.0**exponent
        target = {
            "kind": "logistic",
            "dimension": LOGISTIC_DIMENSION,
            "samples": LOGISTIC_TRAIN,
            "test_samples": LOGISTIC_TEST,
            "feature_condition": feature_condition,
            "prior_precision": LOGISTIC_PRIOR,
            "initial_covariance": "prior",
        }
        for dataset in range(LOGISTIC_DATASETS):
            master_seed = 20260700 + 10 * exponent + dataset
            constants = cell_constants(target, master_seed, "L")
            certified = constants["certified_step_sizes"]
            methods = [
                *_frozen_methods(constants, multipliers),
                {"name": "Laplace"},
            ]
            for specification in methods:
                slug = str(specification["name"]).lower().replace("--", "-")
                configs.append(
                    {
                        "id": (
                            f"F3logistic_d{LOGISTIC_DIMENSION}_kX1e{exponent}"
                            f"_s{dataset}_{slug}"
                        ),
                        "experiment": "L",
                        "tier": TIER,
                        "master_seed": master_seed,
                        "grid": {
                            "dimension": LOGISTIC_DIMENSION,
                            "feature_condition": feature_condition,
                            "prior_precision": LOGISTIC_PRIOR,
                            "dataset": dataset,
                            "figure": "3",
                        },
                        "target": target,
                        "quadrature_order": LOGISTIC_UPDATE_ORDER,
                        "evaluation_quadrature_order": LOGISTIC_EVALUATION_ORDER,
                        # kappa_star is 140 to 360 on these posteriors and the
                        # whitened initial covariance is roughly 100 times too
                        # large, so the frozen practical step is near 1e-2.  The
                        # horizon is set by the slowest method on the slowest cell:
                        # the Fisher-Rao schemes need about 1700 iterations at
                        # kappa_X = 1, where FB--GVI converges in 200.  Stopping
                        # earlier would show the Fisher-Rao curves mid-descent and
                        # read as a floor rather than as a slower rate.
                        "iterations": LOGISTIC_BUDGET,
                        "record_every": 10,
                        "curvature": constants,
                        "certified_step_sizes": certified,
                        "methods": [specification],
                    }
                )
    return configs


GROUPS: dict[str, Any] = {
    "figure1_gaussian_burnin": figure1_gaussian_burnin,
    "figure1_affine_equivariance": figure1_affine_equivariance,
    "figure1_anisotropic_gaussian": figure1_anisotropic_gaussian,
    "figure2_logcosh_global": figure2_logcosh_global,
    "figure2_logcosh_local": figure2_logcosh_local,
    "figure3_logistic": figure3_logistic,
}


def trajectory_count(configs: Iterable[dict[str, Any]]) -> int:
    total = 0
    for config in configs:
        repeats = int(config.get("seeds", 1))
        for specification in config.get("methods", []):
            total += int(specification.get("seeds", repeats))
    return total


def write_group(name: str, destination: Path) -> tuple[int, int]:
    directory = destination / name
    directory.mkdir(parents=True, exist_ok=True)
    configs = GROUPS[name]()
    for config in configs:
        path = directory / f"{config['id']}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(configs), trajectory_count(configs)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=CONFIG_ROOT)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument(
        "--pilot", action="store_true", help="write only the stepsize pilot config"
    )
    args = parser.parse_args(arguments)

    if args.pilot:
        directory = args.destination / "pilot"
        directory.mkdir(parents=True, exist_ok=True)
        configs = pilot_configs()
        for config in configs:
            (directory / f"{config['id']}.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(f"pilot: {len(configs)} configs, {trajectory_count(configs)} trajectories")
        return 0

    total_configs = 0
    total_runs = 0
    for name in args.groups or list(GROUPS):
        count, runs = write_group(name, args.destination)
        total_configs += count
        total_runs += runs
        print(f"{name}: {count} configs, {runs} trajectories")
    print(f"total: {total_configs} configs, {total_runs} trajectories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
