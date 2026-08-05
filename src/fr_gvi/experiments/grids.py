"""Programmatic generation of the full experiment grids.

Every grid cell is materialized as an ordinary campaign config, so the generated
files are inspectable, hashable and resumable exactly like hand-written ones.

Step sizes are emitted as explicit multiples of the certified scale admitted by
the theory for each method, which requires knowing the curvature constants of
the cell.  The generator therefore instantiates each problem with the same
deterministic seeds the runner will use, solves its reference, and records both
the actual step and its normalized multiplier.  This is what makes the
comparisons stepsize-fair: no two geometries are forced onto the same numerical
step, and each method is swept around its own certified scale.
"""

from __future__ import annotations

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
CONFIG_ROOT = ROOT / "configs"

# Multipliers of each method's certified stepsize used by the sweep experiments.
STEP_MULTIPLIERS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

STOCHASTIC_SEEDS = 30


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
    """Curvature constants and certified stepsizes for one grid cell."""

    problem = build_problem(target_config, _seed_for(master_seed, 0, 0), experiment)
    reference = solve_reference(
        problem.target,
        problem.initial_state,
        points=reference_points,
        seed=_seed_for(master_seed, 3, 0),
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


def _tag(multiplier: float) -> str:
    text = f"{multiplier:g}".replace(".", "p").replace("-", "m")
    return f"x{text}"


def swept_methods(
    names: Iterable[str],
    certified: dict[str, float],
    *,
    multipliers: Iterable[float] = STEP_MULTIPLIERS,
    seeds: int | None = None,
    extra: dict[str, Any] | None = None,
    per_method_extra: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
    for name in names:
        base = certified[name]
        for multiplier in multipliers:
            specification: dict[str, Any] = {
                "name": name,
                "step_size": float(base * multiplier),
                "normalized_step_size": float(multiplier),
                "certified_step_size": float(base),
                "tag": _tag(multiplier),
            }
            if seeds is not None:
                specification["seeds"] = int(seeds)
            if extra:
                specification.update(extra)
            if per_method_extra and name in per_method_extra:
                specification.update(per_method_extra[name])
            methods.append(specification)
    return methods


# --------------------------------------------------------------------------
# Experiment builders
# --------------------------------------------------------------------------


def experiment_a() -> list[dict[str, Any]]:
    """Covariance burn-in: N_cov h against log[1/(beta_star lambda_{0,star})]."""

    configs: list[dict[str, Any]] = []
    dimension = 20
    for condition in (10.0, 100.0, 1000.0):
        for exponent in (2, 4, 6, 8, 10, 12):
            scale = 10.0 ** (-exponent)
            master_seed = 20260901 + int(np.log10(condition)) * 100 + exponent
            target = {
                "kind": "gaussian",
                "dimension": dimension,
                "condition": condition,
                "spectrum_scale": "unit_min",
                "rotation": True,
                "mean_scale": 0.2,
                "initial_covariance_scale": scale,
            }
            certified = cell_constants(target, master_seed, "A")["certified_step_sizes"]
            # Fixed step of 0.1 keeps N_cov * h directly comparable across cells,
            # and 0.1 is inside every certified Fisher--Rao window here.
            step = 0.1
            iterations = int(np.ceil((np.log(1.0 / scale) + 8.0) / step))
            configs.append(
                {
                    "id": f"A_burnin_d{dimension}_k{condition:g}_lam1e-{exponent}",
                    "experiment": "A",
                    "tier": "full",
                    "master_seed": master_seed,
                    "grid": {"dimension": dimension, "condition": condition, "lambda0": scale},
                    "target": target,
                    "iterations": iterations,
                    "certified_step_sizes": certified,
                    "methods": [
                        {"name": "FR--R", "step_size": step, "normalized_step_size": step},
                        {"name": "FR--KL", "step_size": step, "normalized_step_size": step},
                    ],
                }
            )
    return configs


def experiment_b() -> list[dict[str, Any]]:
    """Affine equivariance across conditioning of the coordinate change."""

    configs: list[dict[str, Any]] = []
    for exponent in (0, 2, 4, 6, 8):
        condition = 10.0**exponent
        master_seed = 20260920 + exponent
        target = {"kind": "gaussian", "dimension": 5, "affine_condition": condition}
        certified = cell_constants(target, master_seed, "B")["certified_step_sizes"]
        configs.append(
            {
                "id": f"B_affine_K1e{exponent}",
                "experiment": "B",
                "tier": "full",
                "master_seed": master_seed,
                "grid": {"affine_condition": condition},
                "target": target,
                "iterations": 60,
                "certified_step_sizes": certified,
                "methods": [
                    {"name": "FR--R", "step_size": 0.5 * certified["FR--R"]},
                    {"name": "FR--KL", "step_size": 0.5 * certified["FR--KL"]},
                    {"name": "FB--GVI", "step_size": 0.5 * certified["FB--GVI"]},
                ],
            }
        )
    return configs


def experiment_c() -> list[dict[str, Any]]:
    """Diao anisotropic Gaussian benchmark with a full stepsize sweep."""

    master_seed = 20260930
    target = {
        "kind": "gaussian",
        "dimension": 10,
        "diao_spectrum": True,
        "rotation": True,
        "mean_distribution": "uniform",
        "initial_covariance_scale": 1.0,
    }
    certified = cell_constants(target, master_seed, "C")["certified_step_sizes"]
    methods = swept_methods(("FR--R", "FR--KL", "FB--GVI"), certified, seeds=10)
    methods += [
        {
            "name": "FR--R",
            "step_size": certified["FR--R"],
            "quadratic_rescue": True,
            "iterations": 0,
            "seeds": 10,
            "tag": "rescue",
        },
        {
            "name": "FR--KL",
            "step_size": certified["FR--KL"],
            "quadratic_rescue": True,
            "iterations": 0,
            "seeds": 10,
            "tag": "rescue",
        },
    ]
    return [
        {
            "id": "C_diao_gaussian_d10",
            "experiment": "C",
            "tier": "full",
            "master_seed": master_seed,
            "instance_per_seed": True,
            "grid": {"dimension": 10, "condition": 1.0e9},
            "target": target,
            "iterations": 500,
            "certified_step_sizes": certified,
            "methods": methods,
        }
    ]


def experiment_d() -> list[dict[str, Any]]:
    """Strongly log-concave non-Gaussian comparison over d, kappa and rho."""

    configs: list[dict[str, Any]] = []
    for dimension in (2, 10, 50):
        for condition in (1.0, 10.0, 100.0):
            for rho in (0.1, 1.0, 5.0):
                master_seed = (
                    20261000 + dimension * 100 + int(np.log10(condition)) * 10 + int(rho * 2)
                )
                target = {
                    "kind": "logcosh",
                    "dimension": dimension,
                    "condition": condition,
                    "rho": rho,
                    "affine_condition": 10.0,
                    "initial_covariance_scale": 0.5,
                }
                constants = cell_constants(target, master_seed, "D")
                certified = constants["certified_step_sizes"]
                configs.append(
                    {
                        "id": f"D_logcosh_d{dimension}_k{condition:g}_rho{rho:g}",
                        "experiment": "D",
                        "tier": "full",
                        "master_seed": master_seed,
                        "instance_per_seed": True,
                        "grid": {"dimension": dimension, "condition": condition, "rho": rho},
                        "target": target,
                        "iterations": 400,
                        "record_every": 2 if dimension >= 50 else 1,
                        "curvature": constants,
                        "certified_step_sizes": certified,
                        "methods": swept_methods(
                            ("FR--R", "FR--KL", "FB--GVI"), certified, seeds=5
                        ),
                    }
                )
    return configs


def experiment_f() -> list[dict[str, Any]]:
    """Exact Gaussian local region and sharp energy threshold."""

    configs: list[dict[str, Any]] = []
    for dimension in (2, 10, 100):
        for rate in (0.25, 0.5, 1.0, 1.5):
            initial = rate / 2.0
            master_seed = 20261100 + dimension + int(rate * 100)
            diagonal = [initial] + [1.0] * (dimension - 1)
            target = {
                "kind": "gaussian",
                "dimension": dimension,
                "condition": 1.0,
                "rotation": False,
                "mean_scale": 0.0,
                "initial_covariance_diagonal": diagonal,
            }
            certified = cell_constants(target, master_seed, "F")["certified_step_sizes"]
            configs.append(
                {
                    "id": f"F_gaussian_local_d{dimension}_rate{rate:g}",
                    "experiment": "F",
                    "tier": "full",
                    "master_seed": master_seed,
                    "grid": {"dimension": dimension, "rate": rate, "initial_eigenvalue": initial},
                    "target": target,
                    "iterations": 200,
                    "certified_step_sizes": certified,
                    "methods": [
                        {"name": "FR--R", "step_size": 0.1, "normalized_step_size": 0.1},
                        {"name": "FR--KL", "step_size": 0.1, "normalized_step_size": 0.1},
                    ],
                }
            )
    return configs


def experiment_g() -> list[dict[str, Any]]:
    """Near-Gaussian local spectral rate against the linearized generator."""

    configs: list[dict[str, Any]] = []
    for dimension in (2, 3, 5):
        for rho in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0):
            master_seed = 20261200 + dimension * 10 + int(rho * 1000) % 97
            target = {
                "kind": "logcosh",
                "dimension": dimension,
                "condition": 1.0,
                "rho": rho,
                "affine_condition": 1.0,
                "initial_covariance_scale": 0.9,
            }
            constants = cell_constants(target, master_seed, "G")
            configs.append(
                {
                    "id": f"G_local_spectral_d{dimension}_rho{rho:g}",
                    "experiment": "G",
                    "tier": "full",
                    "master_seed": master_seed,
                    "grid": {"dimension": dimension, "rho": rho},
                    "target": target,
                    "local_operator_points": 8192,
                    "iterations": 200,
                    "curvature": constants,
                    "methods": [
                        {"name": "FR--R", "step_size": 0.05, "normalized_step_size": 0.05},
                        {"name": "FR--KL", "step_size": 0.05, "normalized_step_size": 0.05},
                    ],
                }
            )
    return configs


def experiment_h() -> list[dict[str, Any]]:
    """Gaussian STL pathwise cancellation with matched covariance."""

    configs: list[dict[str, Any]] = []
    for dimension in (2, 10, 50):
        master_seed = 20261300 + dimension
        target = {
            "kind": "gaussian",
            "dimension": dimension,
            "condition": 100.0,
            "rotation": True,
            "mean_scale": 0.3,
            "initial_mean_scale": 1.0,
            "initial_covariance": "target",
        }
        certified = cell_constants(target, master_seed, "H")["certified_step_sizes"]
        configs.append(
            {
                "id": f"H_stl_cancellation_d{dimension}",
                "experiment": "H",
                "tier": "full",
                "master_seed": master_seed,
                "grid": {"dimension": dimension},
                "target": target,
                "iterations": 80,
                "certified_step_sizes": certified,
                "methods": [
                    {
                        "name": "FR--R--STL",
                        "step_size": 0.5 * certified["FR--R--STL"],
                        "batch_size": 1,
                        "seeds": STOCHASTIC_SEEDS,
                    },
                    {
                        "name": "FR--KL--STL",
                        "step_size": 0.5 * certified["FR--KL--STL"],
                        "batch_size": 1,
                        "seeds": STOCHASTIC_SEEDS,
                    },
                    {
                        "name": "S--FB--GVI",
                        "step_size": 0.5 * certified["S--FB--GVI"],
                        "batch_size": 1,
                        "seeds": STOCHASTIC_SEEDS,
                    },
                ],
            }
        )
    return configs


def experiment_i() -> list[dict[str, Any]]:
    """Raw-score against sticking-the-landing intrinsic variance."""

    configs: list[dict[str, Any]] = []
    cells = [
        ("gaussian", {"kind": "gaussian", "dimension": 8, "condition": 10.0, "rotation": True,
                      "mean_scale": 0.5, "initial_covariance_scale": 1.5}),
        ("logcosh", {"kind": "logcosh", "dimension": 8, "condition": 10.0, "rho": 1.0,
                     "affine_condition": 1.0, "initial_covariance_scale": 1.5}),
    ]
    for label, target in cells:
        for batch in (1, 4, 16):
            master_seed = 20261400 + batch + (0 if label == "gaussian" else 50)
            configs.append(
                {
                    "id": f"I_variance_{label}_d8_B{batch}",
                    "experiment": "I",
                    "tier": "full",
                    "master_seed": master_seed,
                    "grid": {"family": label, "batch_size": batch},
                    "target": target,
                    "batch_size": batch,
                    "variance_replicates": 2000,
                    "evaluation_points": 4096,
                    "interpolation_levels": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
                    "seeds": 5,
                }
            )
    return configs


def experiment_j() -> list[dict[str, Any]]:
    """Minibatch residual floors against batch size."""

    configs: list[dict[str, Any]] = []
    for dimension in (8, 32):
        for condition in (10.0, 100.0):
            for rho in (0.5, 1.0):
                for batch in (1, 2, 4, 8, 16, 32, 64):
                    master_seed = (
                        20261500
                        + dimension * 10
                        + int(np.log10(condition)) * 3
                        + int(rho * 2)
                    )
                    target = {
                        "kind": "logcosh",
                        "dimension": dimension,
                        "condition": condition,
                        "rho": rho,
                        "affine_condition": 1.0,
                        "initial_covariance_scale": 0.7,
                    }
                    constants = cell_constants(target, master_seed, "J")
                    certified = constants["certified_step_sizes"]
                    configs.append(
                        {
                            "id": (
                                f"J_floor_d{dimension}_k{condition:g}"
                                f"_rho{rho:g}_B{batch}"
                            ),
                            "experiment": "J",
                            "tier": "full",
                            "master_seed": master_seed,
                            "grid": {
                                "dimension": dimension,
                                "condition": condition,
                                "rho": rho,
                                "batch_size": batch,
                            },
                            "target": target,
                            "iterations": 600,
                            "record_every": 2,
                            "curvature": constants,
                            "certified_step_sizes": certified,
                            "methods": [
                                {
                                    "name": "FR--R--STL",
                                    "step_size": 0.5 * certified["FR--R--STL"],
                                    "batch_size": batch,
                                    "quadratic_rescue": True,
                                    "seeds": STOCHASTIC_SEEDS,
                                },
                                {
                                    "name": "FR--KL--STL",
                                    "step_size": 0.5 * certified["FR--KL--STL"],
                                    "batch_size": batch,
                                    "quadratic_rescue": True,
                                    "seeds": STOCHASTIC_SEEDS,
                                },
                                {
                                    "name": "FR--R--STL",
                                    "step_size": 0.5 * certified["FR--R--STL"],
                                    "batch_size": batch,
                                    "seeds": STOCHASTIC_SEEDS,
                                    "tag": "noqr",
                                },
                                {
                                    "name": "FR--KL--STL",
                                    "step_size": 0.5 * certified["FR--KL--STL"],
                                    "batch_size": batch,
                                    "seeds": STOCHASTIC_SEEDS,
                                    "tag": "noqr",
                                },
                                {
                                    "name": "S--FB--GVI",
                                    "step_size": 0.5 * certified["S--FB--GVI"],
                                    "batch_size": batch,
                                    "seeds": STOCHASTIC_SEEDS,
                                },
                            ],
                        }
                    )
    return configs


def experiment_k() -> list[dict[str, Any]]:
    """Decreasing stepsize schedule h_n = 8 kappa_star / (n + n0)."""

    configs: list[dict[str, Any]] = []
    for dimension in (4, 8):
        for rho in (0.3, 1.0):
            for batch in (1, 8):
                master_seed = 20261600 + dimension * 10 + int(rho * 10) + batch
                target = {
                    "kind": "logcosh",
                    "dimension": dimension,
                    "condition": 1.0,
                    "rho": rho,
                    "affine_condition": 1.0,
                    "initial_covariance_scale": 0.8,
                }
                constants = cell_constants(target, master_seed, "K")
                kappa_star = float(constants["kappa_star"])
                n0 = int(np.ceil(64.0 * kappa_star * kappa_star))
                configs.append(
                    {
                        "id": f"K_decreasing_d{dimension}_rho{rho:g}_B{batch}",
                        "experiment": "K",
                        "tier": "full",
                        "master_seed": master_seed,
                        "grid": {"dimension": dimension, "rho": rho, "batch_size": batch},
                        "target": target,
                        "iterations": 4000,
                        "record_every": 4,
                        "curvature": constants,
                        "methods": [
                            {
                                "name": "FR--R--STL",
                                "schedule": "manuscript_decreasing",
                                "kappa_star": kappa_star,
                                "n0": n0,
                                "batch_size": batch,
                                "quadratic_rescue": True,
                                "seeds": STOCHASTIC_SEEDS,
                            },
                            {
                                "name": "FR--KL--STL",
                                "schedule": "manuscript_decreasing",
                                "kappa_star": kappa_star,
                                "n0": n0,
                                "batch_size": batch,
                                "quadratic_rescue": True,
                                "seeds": STOCHASTIC_SEEDS,
                            },
                        ],
                    }
                )
    return configs


def experiment_l() -> list[dict[str, Any]]:
    """Bayesian logistic regression with a proper Gaussian prior."""

    configs: list[dict[str, Any]] = []
    for dimension in (10, 50, 100):
        for prior in (0.1, 1.0):
            for feature_exponent in (0, 2, 4):
                feature_condition = 10.0**feature_exponent
                master_seed = (
                    20261700 + dimension * 10 + int(prior * 10) + feature_exponent
                )
                samples = 10 * dimension
                target = {
                    "kind": "logistic",
                    "dimension": dimension,
                    "samples": samples,
                    "test_samples": max(500, 2 * samples),
                    "feature_condition": feature_condition,
                    "prior_precision": prior,
                }
                update_points = 256 if dimension <= 50 else 128
                constants = cell_constants(
                    target, master_seed, "L", reference_points=2048
                )
                certified = constants["certified_step_sizes"]
                deterministic = swept_methods(
                    ("FR--R", "FR--KL", "FB--GVI"),
                    certified,
                    multipliers=(0.125, 0.5, 1.0, 2.0, 8.0),
                )
                stochastic = [
                    {
                        "name": name,
                        "step_size": 0.5 * certified[name],
                        "batch_size": 16,
                        "seeds": 10,
                    }
                    for name in ("FR--R--STL", "FR--KL--STL", "S--FB--GVI")
                ]
                configs.append(
                    {
                        "id": (
                            f"L_logistic_d{dimension}_lam{prior:g}"
                            f"_fc1e{feature_exponent}"
                        ),
                        "experiment": "L",
                        "tier": "full",
                        "master_seed": master_seed,
                        "grid": {
                            "dimension": dimension,
                            "prior_precision": prior,
                            "feature_condition": feature_condition,
                        },
                        "target": target,
                        "update_points": update_points,
                        "evaluation_points": 512,
                        "reference_points": 2048,
                        "iterations": 200 if dimension <= 50 else 100,
                        "record_every": 1 if dimension <= 50 else 2,
                        "curvature": constants,
                        "certified_step_sizes": certified,
                        "methods": [*deterministic, *stochastic, {"name": "Laplace"}],
                    }
                )
    return configs


def experiment_m() -> list[dict[str, Any]]:
    """Section 5: modal rates of the general affine-invariant metric family."""

    configs: list[dict[str, Any]] = []
    for dimension in (2, 5, 10):
        metrics: list[dict[str, float]] = []
        for omega in (0.25, 0.5, 1.0, 2.0):
            for tau_scale in (-0.5, 0.0, 1.0, 4.0):
                # tau > -omega / N is the positive-definiteness condition; the
                # negative member is placed at half of the admissible distance.
                tau = tau_scale * omega / dimension
                if tau <= -omega / dimension:
                    continue
                metrics.append({"omega": float(omega), "tau": float(tau)})
        master_seed = 20261800 + dimension
        configs.append(
            {
                "id": f"M_affine_metric_d{dimension}",
                "experiment": "M",
                "tier": "full",
                "master_seed": master_seed,
                "grid": {"dimension": dimension},
                "target": {
                    "kind": "gaussian",
                    "dimension": dimension,
                    "condition": 20.0,
                    "rotation": True,
                    "mean_scale": 0.0,
                },
                "step_size": 0.01,
                "iterations": 600,
                "perturbation": 1.0e-4,
                "fit_fraction": 0.6,
                "seeds": 5,
                "metrics": metrics,
            }
        )
    return configs


BUILDERS = {
    "A": experiment_a,
    "B": experiment_b,
    "C": experiment_c,
    "D": experiment_d,
    "F": experiment_f,
    "G": experiment_g,
    "H": experiment_h,
    "I": experiment_i,
    "J": experiment_j,
    "K": experiment_k,
    "L": experiment_l,
    "M": experiment_m,
}


def write_configs(destination: Path | None = None, experiments: Iterable[str] | None = None) -> int:
    destination = destination or (CONFIG_ROOT / "full")
    destination.mkdir(parents=True, exist_ok=True)
    selected = list(experiments) if experiments else sorted(BUILDERS)
    written = 0
    for key in selected:
        for config in BUILDERS[key]():
            path = destination / f"{config['id']}.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written += 1
    return written


def main(arguments: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args(arguments)
    count = write_configs(args.destination, args.experiments)
    print(f"wrote {count} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
