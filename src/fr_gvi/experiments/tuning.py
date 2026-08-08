"""An implementable stepsize rule for the practical benchmark.

The theorem-diagnostic panels use certified stepsizes, and the earlier practical
panels use a multiple of ``1 / (beta_star max{lambda_{0,star}^max, 1})``.  Both
are legitimate where they are used and neither is usable here: they are written
in optimizer-whitened quantities, so computing them requires ``C_star`` -- the
answer.  A benchmark whose stepsizes were chosen with the optimizer in hand is
easy to dismiss no matter how carefully the rest of it was done.

This module supplies the replacement.  Every quantity it touches is available to
someone who has the model and the initialization and nothing else:

* the base scale ``h_0`` comes from the geometry of the update and, where the
  update is not scale free, from one expected Hessian at the *initial* state;
* the candidates are the dyadic grid ``h_0 2^{-k}``;
* the screen is stability along the whole pilot path -- finite objective, a
  positive definite and numerically well-conditioned covariance at every iterate,
  no repair, and a decrease overall;
* among the survivors the choice is the lowest **training** objective after a
  fixed pilot oracle budget.

Selection runs on a stratified subsample of the training rows, so no number that
reaches a figure was used to pick a stepsize; the multiplier is then frozen and
the final run recomputes ``h_0`` on the full data.  The oracle calls the pilot
spends are counted and reported, so the tuning is charged rather than assumed
free.

Neither ``C_star``, ``beta_star``, ``kappa_star`` nor any reference solution
enters at any point; :func:`selection_uses_only_implementable_inputs` states that
as an executable claim and the regression suite checks it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fr_gvi.algorithms.core import AlgorithmFailure, GaussianState, Method, step
from fr_gvi.diagnostics.core import objective
from fr_gvi.expectations.core import ExpectationEngine
from fr_gvi.experiments.factories import BuiltProblem
from fr_gvi.targets.core import Target
from fr_gvi.utils.accounting import OperationCounts

ROOT = Path(__file__).resolve().parents[3]
PRACTICAL_STEPS = ROOT / "configs" / "manuscript" / "practical_steps.json"

# 2^3 down to 2^-11: fifteen candidates, wide enough that the selected step is an
# interior point on every cell run so far.  A boundary selection is recorded as
# censored rather than reported as if the grid had bracketed it.
EXPONENTS = tuple(range(-3, 12))
PILOT_ITERATIONS = 60
PILOT_SUBSAMPLE = 0.5
PILOT_SEED = 20260803
# Independent draws of a randomly generated pilot problem.  A fixed dataset has
# one instance; a synthetic cell is redrawn, and the frozen multiplier is the
# most conservative choice across the draws.
PILOT_INSTANCES = (20260805, 20260806, 20260807)
# The covariance may not become numerically singular at any pilot iterate.  Well
# below 1/eps: a covariance this ill-conditioned makes every solve downstream
# meaningless long before it stops being representable.
#
# This screen replaced an earlier one on the objective's excursion above its
# starting value, which was the wrong instrument.  Both were tried against the
# two cases that motivated a path screen at all.  On one real posterior the
# selected step drove the covariance conditioning to 4.7e10 within a few
# iterations and needed a roundoff repair; on another, the KL/Bregman scheme's
# first-step covariance contraction raises the objective more than twentyfold
# while the conditioning never exceeds 62, and it converges to the same optimum
# 30 times faster than any other step in the grid.  The excursion screen rejects
# both; the conditioning screen separates them, which is what a screen is for.
CONDITIONING_LIMIT = 1.0e10


def implementable_base_step(
    method: Method,
    target: Target,
    initial: GaussianState,
    engine: ExpectationEngine,
) -> tuple[float, int]:
    """``h_0`` for one method, and the oracle calls spent computing it.

    The Fisher--Rao and square-root natural-gradient updates are affine
    equivariant, so their stepsize is dimensionless: multiplying the target by a
    matrix leaves the admissible range unchanged, and ``h_0 = 1`` is the natural
    origin of the grid.  The Bures--Wasserstein and parameter-space updates are
    Euclidean gradient steps whose stepsize carries units of inverse curvature,
    so they are scaled by one expected Hessian at the initialization -- one
    oracle call, at a state the practitioner chose.
    """

    if method.geometry in {"fisher-rao", "fisher-rao-square-root"}:
        return 1.0, 0
    result = engine.evaluate(target, initial.mean, initial.covariance)
    curvature = float(np.linalg.eigvalsh(np.asarray(result.hessian, dtype=np.float64))[-1])
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError(f"non-positive curvature {curvature} at the initialization")
    return 1.0 / curvature, 1


def selection_uses_only_implementable_inputs() -> tuple[str, ...]:
    """The inputs the rule is allowed to read, as an explicit contract."""

    return (
        "the target's own gradient and Hessian oracle",
        "the initialization chosen by the practitioner",
        "the training objective on a subsample of the training data",
    )


@dataclass
class Candidate:
    exponent: int
    multiplier: float
    step_size: float
    admissible: bool
    reason: str
    initial_objective: float
    final_objective: float
    iterations: int
    oracle_pairs: int
    gradient_evaluations: int
    worst_conditioning: float = 1.0


@dataclass
class Selection:
    problem: str
    method: str
    exponent: int
    multiplier: float
    pilot_step_size: float
    pilot_base_step: float
    pilot_objective: float
    censored: bool
    tuning_oracle_pairs: int
    tuning_gradient_evaluations: int
    tuning_seconds: float
    candidates: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return payload


def _pilot_trajectory(
    method: Method,
    problem: BuiltProblem,
    engine: ExpectationEngine,
    step_size: float,
    *,
    iterations: int,
    batch_size: int,
    seed: int,
    projection_floor: float | None,
) -> Candidate:
    """One candidate, screened on stability and scored on the training objective."""

    state = problem.initial_state
    counts = OperationCounts()
    rng = np.random.default_rng(seed)
    initial_objective, _ = objective(problem.target, state, engine)
    current = initial_objective
    reason = ""
    completed = 0
    worst_conditioning = 1.0
    for _ in range(iterations):
        try:
            state, diagnostics = step(
                method,
                problem.target,
                state,
                step_size,
                engine=engine,
                rng=rng,
                batch_size=batch_size,
                counts=counts,
                projection_floor=projection_floor,
            )
            if diagnostics.repair is not None:
                reason = "a covariance repair was logged"
                break
            current, _ = objective(problem.target, state, engine)
        except (AlgorithmFailure, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            # A candidate that breaks down is inadmissible, not an error: the
            # grid is deliberately swept past each method's stability boundary,
            # and where the boundary is is part of what the sweep measures.
            reason = f"{type(exc).__name__}: {exc}"
            break
        completed += 1
        if not np.isfinite(current):
            reason = "non-finite objective"
            break
        # A screen on the *path*, not just its endpoint.  Comparing the last
        # iterate with the first accepts a candidate that passes through a state
        # no downstream solve could use and then recovers, which is exactly what
        # one real posterior did.  The conditioning of the current covariance is
        # a property of the iterate a practitioner holds, so screening on it
        # costs no information the rule is not allowed to have.
        eigenvalues = np.linalg.eigvalsh(state.covariance)
        conditioning = float(eigenvalues[-1] / max(float(eigenvalues[0]), np.finfo(np.float64).tiny))
        worst_conditioning = max(worst_conditioning, conditioning)
        if conditioning > CONDITIONING_LIMIT:
            reason = f"covariance conditioning {conditioning:.3e} above {CONDITIONING_LIMIT:.3e}"
            break
    admissible = (
        not reason
        and completed == iterations
        and np.isfinite(current)
        and current < initial_objective
    )
    if not reason and not admissible:
        reason = "no decrease over the pilot horizon"
    return Candidate(
        exponent=0,
        multiplier=0.0,
        step_size=step_size,
        admissible=bool(admissible),
        reason=reason,
        initial_objective=float(initial_objective),
        final_objective=float(current),
        iterations=completed,
        oracle_pairs=int(counts.oracle_pairs),
        gradient_evaluations=int(counts.gradient_evaluations),
        worst_conditioning=float(worst_conditioning),
    )


def select_step(
    method: Method,
    problem: BuiltProblem,
    engine: ExpectationEngine,
    *,
    problem_name: str,
    batch_size: int = 1,
    iterations: int = PILOT_ITERATIONS,
    seed: int = PILOT_SEED,
    projection_floor: float | None = None,
) -> Selection:
    """Sweep the dyadic grid on the pilot problem and freeze the multiplier."""

    started = time.perf_counter()
    base, base_oracles = implementable_base_step(method, problem.target, problem.initial_state, engine)
    candidates: list[Candidate] = []
    for exponent in EXPONENTS:
        multiplier = 2.0 ** (-exponent)
        candidate = _pilot_trajectory(
            method,
            problem,
            engine,
            base * multiplier,
            iterations=iterations,
            batch_size=batch_size,
            seed=seed,
            projection_floor=projection_floor,
        )
        candidate.exponent = exponent
        candidate.multiplier = multiplier
        candidates.append(candidate)

    admissible = [candidate for candidate in candidates if candidate.admissible]
    if not admissible:
        raise RuntimeError(
            f"{problem_name}/{method.value}: no admissible stepsize in "
            f"[{base * 2.0 ** -EXPONENTS[-1]:.3e}, {base * 2.0 ** -EXPONENTS[0]:.3e}]"
        )
    # Lowest pilot training objective; ties are broken towards the smaller step,
    # so the choice never sits on the stability boundary by accident.
    best = min(admissible, key=lambda candidate: (candidate.final_objective, -candidate.exponent))
    interior = best.exponent not in (EXPONENTS[0], EXPONENTS[-1])
    return Selection(
        problem=problem_name,
        method=method.value,
        exponent=best.exponent,
        multiplier=best.multiplier,
        pilot_step_size=best.step_size,
        pilot_base_step=base,
        pilot_objective=best.final_objective,
        censored=not interior,
        tuning_oracle_pairs=base_oracles + sum(candidate.oracle_pairs for candidate in candidates),
        tuning_gradient_evaluations=sum(candidate.gradient_evaluations for candidate in candidates),
        tuning_seconds=time.perf_counter() - started,
        candidates=candidates,
    )


def load_selected() -> dict[str, dict[str, Any]]:
    if not PRACTICAL_STEPS.exists():
        raise FileNotFoundError(
            f"{PRACTICAL_STEPS} is missing; run `make manuscript-tuning` first"
        )
    return json.loads(PRACTICAL_STEPS.read_text(encoding="utf-8"))["selections"]


def selected_multiplier(problem_name: str, method: str) -> float:
    selections = load_selected()
    key = f"{problem_name}:{method}"
    if key not in selections:
        raise KeyError(f"no frozen multiplier for {key}")
    return float(selections[key]["multiplier"])


def _benchmark_methods(stochastic: bool) -> tuple[Method, ...]:
    if stochastic:
        return (
            Method.FR_R_STL,
            Method.FR_KL_STL,
            Method.S_FB_GVI,
            Method.PRICE_BBVI,
            Method.BBVI_STL,
        )
    return (Method.FR_R, Method.FR_KL, Method.FB_GVI, Method.SQ_NGVI)


@dataclass(frozen=True)
class PilotProblem:
    """One problem the rule is calibrated on, and the methods to calibrate.

    ``seeds`` may name more than one instance.  Where the problem is drawn at
    random the stability boundary moves with the draw, and a multiplier selected
    on a single instance need not be admissible on another: on the Gaussian
    cancellation cell the boundary for the gradient-only estimator shifts by a
    full grid factor between draws, so a step chosen on one instance diverges on
    the next.  With several instances the frozen multiplier is the most
    conservative of the per-instance choices, which is what makes it transfer.
    Fixed problems -- the real datasets -- have exactly one instance and are
    unaffected.
    """

    name: str
    target_config: dict[str, Any]
    experiment: str
    seeds: tuple[int, ...]
    methods: tuple[Method, ...]
    # The batch size the *run* will use.  A stochastic step admissible at one
    # batch size need not be admissible at another -- the noise the update has to
    # absorb scales like 1/B -- so calibrating at a batch size the deployment
    # does not use is calibrating a different algorithm.
    batch_size: int | None = None


def dataset_pilots(
    datasets: tuple[str, ...], prior_precision: float
) -> tuple[PilotProblem, ...]:
    """Real posteriors, tuned on a disjoint subsample of the training rows."""

    methods = (*_benchmark_methods(False), *_benchmark_methods(True))
    return tuple(
        PilotProblem(
            dataset,
            {
                "kind": "logistic_dataset",
                "dataset": dataset,
                "prior_precision": prior_precision,
                "subsample_fraction": PILOT_SUBSAMPLE,
            },
            "R",
            (0,),
            methods,
        )
        for dataset in datasets
    )


def cancellation_pilot() -> tuple[PilotProblem, ...]:
    """The Gaussian cell of the stochastic cancellation panel.

    That panel compares five estimators on one target, so they must not be forced
    onto one stepsize: the Fisher--Rao updates are affine equivariant and the
    parameter-space ones are not, and this target's optimizing covariance spans
    two decades, so a step admissible for the first pair is outside the stable
    range of the second.  Tuning each method by the same rule as the benchmark
    makes the panel a comparison of estimators rather than of stepsize luck.
    """

    from fr_gvi.experiments.manuscript import (
        CANCELLATION_BATCH,
        CANCELLATION_CELL,
        CANCELLATION_METHODS,
    )

    return (
        PilotProblem(
            "cancellation",
            dict(CANCELLATION_CELL),
            "H",
            PILOT_INSTANCES,
            tuple(Method(name) for name in CANCELLATION_METHODS),
            batch_size=CANCELLATION_BATCH,
        ),
    )


def scaling_pilots() -> tuple[PilotProblem, ...]:
    """Synthetic scaling cells, tuned on a *different* problem instance.

    The scaling study reports time to a fixed accuracy as well as cost per
    iteration, and the first of those depends on the stepsize, so it is tuned by
    the same implementable rule as the benchmark rather than by a scale written
    in optimizer-whitened constants.  The pilot seed differs from the one the
    scaling runs use, so the calibration instance is not a measured instance.
    """

    from fr_gvi.experiments.manuscript import SCALING_DIMENSIONS

    return tuple(
        PilotProblem(
            f"scaling_d{dimension}",
            {
                "kind": "logistic",
                "dimension": dimension,
                "samples": 10 * dimension,
                "test_samples": 10 * dimension,
                "feature_condition": 100.0,
                "prior_precision": 1.0,
                "initial_covariance": "prior",
            },
            "S",
            tuple(seed + dimension for seed in PILOT_INSTANCES),
            _benchmark_methods(False),
        )
        for dimension in SCALING_DIMENSIONS
    )


def run_pilot(
    problems: tuple[PilotProblem, ...],
    *,
    batch_size: int,
    iterations: int = PILOT_ITERATIONS,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sweep the dyadic grid for every method on every pilot problem.

    ``existing`` carries selections already computed.  The sweep is deterministic
    given the code and the pilot problem, so reusing an unchanged entry is a
    cache hit rather than a shortcut; the audit checks that every benchmark
    config's stepsize still reproduces from the record either way.
    """

    from fr_gvi.diagnostics.curvature import curvature_constants
    from fr_gvi.expectations.core import ExactGaussianExpectation, LogisticExactExpectation
    from fr_gvi.experiments.factories import build_problem
    from fr_gvi.targets.core import GaussianTarget

    selections: dict[str, Any] = dict(existing or {})
    for pilot in problems:
        instances = []
        for seed in pilot.seeds:
            problem = build_problem(pilot.target_config, seed, pilot.experiment)
            engine = (
                ExactGaussianExpectation()
                if isinstance(problem.target, GaussianTarget)
                else LogisticExactExpectation(48)
            )
            # The log-smoothness in the original coordinates is a closed-form
            # property of the design matrix, so BBVI--STL's projection radius
            # costs nothing and leaks nothing.  It is evaluated against the
            # *initial* state, which is all the rule is allowed to see.
            smoothness = float(curvature_constants(problem.target, problem.initial_state).beta)
            instances.append((seed, problem, engine, smoothness))
        for method in pilot.methods:
            key = f"{pilot.name}:{method.value}"
            if key in selections:
                print(f"{pilot.name:<13} {method.value:<12} cached", flush=True)
                continue
            per_instance = [
                select_step(
                    method,
                    problem,
                    engine,
                    problem_name=f"{pilot.name}@{seed}",
                    batch_size=(pilot.batch_size or batch_size) if method.stochastic else 1,
                    iterations=iterations,
                    projection_floor=1.0 / np.sqrt(smoothness)
                    if method is Method.BBVI_STL
                    else None,
                )
                for seed, problem, engine, smoothness in instances
            ]
            # The most conservative choice across the draws.  A larger exponent is
            # a smaller step, so the transferable multiplier is the maximum
            # exponent: it is admissible on every instance the pilot saw, which
            # is what a single-instance selection cannot promise.
            selection = max(per_instance, key=lambda candidate: candidate.exponent)
            selection.problem = pilot.name
            selection.tuning_oracle_pairs = sum(c.tuning_oracle_pairs for c in per_instance)
            selection.tuning_gradient_evaluations = sum(
                c.tuning_gradient_evaluations for c in per_instance
            )
            selection.tuning_seconds = sum(c.tuning_seconds for c in per_instance)
            selection.censored = any(c.censored for c in per_instance)
            selections[key] = {
                **selection.to_dict(),
                "instances": [
                    {"seed": int(seed), "exponent": int(candidate.exponent)}
                    for (seed, _, _, _), candidate in zip(instances, per_instance, strict=True)
                ],
            }
            spread = sorted({candidate.exponent for candidate in per_instance})
            flag = " (censored)" if selection.censored else ""
            note = f" instances 2^-{spread}" if len(spread) > 1 else ""
            print(
                f"{pilot.name:<13} {method.value:<12} 2^-{selection.exponent:<3} "
                f"h={selection.pilot_step_size:.4e} F={selection.pilot_objective:.6e}"
                f"{flag}{note}",
                flush=True,
            )
    return selections


def write(selections: dict[str, Any], *, problems: tuple[str, ...], batch_size: int) -> None:
    document = {
        "schema_version": 1,
        "rule": {
            "base_step": {
                "fisher-rao": "1 (the natural-gradient step is dimensionless)",
                "fisher-rao-square-root": "1",
                "bures-wasserstein": "1 / lambda_max(H(a_0))",
                "parameter-space": "1 / lambda_max(H(a_0))",
            },
            "grid": [f"2^-{exponent}" for exponent in EXPONENTS],
            "admissibility": [
                "positive definite covariance for the whole pilot horizon",
                "finite objective at every recorded iterate",
                "no covariance repair logged",
                f"covariance conditioning below {CONDITIONING_LIMIT:g} throughout",
                "objective decrease over the pilot horizon",
            ],
            "selection": "lowest training objective after the fixed pilot budget",
            "pilot_iterations": PILOT_ITERATIONS,
            "pilot_subsample_fraction": PILOT_SUBSAMPLE,
            "inputs": list(selection_uses_only_implementable_inputs()),
            "forbidden_inputs": ["C_star", "alpha_star", "beta_star", "kappa_star", "any reference solution"],
        },
        "problems": list(problems),
        "batch_size": batch_size,
        "selections": selections,
    }
    PRACTICAL_STEPS.parent.mkdir(parents=True, exist_ok=True)
    PRACTICAL_STEPS.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    from fr_gvi.experiments.datasets import DATASET_KEYS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_KEYS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prior-precision", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=PILOT_ITERATIONS)
    parser.add_argument(
        "--no-scaling", action="store_true", help="tune the datasets only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep selections already recorded and compute only the missing ones",
    )
    args = parser.parse_args(arguments)
    problems = (
        *dataset_pilots(tuple(args.datasets), args.prior_precision),
        *cancellation_pilot(),
    )
    if not args.no_scaling:
        problems = (*problems, *scaling_pilots())
    existing = {}
    if args.resume and PRACTICAL_STEPS.exists():
        existing = json.loads(PRACTICAL_STEPS.read_text(encoding="utf-8")).get("selections", {})
    selections = run_pilot(
        problems,
        batch_size=args.batch_size,
        iterations=args.iterations,
        existing=existing,
    )
    write(
        selections,
        problems=tuple(pilot.name for pilot in problems),
        batch_size=args.batch_size,
    )
    censored = [key for key, value in selections.items() if value["censored"]]
    total = sum(int(value["tuning_oracle_pairs"]) for value in selections.values())
    print(f"\n{len(selections)} selections, {total} tuning oracle pairs")
    if censored:
        print(f"censored at a grid endpoint: {', '.join(sorted(censored))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
