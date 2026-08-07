"""Resumable, idempotent experiment campaign runner."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit

from fr_gvi.algorithms.affine_metric import (
    AffineMetric,
    modal_decomposition,
    retraction_step,
)
from fr_gvi.algorithms.core import (
    AlgorithmFailure,
    GaussianState,
    Method,
    quadratic_rescue,
    step,
)
from fr_gvi.diagnostics.curvature import (
    certified_step_sizes,
    curvature_constants,
    whitened_initialization,
)
from fr_gvi.diagnostics.core import (
    certified_gap_bound,
    gaussian_core_rate,
    gaussian_kl_gap,
    gaussian_w2_squared,
    objective,
    optimizer_relative,
    residuals,
)
from fr_gvi.diagnostics.local_operator import (
    assemble_local_operator,
    discrete_rate,
    kl_local_gap,
    symmetric_star_basis,
)
from fr_gvi.expectations.core import (
    ExactGaussianExpectation,
    ExpectationEngine,
    FixedNormalExpectation,
    GaussHermiteLogCoshExpectation,
    LogisticExactExpectation,
    expected_bernoulli_probabilities,
)
from fr_gvi.experiments.factories import BuiltProblem, build_problem
from fr_gvi.experiments.reference import ReferenceSolution, laplace_approximation, solve_reference
from fr_gvi.linear_algebra.spd import spd_solve, spd_sqrt, symmetrize as symmetrize_matrix
from fr_gvi.targets.core import (
    GaussianTarget,
    LogisticRegressionTarget,
    ShiftedLogCoshTarget,
    Target,
    mean_hessian,
)
from fr_gvi.utils.accounting import OperationCounts

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
STATE_PATH = RESULTS / "manifests" / "campaign_state.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


# Modules that read results and write reports.  They cannot change a trajectory, so
# they stay out of the numerical source hash: otherwise correcting a caption or
# tightening an audit check would invalidate a completed campaign and force a rerun
# that could not alter a single number.  The plotting package is excluded for the
# same reason and always has been.
REPORTING_MODULES = frozenset(
    {"audit.py", "tables.py", "manuscript_audit.py", "manuscript_tables.py"}
)


def code_hash() -> str:
    """Hash of everything that can affect what a trajectory computes."""

    digest = hashlib.sha256()
    paths = [
        path
        for path in sorted((ROOT / "src").rglob("*.py"))
        if "plotting" not in path.relative_to(ROOT / "src").parts
        and path.name not in REPORTING_MODULES
    ]
    paths += [ROOT / "pyproject.toml", ROOT / "requirements-lock.txt"]
    paths = sorted(paths)
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reference_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hash_file(path)
        for path in sorted((ROOT / "references").glob("*"))
        if path.is_file()
    }


# The paths that determine what a run computes.  A campaign writes its own results
# and manifests as it goes, so a plain ``git status`` is dirty from the second
# trajectory onwards and would report every later run as unreproducible.  The flag
# has to mean "the source differs from the commit", not "outputs exist".
SOURCE_PATHS = ("src", "configs", "scripts", "pyproject.toml", "requirements-lock.txt")


def git_metadata() -> tuple[str, bool, str]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD") or "unborn"
    status = run("status", "--short", "--", *SOURCE_PATHS)
    return commit, bool(status), status


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "matplotlib", "pytest"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def blas_information() -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    return stream.getvalue()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "runs": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_state(state: dict[str, Any]) -> None:
    save_json(STATE_PATH, state)


def load_configs(paths: Iterable[str]) -> list[tuple[Path, dict[str, Any]]]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            # Recursive so a config tree may be grouped into per-figure directories.
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    configs: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        required = {"id", "experiment", "tier", "target"}
        missing = required.difference(config)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        configs.append((path, config))
    return configs


def seed_for(master_seed: int, stream: int, repeat: int = 0) -> int:
    sequence = np.random.SeedSequence(master_seed, spawn_key=(stream, repeat))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def engines(
    target: Target,
    *,
    update_points: int,
    evaluation_points: int,
    update_seed: int,
    evaluation_seed: int,
) -> tuple[ExpectationEngine, ExpectationEngine]:
    if isinstance(target, GaussianTarget):
        exact = ExactGaussianExpectation()
        return exact, exact
    if isinstance(target, ShiftedLogCoshTarget):
        # The panelled rule holds ~1e-13 relative accuracy from order 24 upward,
        # uniformly in the marginal width, so these orders are comfortably
        # converged; a doubling test is in the unit suite.
        return GaussHermiteLogCoshExpectation(32), GaussHermiteLogCoshExpectation(64)
    if isinstance(target, LogisticRegressionTarget):
        # The logistic potential depends on theta only through the linear
        # predictors, so its Gaussian expectations are exact one-dimensional
        # integrals.  There is no sampling design to share or to transfer between,
        # and the reported gaps are gaps to the true Gaussian variational optimum.
        del update_points, evaluation_points, update_seed, evaluation_seed
        return LogisticExactExpectation(48), LogisticExactExpectation(96)
    # Nonseparable targets share one fixed quadrature design between the
    # deterministic updates, the objective evaluation and the reference solve.
    # The discretized problem is then internally exact, and the quadrature error
    # is reported separately by the reference's design-transfer diagnostics.
    del evaluation_points, evaluation_seed
    shared = FixedNormalExpectation.qmc(target.dimension, update_points, update_seed)
    return shared, shared


def _method_slug(method: Method, specification: dict[str, Any]) -> str:
    slug = method.value.lower().replace("--", "-").replace("+", "-plus-")
    if specification.get("quadratic_rescue", False):
        slug += "-qr"
    if specification.get("raw_mean_ablation", False):
        slug += "-raw"
    if "batch_size" in specification:
        slug += f"-b{int(specification['batch_size'])}"
    if "tag" in specification:
        tag = str(specification["tag"]).lower().replace("_", "-")
        if not tag or any(not (character.isalnum() or character == "-") for character in tag):
            raise ValueError("method tag must contain only letters, digits, underscores, or hyphens")
        slug += f"-{tag}"
    return slug


def _predictive_metrics(
    problem: BuiltProblem,
    state: GaussianState,
    normals: np.ndarray,
) -> dict[str, float]:
    """Held-out predictive quality of the Gaussian posterior approximation.

    For a logistic model the posterior predictive probability is a
    one-dimensional Gaussian integral per test point, so it is computed exactly
    and the sampling design is unused; the argument is retained for other target
    families that have no such reduction.
    """

    if problem.heldout is None:
        return {"predictive_nll": np.nan, "classification_error": np.nan, "brier": np.nan}
    features, labels = problem.heldout
    if isinstance(problem.target, LogisticRegressionTarget):
        probabilities = expected_bernoulli_probabilities(
            features, state.mean, state.covariance
        )
    else:
        root = spd_sqrt(state.covariance)
        parameter_samples = state.mean + normals @ root.T
        probabilities = np.mean(expit(parameter_samples @ features.T), axis=0)
    epsilon = 1.0e-12
    nll = -np.mean(labels * np.log(probabilities + epsilon) + (1.0 - labels) * np.log(1.0 - probabilities + epsilon))
    classification = np.mean((probabilities >= 0.5) != labels)
    brier = np.mean((probabilities - labels) ** 2)
    return {
        "predictive_nll": float(nll),
        "classification_error": float(classification),
        "brier": float(brier),
    }


def trajectory_row(
    *,
    config: dict[str, Any],
    method: Method | str,
    seed: int,
    iteration: int,
    step_size: float,
    state: GaussianState,
    target: Target,
    optimum: GaussianState,
    reference_objective: float,
    evaluation_engine: ExpectationEngine,
    counts: OperationCounts,
    elapsed: float,
    repair: dict[str, float] | None,
    problem: BuiltProblem,
    predictive_normals: np.ndarray,
    algorithm_elapsed: float = np.nan,
    equivariance_error_mean: float = np.nan,
    equivariance_error_covariance: float = np.nan,
    equivariance_error_back: float = np.nan,
    local_gamma: float = np.nan,
    local_lambda: float = np.nan,
) -> dict[str, Any]:
    objective_value, expectation = objective(target, state, evaluation_engine)
    relative = optimizer_relative(state, optimum)
    certificate = residuals(state, expectation)
    eigenvalues = np.linalg.eigvalsh(state.covariance)
    exact_gap = gaussian_kl_gap(target, state) if isinstance(target, GaussianTarget) else np.nan
    gap = exact_gap if isinstance(target, GaussianTarget) else objective_value - reference_objective
    w2 = gaussian_w2_squared(state, optimum)
    predictive = _predictive_metrics(problem, state, predictive_normals)
    row: dict[str, Any] = {
        "tier": config["tier"],
        "experiment": config["experiment"],
        "job_id": config["id"],
        "method": method.value if isinstance(method, Method) else method,
        "seed": seed,
        "iteration": iteration,
        "step_size": step_size,
        "objective": objective_value,
        "objective_gap": gap,
        "exact_gaussian_kl": exact_gap,
        "w2_squared": w2,
        "mean_error": relative.mean_norm,
        "covariance_error": relative.covariance_frobenius,
        "covariance_min_eigenvalue": float(eigenvalues[0]),
        "covariance_max_eigenvalue": float(eigenvalues[-1]),
        "relative_covariance_min_eigenvalue": relative.covariance_minimum_eigenvalue,
        "relative_covariance_max_eigenvalue": relative.covariance_maximum_eigenvalue,
        "gaussian_core_rate": gaussian_core_rate(state, optimum),
        "fisher_rao_residual_squared": certificate.fisher_rao_squared,
        "bures_wasserstein_residual_squared": certificate.bures_wasserstein_squared,
        "wall_time_seconds": elapsed,
        # Cumulative time spent inside the algorithm's own update, excluding the
        # per-iteration diagnostics, which are not part of any method's cost.
        "algorithm_seconds": algorithm_elapsed,
        "repair": json.dumps(repair, sort_keys=True) if repair else "",
        "equivariance_error_mean": equivariance_error_mean,
        "equivariance_error_covariance": equivariance_error_covariance,
        "equivariance_error_back": equivariance_error_back,
        "local_gamma": local_gamma,
        "local_lambda": local_lambda,
        **predictive,
        **counts.to_dict(),
    }
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _local_eigenmode_state(
    operator: np.ndarray, optimum: GaussianState, radius: float
) -> tuple[GaussianState, np.ndarray]:
    """``a_0 = a_star + r v_min`` along the slowest mode of the linearized generator.

    ``v_min = (u, X)`` is the unit ``star``-norm eigenvector of ``L_star`` for its
    smallest eigenvalue ``gamma_star``.  A tangent vector at ``a_star`` acts on the
    state by ``m = m_star + r C_star^{1/2} u`` and
    ``C = C_star^{1/2} (I + r X) C_star^{1/2}``, which is the exponential-free
    first-order parameterization used throughout the local analysis.  The sign is
    pinned by the largest-magnitude component so the construction is reproducible.
    """

    dimension = optimum.mean.size
    values, vectors = np.linalg.eigh(operator)
    vector = np.asarray(vectors[:, 0], dtype=np.float64)
    vector = vector * np.sign(vector[int(np.argmax(np.abs(vector)))] or 1.0)
    basis = symmetric_star_basis(dimension)
    direction_mean = vector[:dimension]
    direction_covariance = np.einsum("k,kij->ij", vector[dimension:], np.asarray(basis))
    root = spd_sqrt(optimum.covariance)
    identity = np.eye(dimension, dtype=np.float64)
    perturbed = symmetrize_matrix(identity + radius * direction_covariance)
    if float(np.linalg.eigvalsh(perturbed)[0]) <= 0.0:
        raise ValueError(f"local eigenmode radius {radius} leaves the positive cone")
    state = GaussianState(
        optimum.mean + radius * root @ direction_mean,
        symmetrize_matrix(root @ perturbed @ root),
    )
    return state, values


def _base_equivariance_problem(problem: BuiltProblem) -> tuple[GaussianTarget, GaussianState, np.ndarray, np.ndarray]:
    metadata = problem.metadata
    transform = np.asarray(metadata["transform"], dtype=np.float64)
    shift = np.asarray(metadata["shift"], dtype=np.float64)
    target = GaussianTarget(
        np.asarray(metadata["base_mean"], dtype=np.float64),
        np.asarray(metadata["base_precision"], dtype=np.float64),
    )
    state = GaussianState(
        np.asarray(metadata["base_initial_mean"], dtype=np.float64),
        np.asarray(metadata["base_initial_covariance"], dtype=np.float64),
    )
    return target, state, transform, shift


def _run_trajectory(
    *,
    config: dict[str, Any],
    problem: BuiltProblem,
    reference: ReferenceSolution,
    method_specification: dict[str, Any],
    run_seed: int,
    update_engine: ExpectationEngine,
    evaluation_engine: ExpectationEngine,
    output_path: Path,
    instance: int = 0,
    curvature: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    method = Method(method_specification["name"])
    iterations = int(method_specification.get("iterations", config.get("iterations", 50)))
    batch_size = int(method_specification.get("batch_size", config.get("batch_size", 1)))
    base_step_size = float(method_specification.get("step_size", config.get("step_size", 0.1)))
    state = problem.initial_state
    rescue = bool(method_specification.get("quadratic_rescue", False))
    setup_seconds = 0.0
    if rescue:
        state = quadratic_rescue(problem.target, state.mean)
    if method == Method.LAPLACE:
        if not isinstance(problem.target, LogisticRegressionTarget):
            raise ValueError("Laplace is only an approximation-quality baseline for logistic regression")
        # Laplace is non-iterative, so its whole cost is the mode solve; timing it
        # keeps the wall-clock column comparable with the iterative methods.
        laplace_started = time.perf_counter()
        state = laplace_approximation(problem.target)
        setup_seconds = time.perf_counter() - laplace_started
        iterations = 0
    rng = np.random.default_rng(run_seed)
    counts = OperationCounts()
    if rescue:
        counts.gradient_evaluations += 1
        counts.hessian_evaluations += 1
        counts.oracle_pairs += 1
        counts.cholesky_factorizations += 2
        counts.cholesky_solves += 2
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    algorithm_elapsed = setup_seconds

    local_gamma = np.nan
    local_lambda = np.nan
    local_kl_gamma = np.nan
    local_discrete_rate = np.nan
    predicted_contraction = np.nan
    if config["experiment"] == "G":
        local_normals = FixedNormalExpectation.qmc(
            state.mean.size, int(config.get("local_operator_points", 2048)), run_seed + 37
        ).normals
        operator = assemble_local_operator(problem.target, reference.state, local_normals)
        spectrum = np.linalg.eigvalsh(operator)
        local_gamma, local_lambda = float(spectrum[0]), float(spectrum[-1])
        local_kl_gamma = kl_local_gap(operator, state.mean.size, base_step_size)
        gap = local_kl_gamma if method == Method.FR_KL else local_gamma
        local_discrete_rate = discrete_rate(gap, base_step_size)
        predicted_contraction = 1.0 - base_step_size * gap
        local_initialization = config.get("local_initialization")
        if local_initialization is not None:
            state, _ = _local_eigenmode_state(
                operator, reference.state, float(local_initialization["radius"])
            )

    predictive_normals = FixedNormalExpectation.qmc(
        state.mean.size, min(512, int(config.get("evaluation_points", 1024))), run_seed + 17
    ).normals

    base_target = None
    base_state = None
    transform = None
    shift = None
    base_rng = np.random.default_rng(run_seed)
    base_counts = OperationCounts()
    if config["experiment"] == "B":
        base_target, base_state, transform, shift = _base_equivariance_problem(problem)

    inverse_transform = (
        np.linalg.solve(transform, np.eye(transform.shape[0])) if transform is not None else None
    )

    def add_row(iteration: int, actual_step: float, repair: dict[str, float] | None) -> None:
        mean_error = np.nan
        covariance_error = np.nan
        back_error = np.nan
        if base_state is not None and transform is not None and shift is not None:
            expected_mean = transform @ base_state.mean + shift
            expected_covariance = transform @ base_state.covariance @ transform.T
            denominator_mean = max(float(np.linalg.norm(expected_mean)), 1.0)
            denominator_covariance = max(float(np.linalg.norm(expected_covariance, ord="fro")), 1.0)
            mean_error = float(np.linalg.norm(state.mean - expected_mean) / denominator_mean)
            covariance_error = float(
                np.linalg.norm(state.covariance - expected_covariance, ord="fro") / denominator_covariance
            )
            # The equivariance residual of the protocol is measured after mapping
            # the transformed trajectory *back* to the base coordinates, so the
            # discrepancy is reported in the coordinates the reference lives in.
            assert inverse_transform is not None
            back_mean = inverse_transform @ (state.mean - shift)
            back_covariance = inverse_transform @ state.covariance @ inverse_transform.T
            back_error = float(
                (
                    np.linalg.norm(back_mean - base_state.mean)
                    + np.linalg.norm(back_covariance - base_state.covariance, ord="fro")
                )
                / (
                    1.0
                    + float(np.linalg.norm(base_state.mean))
                    + float(np.linalg.norm(base_state.covariance, ord="fro"))
                )
            )
        rows.append(
            trajectory_row(
                config=config,
                method=method,
                seed=run_seed,
                iteration=iteration,
                step_size=actual_step,
                state=state,
                target=problem.target,
                optimum=reference.state,
                reference_objective=reference.objective,
                evaluation_engine=evaluation_engine,
                counts=counts,
                elapsed=time.perf_counter() - started,
                algorithm_elapsed=algorithm_elapsed,
                repair=repair,
                problem=problem,
                predictive_normals=predictive_normals,
                equivariance_error_mean=mean_error,
                equivariance_error_covariance=covariance_error,
                equivariance_error_back=back_error,
                local_gamma=local_gamma,
                local_lambda=local_lambda,
            )
        )
        rows[-1]["batch_size"] = batch_size
        rows[-1]["quadratic_rescue"] = rescue
        rows[-1]["raw_mean_ablation"] = bool(method_specification.get("raw_mean_ablation", False))
        rows[-1]["normalized_step_size"] = float(
            method_specification.get("normalized_step_size", actual_step)
        )
        rows[-1]["problem_instance"] = int(instance)
        rows[-1]["local_kl_gamma"] = float(local_kl_gamma)
        rows[-1]["local_discrete_rate"] = float(local_discrete_rate)
        rows[-1]["predicted_contraction"] = float(predicted_contraction)
        rows[-1]["local_radius"] = float(
            (config.get("local_initialization") or {}).get("radius", np.nan)
        )
        for key in ("alpha", "beta", "alpha_star", "beta_star", "kappa_star",
                    "lambda_0_star", "lambda_0_star_max", "lambda_max_star"):
            rows[-1][key] = float((curvature or {}).get(key, np.nan))
        rows[-1]["certified_gap"] = certified_gap_bound(
            fisher_rao_squared=float(rows[-1]["fisher_rao_residual_squared"]),
            bures_wasserstein_squared=float(rows[-1]["bures_wasserstein_residual_squared"]),
            alpha_star=float(rows[-1]["alpha_star"]),
            covariance_min_eigenvalue=float(rows[-1]["covariance_min_eigenvalue"]),
        )
        for key, value in config.get("grid", {}).items():
            rows[-1][f"grid_{key}"] = value

    record_every = max(1, int(config.get("record_every", 1)))
    add_row(0, 0.0, None)
    initial_objective = float(rows[0]["objective"])
    failure_reason = ""
    failure_iteration: int | None = None
    status = "completed"
    for iteration in range(iterations):
        if method == Method.LAPLACE:
            break
        if method_specification.get("schedule") == "manuscript_decreasing":
            kappa = float(method_specification.get("kappa_star", config.get("kappa_star", 1.0)))
            n0 = int(method_specification.get("n0", np.ceil(64.0 * kappa * kappa)))
            actual_step = 8.0 * kappa / (iteration + n0)
        else:
            actual_step = base_step_size
        try:
            update_started = time.perf_counter()
            state, diagnostics = step(
                method,
                problem.target,
                state,
                actual_step,
                engine=update_engine,
                rng=rng,
                batch_size=batch_size,
                counts=counts,
                raw_mean_ablation=bool(method_specification.get("raw_mean_ablation", False)),
            )
            algorithm_elapsed += time.perf_counter() - update_started
            if base_state is not None and base_target is not None:
                base_state, _ = step(
                    method,
                    base_target,
                    base_state,
                    actual_step,
                    engine=ExactGaussianExpectation(),
                    rng=base_rng,
                    batch_size=batch_size,
                    counts=base_counts,
                )
            if (iteration + 1) % record_every == 0 or iteration + 1 == iterations:
                add_row(iteration + 1, actual_step, diagnostics.repair)
                current_objective = float(rows[-1]["objective"])
                explosion_limit = max(
                    initial_objective + 1.0e6, 1.0e6 * max(abs(initial_objective), 1.0)
                )
                if not np.isfinite(current_objective) or current_objective > explosion_limit:
                    raise AlgorithmFailure(
                        f"explosive objective: {current_objective:.6e} > {explosion_limit:.6e}"
                    )
        except Exception as exc:
            status = "failed"
            failure_iteration = iteration
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

    write_rows(output_path, rows)
    burn_in = None
    if config["experiment"] == "A":
        # The manuscript's covariance bootstrap is stated in optimizer-whitened
        # coordinates: the band is C_n >= (2 beta_star)^{-1} C_star, i.e.
        # lambda_min(C_star^{-1/2} C_n C_star^{-1/2}) >= 1 / (2 beta_star).
        beta_star = float((curvature or {}).get("beta_star", 1.0))
        threshold = 1.0 / (2.0 * beta_star)
        for row in rows:
            if float(row["relative_covariance_min_eigenvalue"]) >= threshold:
                burn_in = int(row["iteration"])
                break
    summary = {
        "status": status,
        "failure_reason": failure_reason,
        "failure_iteration": failure_iteration,
        "iterations_completed": counts.iterations,
        "final_objective_gap": float(rows[-1]["objective_gap"]),
        "minimum_covariance_eigenvalue": min(float(row["covariance_min_eigenvalue"]) for row in rows),
        "burn_in_iteration": burn_in,
        "operation_counts": counts.to_dict(),
        "algorithm_seconds": algorithm_elapsed,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    return status, summary


def _run_variance_ablation(
    *,
    config: dict[str, Any],
    problem: BuiltProblem,
    reference: ReferenceSolution,
    run_seed: int,
    output_path: Path,
    evaluation_engine: ExpectationEngine,
) -> tuple[str, dict[str, Any]]:
    rng = np.random.default_rng(run_seed)
    batch_size = int(config.get("batch_size", 4))
    replicates = int(config.get("variance_replicates", 200))
    levels = np.asarray(config.get("interpolation_levels", [0.0, 0.5, 0.9, 1.0]), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, level in enumerate(levels):
        mean = (1.0 - level) * problem.initial_state.mean + level * reference.state.mean
        covariance = (1.0 - level) * problem.initial_state.covariance + level * reference.state.covariance
        state = GaussianState(mean, covariance)
        root = spd_sqrt(covariance)
        target_dimension = state.mean.size
        # Population quantities come from the cell's own evaluation engine --
        # exact for Gaussian targets, Gauss--Hermite for the separable ones -- so
        # the Lemma 4.7 comparison is not limited by quadrature error.
        evaluation = evaluation_engine.evaluate(problem.target, mean, covariance)
        g = -evaluation.grad
        population_hessian = np.asarray(evaluation.hessian, dtype=np.float64)
        certificate = residuals(state, evaluation)
        whitened_population = symmetrize_matrix(root @ population_hessian @ root)
        raw_errors: list[float] = []
        stl_errors: list[float] = []
        tangent_errors: list[float] = []
        fluctuations: list[float] = []
        for _ in range(replicates):
            normals = rng.standard_normal((batch_size, target_dimension), dtype=np.float64)
            samples = mean + normals @ root.T
            gradients = np.asarray(problem.target.grad(samples), dtype=np.float64)
            score = spd_solve(covariance, (samples - mean).T).T
            raw = np.mean(-gradients, axis=0)
            stl = np.mean(-gradients + score, axis=0)
            raw_error = raw - g
            stl_error = stl - g
            raw_errors.append(float(raw_error @ covariance @ raw_error))
            stl_errors.append(float(stl_error @ covariance @ stl_error))
            # Full Fisher--Rao tangent error of the stochastic gradient, and the
            # intrinsic Hessian fluctuation Psi of Definition 4.6, both evaluated
            # on the same batch so that Lemma 4.7 can be checked directly.
            sampled_hessian = mean_hessian(problem.target, samples)
            whitened_error = symmetrize_matrix(
                root @ (sampled_hessian - population_hessian) @ root
            )
            tangent_errors.append(
                float(stl_error @ covariance @ stl_error)
                + 0.5 * float(np.linalg.norm(whitened_error, ord="fro") ** 2)
            )
            single = samples[:1]
            single_hessian = mean_hessian(problem.target, single)
            single_whitened = symmetrize_matrix(
                root @ (single_hessian - population_hessian) @ root
            )
            fluctuations.append(float(np.linalg.norm(single_whitened, ord="fro") ** 2))
        relative = optimizer_relative(state, reference.state)
        raw_variance = float(np.mean(raw_errors))
        stl_variance = float(np.mean(stl_errors))
        psi = float(np.mean(fluctuations))
        gradient_norm_squared = float(certificate.fisher_rao_squared)
        measured_tangent_variance = float(np.mean(tangent_errors))
        # Lemma 4.7 bounds the single-sample tangent variance; averaging over a
        # batch of size B divides it by B.
        lemma_bound = (2.0 * gradient_norm_squared + 1.5 * psi) / batch_size
        rows.append(
            {
                "tier": config["tier"],
                "experiment": config["experiment"],
                "job_id": config["id"],
                "method": "raw-vs-STL-ablation",
                "seed": run_seed,
                "iteration": index,
                "interpolation_level": float(level),
                "optimizer_relative_distance": float(
                    np.hypot(relative.mean_norm, relative.covariance_frobenius)
                ),
                "raw_intrinsic_variance": raw_variance,
                "stl_intrinsic_variance": stl_variance,
                "stl_raw_variance_ratio": stl_variance / raw_variance if raw_variance > 0.0 else np.nan,
                "psi_hessian_fluctuation": psi,
                "fisher_rao_gradient_norm_squared": gradient_norm_squared,
                "measured_tangent_variance": measured_tangent_variance,
                "lemma_variance_bound": lemma_bound,
                "lemma_bound_slack": lemma_bound - measured_tangent_variance,
                **{f"grid_{key}": value for key, value in config.get("grid", {}).items()},
            }
        )
    write_rows(output_path, rows)
    return "completed", {
        "status": "completed",
        "rows": len(rows),
        "replicates": replicates,
        "batch_size": batch_size,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _fit_log_rate(iterations: np.ndarray, values: np.ndarray, step_size: float) -> float:
    """Least-squares decay rate ``r`` in ``value ~ exp(-r * t)`` with ``t = n h``."""

    iterations = np.asarray(iterations, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & (values > 0.0)
    if finite.sum() < 3:
        return float("nan")
    times = iterations[finite] * step_size
    slope = np.polyfit(times, np.log(values[finite]), 1)[0]
    return float(-slope)


def _run_affine_metric(
    *,
    config: dict[str, Any],
    problem: BuiltProblem,
    run_seed: int,
    output_path: Path,
) -> tuple[str, dict[str, Any]]:
    """Experiment M: modal rates of the Section 5 affine-invariant metric family."""

    target = problem.target
    if not isinstance(target, GaussianTarget):
        raise ValueError("the Section 5 modal-rate verification uses a Gaussian target")
    dimension = target.dimension
    engine = ExactGaussianExpectation()
    step_size = float(config.get("step_size", 0.02))
    iterations = int(config.get("iterations", 400))
    perturbation = float(config.get("perturbation", 1.0e-3))
    fit_fraction = float(config.get("fit_fraction", 0.5))

    rng = np.random.default_rng(run_seed)
    # A fixed traceless direction plus an isotropic direction, so both modes are
    # excited and can be separated along the trajectory.
    raw = rng.standard_normal((dimension, dimension))
    traceless = symmetrize_matrix(raw)
    traceless -= (np.trace(traceless) / dimension) * np.eye(dimension)
    traceless /= np.linalg.norm(traceless, ord="fro")
    isotropic = np.eye(dimension) / np.sqrt(dimension)
    root_target = spd_sqrt(target.covariance)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    failures = 0
    for entry in config.get("metrics", []):
        omega = float(entry["omega"])
        tau = float(entry["tau"])
        metric = AffineMetric(omega, tau, dimension)
        # Whitened perturbation transported to the target's coordinates, so the
        # experiment is run in genuinely non-standard coordinates while the
        # predicted rates remain the optimizer-whitened ones.
        whitened = np.eye(dimension) + perturbation * (traceless + isotropic)
        state = GaussianState(
            target.mean + perturbation * root_target @ np.ones(dimension) / np.sqrt(dimension),
            symmetrize_matrix(root_target @ whitened @ root_target),
        )
        history: list[tuple[int, float, float, float]] = []
        status_entry = "completed"
        for iteration in range(iterations + 1):
            relative = optimizer_relative(state, GaussianState(target.mean, target.covariance))
            whitened_covariance = np.linalg.solve(
                root_target, np.linalg.solve(root_target, state.covariance).T
            ).T
            traceless_norm, trace_value = modal_decomposition(whitened_covariance)
            history.append((iteration, traceless_norm, abs(trace_value), relative.mean_norm))
            if iteration == iterations:
                break
            try:
                state = retraction_step(metric, target, state, step_size, engine=engine)
            except AlgorithmFailure as exc:
                status_entry = f"failed: {exc}"
                failures += 1
                break

        array = np.asarray(history, dtype=np.float64)
        window = max(3, int(fit_fraction * array.shape[0]))
        tail = array[:window]
        fitted_traceless = _fit_log_rate(tail[:, 0], tail[:, 1], step_size)
        fitted_trace = _fit_log_rate(tail[:, 0], tail[:, 2], step_size)
        fitted_mean = _fit_log_rate(tail[:, 0], tail[:, 3], step_size)
        for iteration, traceless_norm, trace_value, mean_norm in history:
            rows.append(
                {
                    "tier": config["tier"],
                    "experiment": config["experiment"],
                    "job_id": config["id"],
                    "method": f"affine-metric-omega{omega:g}-tau{tau:g}",
                    "seed": run_seed,
                    "iteration": int(iteration),
                    "step_size": step_size,
                    "omega": omega,
                    "tau": tau,
                    "dimension": dimension,
                    "traceless_norm": traceless_norm,
                    "trace_absolute": trace_value,
                    "mean_norm": mean_norm,
                    "predicted_traceless_rate": metric.traceless_rate,
                    "predicted_trace_rate": metric.trace_rate,
                    "fitted_traceless_rate": fitted_traceless,
                    "fitted_trace_rate": fitted_trace,
                    "fitted_mean_rate": fitted_mean,
                    "is_fisher_rao": metric.is_fisher_rao,
                    "status": status_entry,
                    **{f"grid_{key}": value for key, value in config.get("grid", {}).items()},
                }
            )
    write_rows(output_path, rows)
    return ("failed" if failures else "completed"), {
        "status": "failed" if failures else "completed",
        "metrics": len(config.get("metrics", [])),
        "rows": len(rows),
        "failures": failures,
        "wall_time_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _manifest_base(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_digest: str,
    source_digest: str,
    target_seed: int,
    run_seed: int,
    output_path: Path,
) -> dict[str, Any]:
    commit, dirty, status = git_metadata()
    return {
        "schema_version": 1,
        "config": config,
        "config_path": str(config_path),
        "config_hash": config_digest,
        "code_hash": source_digest,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status": status,
        "python": sys.version,
        "packages": package_versions(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "blas": blas_information(),
        "target_seed": target_seed,
        "run_seed": run_seed,
        "reference_hashes": reference_hashes(),
        "command": " ".join(sys.argv),
        "output_paths": [str(output_path)],
    }


@dataclass(frozen=True)
class _PreparedInstance:
    problem: BuiltProblem
    update_engine: ExpectationEngine
    evaluation_engine: ExpectationEngine
    reference: ReferenceSolution
    reference_path: Path
    target_seed: int
    curvature: dict[str, Any]


def run_config(
    config_path: Path,
    config: dict[str, Any],
    *,
    state: dict[str, Any],
    force: bool,
    deadline: float,
) -> dict[str, int]:
    counts = {"completed": 0, "failed": 0, "skipped": 0, "interrupted": 0, "pending": 0}
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config_digest = hash_bytes(config_bytes)
    source_digest = code_hash()
    master_seed = int(config.get("master_seed", 20260805))
    instance_per_seed = bool(config.get("instance_per_seed", False))
    prepared: dict[int, _PreparedInstance] = {}

    def prepare(instance: int) -> _PreparedInstance:
        """Build the target, engines and certified reference for one instance.

        With ``instance_per_seed`` the target itself is redrawn per repeat, so a
        grid cell aggregates over random problem instances rather than over
        algorithm randomness alone.
        """

        if instance in prepared:
            return prepared[instance]
        target_seed = seed_for(master_seed, 0, instance)
        problem = build_problem(config["target"], target_seed, str(config["experiment"]))
        update_engine, evaluation_engine = engines(
            problem.target,
            update_points=int(config.get("update_points", 256)),
            evaluation_points=int(config.get("evaluation_points", 1024)),
            update_seed=seed_for(master_seed, 1, instance),
            evaluation_seed=seed_for(master_seed, 2, instance),
        )
        reference = solve_reference(
            problem.target,
            problem.initial_state,
            points=int(
                config.get("reference_points", max(1024, int(config.get("evaluation_points", 1024))))
            ),
            seed=seed_for(master_seed, 3, instance),
            engine=evaluation_engine if isinstance(evaluation_engine, FixedNormalExpectation) else None,
        )
        # Re-evaluate the reference on the *evaluation* engine, the same one every
        # trajectory objective is measured with, so a gap is a difference of two
        # numbers produced by one rule.  For a quadrature-based reference the
        # residuals are recomputed there too, since carrying over the solver's
        # residuals would certify it against a rule no reported quantity uses.
        #
        # An analytic reference is left alone.  For a Gaussian target the optimizer
        # is the target itself and its suboptimality is exactly zero by
        # construction; re-measuring that zero through the residual formula puts a
        # whitening congruence of condition number 1e9 in the way and returns 1e-8,
        # which is the conditioning of the diagnostic and not a property of the
        # reference.
        reference_objective, reference_expectation = objective(
            problem.target, reference.state, evaluation_engine
        )
        if reference.metadata.get("analytic"):
            fisher_rao_squared = reference.fisher_rao_residual_squared
            bures_squared = reference.bures_wasserstein_residual_squared
            extra: dict[str, Any] = {}
        else:
            reference_certificate = residuals(reference.state, reference_expectation)
            fisher_rao_squared = reference_certificate.fisher_rao_squared
            bures_squared = reference_certificate.bures_wasserstein_squared
            extra = {
                "solver_fisher_rao_residual_squared": reference.fisher_rao_residual_squared,
            }
        reference = ReferenceSolution(
            reference.state,
            reference_objective,
            fisher_rao_squared,
            bures_squared,
            {
                **reference.metadata,
                **extra,
                "evaluation_backend": type(evaluation_engine).__name__,
            },
        )
        suffix = f"_instance{instance}" if instance_per_seed else ""
        reference_path = RESULTS / "manifests" / f"reference_{config['id']}{suffix}.json"
        save_json(
            reference_path,
            {
                "job_id": config["id"],
                "instance": instance,
                "mean": reference.state.mean.tolist(),
                "covariance": reference.state.covariance.tolist(),
                "objective": reference.objective,
                "fisher_rao_residual_squared": reference.fisher_rao_residual_squared,
                "bures_wasserstein_residual_squared": reference.bures_wasserstein_residual_squared,
                "metadata": reference.metadata,
            },
        )
        constants = curvature_constants(problem.target, reference.state)
        whitened = whitened_initialization(
            problem.initial_state, reference.state, constants.alpha_star
        )
        curvature = {
            **constants.to_dict(),
            **whitened.to_dict(),
            "certified_step_sizes": certified_step_sizes(constants, whitened),
        }
        prepared[instance] = _PreparedInstance(
            problem,
            update_engine,
            evaluation_engine,
            reference,
            reference_path,
            target_seed,
            curvature,
        )
        return prepared[instance]

    target_seed = seed_for(master_seed, 0, 0)

    if config.get("blocked_reason"):
        key = f"{config['id']}:blocked"
        manifest_path = RESULTS / "manifests" / config["tier"] / f"{config['id']}_blocked.json"
        manifest = _manifest_base(
            config=config,
            config_path=config_path,
            config_digest=config_digest,
            source_digest=source_digest,
            target_seed=target_seed,
            run_seed=master_seed,
            output_path=manifest_path,
        )
        manifest.update(
            {
                "start_utc": utc_now(),
                "end_utc": utc_now(),
                "status": "skipped",
                "failure_reason": config["blocked_reason"],
            }
        )
        save_json(manifest_path, manifest)
        state["runs"][key] = {"status": "skipped", "manifest": str(manifest_path)}
        save_state(state)
        counts["skipped"] += 1
        return counts

    if config["experiment"] == "I":
        specifications = [{"name": "variance-ablation", "seeds": int(config.get("seeds", 1))}]
    elif config["experiment"] == "M":
        specifications = [{"name": "affine-metric-family", "seeds": int(config.get("seeds", 1))}]
    else:
        specifications = list(config.get("methods", []))
    for specification_index, specification in enumerate(specifications):
        repeat_count = int(specification.get("seeds", config.get("seeds", 1)))
        for repeat in range(repeat_count):
            if time.monotonic() >= deadline:
                counts["pending"] += 1
                continue
            run_seed = seed_for(master_seed, 10, repeat)
            if config["experiment"] == "I":
                slug = "raw-vs-stl"
            elif config["experiment"] == "M":
                slug = "affine-metric-family"
            else:
                method = Method(specification["name"])
                slug = _method_slug(method, specification)
            run_id = f"{config['id']}:{slug}:seed{repeat}"
            output_path = (
                RESULTS / "raw" / config["tier"] / str(config["experiment"]) / config["id"] / f"{slug}_seed{repeat}.csv"
            )
            manifest_path = (
                RESULTS / "manifests" / config["tier"] / f"{config['id']}_{slug}_seed{repeat}.json"
            )
            previous = state["runs"].get(run_id)
            if (
                not force
                and previous
                and previous.get("status") == "completed"
                and previous.get("config_hash") == config_digest
                and previous.get("code_hash") == source_digest
                and output_path.exists()
            ):
                counts["skipped"] += 1
                continue
            instance = repeat if instance_per_seed else 0
            prepared_instance = prepare(instance)
            problem = prepared_instance.problem
            reference = prepared_instance.reference
            reference_path = prepared_instance.reference_path
            update_engine = prepared_instance.update_engine
            evaluation_engine = prepared_instance.evaluation_engine
            manifest = _manifest_base(
                config=config,
                config_path=config_path,
                config_digest=config_digest,
                source_digest=source_digest,
                target_seed=prepared_instance.target_seed,
                run_seed=run_seed,
                output_path=output_path,
            )
            manifest.update(
                {
                    "method_specification": specification,
                    "problem_instance": instance,
                    "problem_metadata": problem.metadata,
                    "curvature": prepared_instance.curvature,
                    "reference": {
                        "path": str(reference_path),
                        "objective": reference.objective,
                        "fisher_rao_residual_squared": reference.fisher_rao_residual_squared,
                        "bures_wasserstein_residual_squared": reference.bures_wasserstein_residual_squared,
                    },
                    "start_utc": utc_now(),
                    "status": "running",
                }
            )
            save_json(manifest_path, manifest)
            state["runs"][run_id] = {
                "status": "running",
                "manifest": str(manifest_path),
                "config_hash": config_digest,
                "code_hash": source_digest,
            }
            save_state(state)
            try:
                if config["experiment"] == "I":
                    status_name, summary = _run_variance_ablation(
                        config=config,
                        problem=problem,
                        reference=reference,
                        run_seed=run_seed,
                        output_path=output_path,
                        evaluation_engine=evaluation_engine,
                    )
                elif config["experiment"] == "M":
                    status_name, summary = _run_affine_metric(
                        config=config,
                        problem=problem,
                        run_seed=run_seed,
                        output_path=output_path,
                    )
                else:
                    status_name, summary = _run_trajectory(
                        config=config,
                        problem=problem,
                        reference=reference,
                        method_specification=specification,
                        run_seed=run_seed,
                        update_engine=update_engine,
                        evaluation_engine=evaluation_engine,
                        output_path=output_path,
                        instance=instance,
                        curvature=prepared_instance.curvature,
                    )
            except KeyboardInterrupt:
                status_name = "interrupted"
                summary = {"status": status_name, "failure_reason": "KeyboardInterrupt"}
            except Exception as exc:
                status_name = "failed"
                summary = {
                    "status": status_name,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            manifest.update(summary)
            manifest["status"] = status_name
            manifest["end_utc"] = utc_now()
            save_json(manifest_path, manifest)
            state["runs"][run_id].update(
                {"status": status_name, "config_hash": config_digest, "code_hash": source_digest}
            )
            save_state(state)
            counts[status_name] = counts.get(status_name, 0) + 1
    return counts


SHARD_DIRECTORY = RESULTS / "manifests" / "state_shards"


def _shard_path(config_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in config_id)
    return SHARD_DIRECTORY / f"{safe}.json"


def _worker(payload: tuple[str, dict[str, Any], bool, float]) -> tuple[str, dict[str, int], str]:
    """Run one config in its own process against a private state shard.

    Each worker owns exactly one shard file, so no two processes ever write the
    same JSON.  The parent merges the shards into ``campaign_state.json``.
    """

    config_path_text, config, force, seconds_remaining = payload
    global STATE_PATH
    STATE_PATH = _shard_path(str(config["id"]))
    state = load_state()
    try:
        counts = run_config(
            Path(config_path_text),
            config,
            state=state,
            force=force,
            deadline=time.monotonic() + max(0.0, seconds_remaining),
        )
        return str(config["id"]), counts, ""
    except Exception as exc:  # pragma: no cover - defensive; surfaced to the parent
        return (
            str(config["id"]),
            {"completed": 0, "failed": 1, "skipped": 0, "interrupted": 0, "pending": 0},
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _merge_shards(state: dict[str, Any]) -> dict[str, Any]:
    for shard in sorted(SHARD_DIRECTORY.glob("*.json")):
        try:
            shard_state = json.loads(shard.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        state.setdefault("runs", {}).update(shard_state.get("runs", {}))
    return state


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="JSON config files or directories")
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of worker processes; each pins BLAS to a single thread",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    budget = args.budget_hours
    if budget is None:
        budget = float(os.environ.get("OVERNIGHT_BUDGET_HOURS", "10"))
    deadline = time.monotonic() + max(0.0, budget) * 3600.0
    state = load_state()
    aggregate = {"completed": 0, "failed": 0, "skipped": 0, "interrupted": 0, "pending": 0}
    configs = load_configs(args.configs)

    if args.jobs > 1:
        SHARD_DIRECTORY.mkdir(parents=True, exist_ok=True)
        payloads = [
            (str(path), config, bool(args.force), max(0.0, deadline - time.monotonic()))
            for path, config in configs
        ]
        errors: list[str] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for config_id, counts, error in pool.map(_worker, payloads):
                for key, value in counts.items():
                    aggregate[key] = aggregate.get(key, 0) + value
                if error:
                    errors.append(f"{config_id}: {error}")
        state = _merge_shards(state)
        for error in errors:
            print(error, file=sys.stderr)
    else:
        for config_path, config in configs:
            if time.monotonic() >= deadline:
                aggregate["pending"] += 1
                continue
            outcome = run_config(
                config_path,
                config,
                state=state,
                force=bool(args.force),
                deadline=deadline,
            )
            for key, value in outcome.items():
                aggregate[key] = aggregate.get(key, 0) + value

    state["last_campaign"] = {"end_utc": utc_now(), "summary": aggregate, "command": " ".join(sys.argv)}
    save_state(state)
    print(json.dumps(aggregate, sort_keys=True))
    return 1 if aggregate["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

