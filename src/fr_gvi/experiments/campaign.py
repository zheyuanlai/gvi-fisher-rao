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
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit

from fr_gvi.algorithms.core import (
    AlgorithmFailure,
    GaussianState,
    Method,
    quadratic_rescue,
    step,
)
from fr_gvi.diagnostics.core import (
    gaussian_kl_gap,
    gaussian_w2_squared,
    objective,
    optimizer_relative,
    residuals,
)
from fr_gvi.diagnostics.local_operator import assemble_local_operator
from fr_gvi.expectations.core import (
    ExactGaussianExpectation,
    ExpectationEngine,
    FixedNormalExpectation,
    GaussHermiteLogCoshExpectation,
)
from fr_gvi.experiments.factories import BuiltProblem, build_problem
from fr_gvi.experiments.reference import ReferenceSolution, laplace_approximation, solve_reference
from fr_gvi.linear_algebra.spd import spd_solve, spd_sqrt
from fr_gvi.targets.core import GaussianTarget, LogisticRegressionTarget, ShiftedLogCoshTarget, Target
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


def code_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.sh"))
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


def git_metadata() -> tuple[str, bool, str]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD") or "unborn"
    status = run("status", "--short")
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
            files.extend(sorted(path.glob("*.json")))
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
        return GaussHermiteLogCoshExpectation(32), GaussHermiteLogCoshExpectation(80)
    return (
        FixedNormalExpectation.qmc(target.dimension, update_points, update_seed),
        FixedNormalExpectation.qmc(target.dimension, evaluation_points, evaluation_seed),
    )


def _method_slug(method: Method, specification: dict[str, Any]) -> str:
    slug = method.value.lower().replace("--", "-").replace("+", "-plus-")
    if specification.get("quadratic_rescue", False):
        slug += "-qr"
    if specification.get("raw_mean_ablation", False):
        slug += "-raw"
    if "batch_size" in specification:
        slug += f"-b{int(specification['batch_size'])}"
    return slug


def _predictive_metrics(
    problem: BuiltProblem,
    state: GaussianState,
    normals: np.ndarray,
) -> dict[str, float]:
    if problem.heldout is None:
        return {"predictive_nll": np.nan, "classification_error": np.nan, "brier": np.nan}
    features, labels = problem.heldout
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
    equivariance_error_mean: float = np.nan,
    equivariance_error_covariance: float = np.nan,
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
        "fisher_rao_residual_squared": certificate.fisher_rao_squared,
        "bures_wasserstein_residual_squared": certificate.bures_wasserstein_squared,
        "wall_time_seconds": elapsed,
        "repair": json.dumps(repair, sort_keys=True) if repair else "",
        "equivariance_error_mean": equivariance_error_mean,
        "equivariance_error_covariance": equivariance_error_covariance,
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
) -> tuple[str, dict[str, Any]]:
    method = Method(method_specification["name"])
    iterations = int(method_specification.get("iterations", config.get("iterations", 50)))
    batch_size = int(method_specification.get("batch_size", config.get("batch_size", 1)))
    base_step_size = float(method_specification.get("step_size", config.get("step_size", 0.1)))
    state = problem.initial_state
    rescue = bool(method_specification.get("quadratic_rescue", False))
    if rescue:
        state = quadratic_rescue(problem.target, state.mean)
    if method == Method.LAPLACE:
        if not isinstance(problem.target, LogisticRegressionTarget):
            raise ValueError("Laplace is only an approximation-quality baseline for logistic regression")
        state = laplace_approximation(problem.target)
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
    predictive_normals = FixedNormalExpectation.qmc(
        state.mean.size, min(512, int(config.get("evaluation_points", 1024))), run_seed + 17
    ).normals

    local_gamma = np.nan
    local_lambda = np.nan
    if config["experiment"] == "G":
        local_normals = FixedNormalExpectation.qmc(
            state.mean.size, int(config.get("local_operator_points", 2048)), run_seed + 37
        ).normals
        operator = assemble_local_operator(problem.target, reference.state, local_normals)
        spectrum = np.linalg.eigvalsh(operator)
        local_gamma, local_lambda = float(spectrum[0]), float(spectrum[-1])

    base_target = None
    base_state = None
    transform = None
    shift = None
    base_rng = np.random.default_rng(run_seed)
    base_counts = OperationCounts()
    if config["experiment"] == "B":
        base_target, base_state, transform, shift = _base_equivariance_problem(problem)

    def add_row(iteration: int, actual_step: float, repair: dict[str, float] | None) -> None:
        mean_error = np.nan
        covariance_error = np.nan
        if base_state is not None and transform is not None and shift is not None:
            expected_mean = transform @ base_state.mean + shift
            expected_covariance = transform @ base_state.covariance @ transform.T
            denominator_mean = max(float(np.linalg.norm(expected_mean)), 1.0)
            denominator_covariance = max(float(np.linalg.norm(expected_covariance, ord="fro")), 1.0)
            mean_error = float(np.linalg.norm(state.mean - expected_mean) / denominator_mean)
            covariance_error = float(
                np.linalg.norm(state.covariance - expected_covariance, ord="fro") / denominator_covariance
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
                repair=repair,
                problem=problem,
                predictive_normals=predictive_normals,
                equivariance_error_mean=mean_error,
                equivariance_error_covariance=covariance_error,
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
            add_row(iteration + 1, actual_step, diagnostics.repair)
            current_objective = float(rows[-1]["objective"])
            explosion_limit = max(initial_objective + 1.0e6, 1.0e6 * max(abs(initial_objective), 1.0))
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
    if config["experiment"] == "A" and isinstance(problem.target, GaussianTarget):
        threshold = 1.0 / (2.0 * float(np.linalg.eigvalsh(problem.target.precision)[-1]))
        for row in rows:
            if float(row["covariance_min_eigenvalue"]) >= threshold:
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
        evaluation = FixedNormalExpectation.qmc(
            target_dimension := state.mean.size,
            max(1024, int(config.get("evaluation_points", 1024))),
            run_seed + index + 100,
        ).evaluate(problem.target, mean, covariance)
        g = -evaluation.grad
        raw_errors: list[float] = []
        stl_errors: list[float] = []
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
        relative = optimizer_relative(state, reference.state)
        raw_variance = float(np.mean(raw_errors))
        stl_variance = float(np.mean(stl_errors))
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
    target_seed = seed_for(master_seed, 0)
    problem = build_problem(config["target"], target_seed, str(config["experiment"]))
    update_seed = seed_for(master_seed, 1)
    evaluation_seed = seed_for(master_seed, 2)
    update_engine, evaluation_engine = engines(
        problem.target,
        update_points=int(config.get("update_points", 256)),
        evaluation_points=int(config.get("evaluation_points", 1024)),
        update_seed=update_seed,
        evaluation_seed=evaluation_seed,
    )
    reference = solve_reference(
        problem.target,
        problem.initial_state,
        points=int(config.get("reference_points", max(1024, int(config.get("evaluation_points", 1024))))),
        seed=seed_for(master_seed, 3),
    )
    reference_objective, _ = objective(problem.target, reference.state, evaluation_engine)
    reference = ReferenceSolution(
        reference.state,
        reference_objective,
        reference.fisher_rao_residual_squared,
        reference.bures_wasserstein_residual_squared,
        reference.metadata,
    )
    reference_path = RESULTS / "manifests" / f"reference_{config['id']}.json"
    save_json(
        reference_path,
        {
            "job_id": config["id"],
            "mean": reference.state.mean.tolist(),
            "covariance": reference.state.covariance.tolist(),
            "objective": reference.objective,
            "fisher_rao_residual_squared": reference.fisher_rao_residual_squared,
            "bures_wasserstein_residual_squared": reference.bures_wasserstein_residual_squared,
            "metadata": reference.metadata,
        },
    )

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
            manifest = _manifest_base(
                config=config,
                config_path=config_path,
                config_digest=config_digest,
                source_digest=source_digest,
                target_seed=target_seed,
                run_seed=run_seed,
                output_path=output_path,
            )
            manifest.update(
                {
                    "method_specification": specification,
                    "problem_metadata": problem.metadata,
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


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="JSON config files or directories")
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    budget = args.budget_hours
    if budget is None:
        budget = float(os.environ.get("OVERNIGHT_BUDGET_HOURS", "10"))
    deadline = time.monotonic() + max(0.0, budget) * 3600.0
    state = load_state()
    aggregate = {"completed": 0, "failed": 0, "skipped": 0, "interrupted": 0, "pending": 0}
    for config_path, config in load_configs(args.configs):
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

