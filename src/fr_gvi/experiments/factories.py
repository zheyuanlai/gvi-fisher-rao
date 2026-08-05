from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.targets.core import GaussianTarget, LogisticRegressionTarget, ShiftedLogCoshTarget, Target

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BuiltProblem:
    target: Target
    initial_state: GaussianState
    metadata: dict[str, Any]
    heldout: tuple[FloatArray, FloatArray] | None = None


def random_orthogonal(dimension: int, rng: np.random.Generator) -> FloatArray:
    matrix = rng.standard_normal((dimension, dimension), dtype=np.float64)
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return np.asarray(q * signs, dtype=np.float64)


def conditioned_map(dimension: int, condition: float, rng: np.random.Generator) -> FloatArray:
    left = random_orthogonal(dimension, rng)
    right = random_orthogonal(dimension, rng)
    singular_values = np.geomspace(1.0, condition, dimension)
    return np.asarray((left * singular_values) @ right.T, dtype=np.float64)


def _gaussian(config: dict[str, Any], rng: np.random.Generator) -> BuiltProblem:
    dimension = int(config.get("dimension", 2))
    condition = float(config.get("condition", 10.0))
    if config.get("diao_spectrum", False):
        eigenvalues = np.geomspace(1.0e-9, 1.0, dimension)
    else:
        eigenvalues = np.geomspace(1.0 / condition, 1.0, dimension)
    rotation = random_orthogonal(dimension, rng) if config.get("rotation", True) else np.eye(dimension)
    precision = (rotation * eigenvalues) @ rotation.T
    if config.get("mean_distribution") == "uniform":
        mean = rng.uniform(0.0, 1.0, size=dimension).astype(np.float64)
    else:
        mean_scale = float(config.get("mean_scale", 0.5))
        mean = mean_scale * rng.standard_normal(dimension, dtype=np.float64)
    target = GaussianTarget(mean, precision)
    covariance_scale = float(config.get("initial_covariance_scale", 1.0))
    initial_mean = np.zeros(dimension, dtype=np.float64)
    if "initial_mean_scale" in config:
        initial_mean = float(config["initial_mean_scale"]) * np.ones(dimension, dtype=np.float64)
    if config.get("initial_covariance") == "target":
        initial_covariance = target.covariance
    elif "initial_covariance_diagonal" in config:
        diagonal = np.asarray(config["initial_covariance_diagonal"], dtype=np.float64)
        if diagonal.shape != (dimension,):
            raise ValueError("initial_covariance_diagonal must have length d")
        initial_covariance = np.diag(diagonal)
    else:
        initial_covariance = covariance_scale * np.eye(dimension, dtype=np.float64)
    initial = GaussianState(initial_mean, initial_covariance)
    return BuiltProblem(
        target,
        initial,
        {
            "precision_min": float(eigenvalues[0]),
            "precision_max": float(eigenvalues[-1]),
            "condition": float(eigenvalues[-1] / eigenvalues[0]),
        },
    )


def _logcosh(config: dict[str, Any], rng: np.random.Generator) -> BuiltProblem:
    dimension = int(config.get("dimension", 2))
    condition = float(config.get("condition", 10.0))
    nu = np.geomspace(1.0 / condition, 1.0, dimension).astype(np.float64)
    rho = float(config.get("rho", 1.0))
    offset = np.linspace(-0.5, 0.5, dimension, dtype=np.float64)
    affine_condition = float(config.get("affine_condition", 1.0))
    transform = conditioned_map(dimension, affine_condition, rng)
    shift = float(config.get("shift", 0.2)) * rng.standard_normal(dimension, dtype=np.float64)
    target = ShiftedLogCoshTarget(nu, rho, offset, transform, shift)
    initial_mean = np.zeros(dimension, dtype=np.float64)
    covariance_scale = float(config.get("initial_covariance_scale", 0.5))
    initial = GaussianState(initial_mean, covariance_scale * np.eye(dimension, dtype=np.float64))
    return BuiltProblem(
        target,
        initial,
        {
            "base_curvature_min": float(nu.min()),
            "base_curvature_max": float(nu.max() + rho),
            "base_condition_bound": float((nu.max() + rho) / nu.min()),
            "affine_condition": affine_condition,
        },
    )


def _logistic(config: dict[str, Any], rng: np.random.Generator) -> BuiltProblem:
    dimension = int(config.get("dimension", 2))
    samples = int(config.get("samples", 10 * dimension))
    feature_condition = float(config.get("feature_condition", 10.0))
    eigenvalues = np.geomspace(1.0 / feature_condition, 1.0, dimension)
    rotation = random_orthogonal(dimension, rng)
    feature_root = rotation * np.sqrt(eigenvalues)
    features = rng.standard_normal((samples, dimension), dtype=np.float64) @ feature_root.T
    true_parameter = rng.standard_normal(dimension, dtype=np.float64) / np.sqrt(dimension)
    logits = features @ true_parameter
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    test_samples = int(config.get("test_samples", max(100, samples)))
    heldout_features = rng.standard_normal((test_samples, dimension), dtype=np.float64) @ feature_root.T
    heldout_logits = heldout_features @ true_parameter
    heldout_labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-heldout_logits))).astype(np.float64)
    prior_precision = float(config.get("prior_precision", 1.0))
    target = LogisticRegressionTarget(features, labels, prior_precision)
    initial = GaussianState(np.zeros(dimension), np.eye(dimension))
    return BuiltProblem(
        target,
        initial,
        {
            "samples": samples,
            "feature_condition": feature_condition,
            "prior_precision": prior_precision,
            "true_parameter_norm": float(np.linalg.norm(true_parameter)),
        },
        (heldout_features, heldout_labels),
    )


def _affine_equivariance(config: dict[str, Any], rng: np.random.Generator) -> BuiltProblem:
    dimension = int(config.get("dimension", 3))
    affine_condition = float(config.get("affine_condition", 100.0))
    base_mean = np.linspace(-0.2, 0.3, dimension, dtype=np.float64)
    base_precision = np.diag(np.linspace(0.7, 1.3, dimension, dtype=np.float64))
    base_initial_mean = np.linspace(0.5, -0.3, dimension, dtype=np.float64)
    base_initial_covariance = np.diag(np.linspace(0.4, 1.6, dimension, dtype=np.float64))
    transform = conditioned_map(dimension, affine_condition, rng)
    shift = rng.standard_normal(dimension, dtype=np.float64)
    inverse = np.linalg.solve(transform, np.eye(dimension))
    target_mean = transform @ base_mean + shift
    target_precision = inverse.T @ base_precision @ inverse
    target = GaussianTarget(target_mean, target_precision)
    initial = GaussianState(
        transform @ base_initial_mean + shift,
        transform @ base_initial_covariance @ transform.T,
    )
    return BuiltProblem(
        target,
        initial,
        {
            "affine_condition": affine_condition,
            "transform": transform.tolist(),
            "shift": shift.tolist(),
            "base_mean": base_mean.tolist(),
            "base_precision": base_precision.tolist(),
            "base_initial_mean": base_initial_mean.tolist(),
            "base_initial_covariance": base_initial_covariance.tolist(),
        },
    )


def build_problem(config: dict[str, Any], seed: int, experiment: str) -> BuiltProblem:
    rng = np.random.default_rng(seed)
    kind = str(config.get("kind", "gaussian"))
    if experiment == "B":
        return _affine_equivariance(config, rng)
    if kind == "gaussian":
        return _gaussian(config, rng)
    if kind == "logcosh":
        return _logcosh(config, rng)
    if kind == "logistic":
        return _logistic(config, rng)
    raise ValueError(f"unknown target kind: {kind}")

