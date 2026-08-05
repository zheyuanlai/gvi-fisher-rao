"""Reference Gaussian-VI solutions and non-iterative Laplace baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.diagnostics.core import objective, residuals
from fr_gvi.expectations.core import FixedNormalExpectation, GaussHermiteLogCoshExpectation
from fr_gvi.linear_algebra.spd import spd_inverse
from fr_gvi.targets.core import GaussianTarget, LogisticRegressionTarget, ShiftedLogCoshTarget, Target

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceSolution:
    state: GaussianState
    objective: float
    fisher_rao_residual_squared: float
    bures_wasserstein_residual_squared: float
    metadata: dict[str, float | int | str | bool]


def _logcosh_reference(target: ShiftedLogCoshTarget, order: int) -> GaussianState:
    nodes, weights = hermgauss(order)
    nodes = np.sqrt(2.0) * nodes
    weights = weights / np.sqrt(np.pi)
    means = np.zeros(target.dimension, dtype=np.float64)
    variances = np.ones(target.dimension, dtype=np.float64)
    for index in range(target.dimension):
        nu = target.nu[index]
        rho = target.rho
        offset = target.offset[index]

        def conditions(parameters: FloatArray) -> FloatArray:
            mean = parameters[0]
            variance = np.exp(parameters[1])
            samples = mean + np.sqrt(variance) * nodes
            tanh = np.tanh(samples - offset)
            expected_gradient = float(weights @ (nu * samples + rho * tanh))
            expected_hessian = float(weights @ (nu + rho * (1.0 - tanh**2)))
            return np.asarray([expected_gradient, variance * expected_hessian - 1.0])

        solution = least_squares(conditions, np.asarray([0.0, -np.log(nu + rho)]), xtol=1e-13, ftol=1e-13, gtol=1e-13)
        if not solution.success or np.linalg.norm(solution.fun) > 1.0e-9:
            raise RuntimeError(f"log-cosh reference solve failed in coordinate {index}: {solution.message}")
        means[index] = solution.x[0]
        variances[index] = np.exp(solution.x[1])
    mean = target.transform @ means + target.shift
    covariance = target.transform @ np.diag(variances) @ target.transform.T
    return GaussianState(mean, covariance)


def _parameter_to_state(parameters: FloatArray, dimension: int) -> GaussianState:
    mean = parameters[:dimension]
    lower = np.zeros((dimension, dimension), dtype=np.float64)
    cursor = dimension
    for row in range(dimension):
        for column in range(row + 1):
            value = parameters[cursor]
            lower[row, column] = np.exp(value) if row == column else value
            cursor += 1
    return GaussianState(mean, lower @ lower.T)


def _state_to_parameter(state: GaussianState) -> FloatArray:
    dimension = state.mean.size
    lower = np.linalg.cholesky(state.covariance)
    values = list(state.mean)
    for row in range(dimension):
        for column in range(row + 1):
            values.append(float(np.log(lower[row, column])) if row == column else float(lower[row, column]))
    return np.asarray(values, dtype=np.float64)


def _qmc_reference(
    target: Target,
    initial: GaussianState,
    *,
    points: int,
    seed: int,
) -> GaussianState:
    dimension = initial.mean.size
    engine = FixedNormalExpectation.qmc(dimension, points, seed)

    def value_gradient(parameters: FloatArray) -> tuple[float, FloatArray]:
        state = _parameter_to_state(parameters, dimension)
        lower = np.linalg.cholesky(state.covariance)
        samples = state.mean + engine.normals @ lower.T
        values = np.asarray(target.value(samples), dtype=np.float64)
        gradients = np.asarray(target.grad(samples), dtype=np.float64)
        objective_value = float(np.mean(values) - np.log(np.diag(lower)).sum())
        mean_gradient = np.mean(gradients, axis=0)
        lower_gradient = np.einsum("si,sj->ij", gradients, engine.normals) / points
        parameter_gradient = list(mean_gradient)
        for row in range(dimension):
            for column in range(row + 1):
                if row == column:
                    parameter_gradient.append(float(lower_gradient[row, column] * lower[row, row] - 1.0))
                else:
                    parameter_gradient.append(float(lower_gradient[row, column]))
        return objective_value, np.asarray(parameter_gradient, dtype=np.float64)

    start = _state_to_parameter(initial)
    candidates = [start]
    candidates.append(start + np.random.default_rng(seed + 1).normal(0.0, 0.03, start.size))
    best = None
    for candidate in candidates:
        result = minimize(
            value_gradient,
            candidate,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 400, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    if not best.success and np.linalg.norm(best.jac) > 2.0e-5:
        raise RuntimeError(f"QMC Gaussian-VI reference failed: {best.message}; grad={np.linalg.norm(best.jac):.3e}")
    return _parameter_to_state(np.asarray(best.x, dtype=np.float64), dimension)


def solve_reference(
    target: Target,
    initial: GaussianState,
    *,
    points: int,
    seed: int,
) -> ReferenceSolution:
    if isinstance(target, GaussianTarget):
        state = GaussianState(target.mean, target.covariance)
        engine = FixedNormalExpectation.qmc(target.dimension, max(points, 256), seed + 101)
        objective_value, expectation = objective(target, state, engine)
        return ReferenceSolution(state, objective_value, 0.0, 0.0, {"backend": "exact"})
    if isinstance(target, ShiftedLogCoshTarget):
        order = max(32, min(96, points // 8))
        state = _logcosh_reference(target, order)
        engine = GaussHermiteLogCoshExpectation(order=min(128, 2 * order))
        objective_value, expectation = objective(target, state, engine)
        certificate = residuals(state, expectation)
        return ReferenceSolution(
            state,
            objective_value,
            certificate.fisher_rao_squared,
            certificate.bures_wasserstein_squared,
            {"backend": "gauss_hermite_foc", "order": order},
        )
    state = _qmc_reference(target, initial, points=points, seed=seed)
    evaluation = FixedNormalExpectation.qmc(target.dimension, 2 * points, seed + 104729)
    objective_value, expectation = objective(target, state, evaluation)
    certificate = residuals(state, expectation)
    return ReferenceSolution(
        state,
        objective_value,
        certificate.fisher_rao_squared,
        certificate.bures_wasserstein_squared,
        {"backend": "fixed_qmc_lbfgs", "points": points, "evaluation_points": 2 * points},
    )


def laplace_approximation(target: LogisticRegressionTarget) -> GaussianState:
    start = np.zeros(target.dimension, dtype=np.float64)
    result = minimize(
        lambda x: (float(target.value(x)), np.asarray(target.grad(x), dtype=np.float64)),
        start,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
    )
    if not result.success and np.linalg.norm(result.jac) > 1.0e-7:
        raise RuntimeError(f"Laplace mode solve failed: {result.message}")
    hessian = np.asarray(target.hessian(result.x), dtype=np.float64)
    return GaussianState(np.asarray(result.x, dtype=np.float64), spd_inverse(hessian))

