"""Reference Gaussian-VI solutions and non-iterative Laplace baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.diagnostics.core import objective, residuals
from fr_gvi.expectations.core import (
    ExactGaussianExpectation,
    FixedNormalExpectation,
    GaussHermiteLogCoshExpectation,
)
from fr_gvi.linear_algebra.spd import spd_inverse, spd_solve
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


def _triangular_indices(dimension: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Row/column indices of the lower triangle in row-major order, plus a diagonal mask."""

    rows, columns = np.tril_indices(dimension)
    return rows, columns, rows == columns


def _lower_from_parameters(parameters: FloatArray, dimension: int) -> FloatArray:
    rows, columns, diagonal = _triangular_indices(dimension)
    entries = parameters[dimension:].copy()
    entries[diagonal] = np.exp(entries[diagonal])
    lower = np.zeros((dimension, dimension), dtype=np.float64)
    lower[rows, columns] = entries
    return lower


def _parameter_to_state(parameters: FloatArray, dimension: int) -> GaussianState:
    lower = _lower_from_parameters(parameters, dimension)
    return GaussianState(parameters[:dimension], lower @ lower.T)


def _state_to_parameter(state: GaussianState) -> FloatArray:
    dimension = state.mean.size
    lower = np.linalg.cholesky(state.covariance)
    rows, columns, diagonal = _triangular_indices(dimension)
    entries = lower[rows, columns].copy()
    entries[diagonal] = np.log(entries[diagonal])
    return np.concatenate([np.asarray(state.mean, dtype=np.float64), entries])


def _foc_newton(
    target: Target,
    engine: FixedNormalExpectation,
    initial: GaussianState,
    *,
    iterations: int = 200,
    tolerance: float = 1e-14,
) -> GaussianState:
    """Solve the fixed-design first-order conditions by damped Newton.

    The stationarity conditions of the Gaussian variational problem are

        E_q[grad V] = 0,        C^{-1} = E_q[grad^2 V],

    and the map ``(m, C) -> (m - H^{-1} g, H^{-1})`` with ``H = E_q[grad^2 V]``
    is the corresponding Newton step when third derivatives are neglected.  It
    is used purely as a solver for the reference; the resulting point is
    certified afterwards by *both* the Fisher--Rao and Bures--Wasserstein
    residuals, so no geometry is privileged by the choice of solver.
    """

    state = initial
    for _ in range(iterations):
        expectation = engine.evaluate(target, state.mean, state.covariance)
        hessian = np.asarray(expectation.hessian, dtype=np.float64)
        proposal_mean = state.mean - spd_solve(hessian, np.asarray(expectation.grad))
        proposal_covariance = spd_inverse(hessian)
        movement = max(
            float(np.max(np.abs(proposal_mean - state.mean))),
            float(np.max(np.abs(proposal_covariance - state.covariance))),
        )
        state = GaussianState(proposal_mean, proposal_covariance)
        if movement <= tolerance:
            break
    return state


def _qmc_reference(
    target: Target,
    initial: GaussianState,
    engine: FixedNormalExpectation,
    *,
    seed: int,
    certify: bool = True,
) -> tuple[GaussianState, dict[str, float | int | str | bool]]:
    """Minimize the fixed-design Gaussian VI objective to high accuracy."""

    dimension = initial.mean.size
    points = engine.normals.shape[0]
    rows, columns, diagonal_mask = _triangular_indices(dimension)

    def value_gradient(parameters: FloatArray) -> tuple[float, FloatArray]:
        # The Cholesky factor is rebuilt directly from the parameters rather than
        # re-factorizing L L^T, which is both faster and exactly consistent with
        # the reparameterization gradient below.
        lower = _lower_from_parameters(parameters, dimension)
        mean = parameters[:dimension]
        samples = mean + engine.normals @ lower.T
        values = np.asarray(target.value(samples), dtype=np.float64)
        gradients = np.asarray(target.grad(samples), dtype=np.float64)
        objective_value = float(np.mean(values) - np.log(np.diag(lower)).sum())
        mean_gradient = np.mean(gradients, axis=0)
        lower_gradient = (gradients.T @ engine.normals) / points
        entries = lower_gradient[rows, columns].copy()
        entries[diagonal_mask] = entries[diagonal_mask] * np.diag(lower) - 1.0
        return objective_value, np.concatenate([mean_gradient, entries])

    # The reference *state* is the fixed-design first-order-condition point,
    # because that is the common fixed point of every deterministic method in
    # the comparison: FR--R, FR--KL and FB--GVI all consume (E[grad V],
    # E[grad^2 V]) and all stall exactly at E[grad V] = 0, C^{-1} = E[grad^2 V].
    #
    # On a finite quadrature design the Bonnet/Price identities hold only up to
    # the design error, so this point is not exactly the argmin of the
    # reparameterized surrogate.  We therefore also minimize the surrogate
    # directly and take the smaller of the two objective values as the reference
    # objective, which keeps every reported gap non-negative.  The difference is
    # recorded as the quadrature resolution floor of the cell.
    newton_state = _foc_newton(target, engine, initial)
    if not certify:
        # Curvature constants only need the reference state, so the surrogate
        # minimization used for the resolution floor is skipped.
        return newton_state, {
            "backend": "fixed_qmc_foc_newton",
            "points": points,
            "certified": False,
        }
    # The Newton point is already stationary, so a single polish from it is
    # normally enough; the extra starts exist only as a guard when it stalls.
    candidates: list[FloatArray] = [_state_to_parameter(newton_state)]
    _, newton_gradient = value_gradient(candidates[0])
    if float(np.linalg.norm(newton_gradient)) > 1.0e-3:
        if isinstance(target, LogisticRegressionTarget):
            candidates.append(_state_to_parameter(laplace_approximation(target)))
        candidates.append(_state_to_parameter(initial))

    best = None
    for candidate in candidates:
        result = minimize(
            value_gradient,
            candidate,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-12, "maxls": 60},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    gradient_norm = float(np.linalg.norm(best.jac))
    if gradient_norm > 1.0e-4:
        raise RuntimeError(
            f"QMC Gaussian-VI reference failed: {best.message}; grad={gradient_norm:.3e}"
        )
    newton_objective, _ = value_gradient(_state_to_parameter(newton_state))
    metadata = {
        "backend": "fixed_qmc_foc_newton",
        "points": points,
        "surrogate_parameter_gradient_norm": gradient_norm,
        "surrogate_objective": float(best.fun),
        "foc_objective": float(newton_objective),
        "quadrature_resolution_floor": float(newton_objective - best.fun),
        "lbfgs_message": str(best.message),
    }
    return newton_state, metadata


def solve_reference(
    target: Target,
    initial: GaussianState,
    *,
    points: int,
    seed: int,
    engine: FixedNormalExpectation | None = None,
    certify: bool = True,
) -> ReferenceSolution:
    if isinstance(target, GaussianTarget):
        state = GaussianState(target.mean, target.covariance)
        exact = ExactGaussianExpectation()
        objective_value, _ = objective(target, state, exact)
        return ReferenceSolution(state, objective_value, 0.0, 0.0, {"backend": "exact"})
    if isinstance(target, ShiftedLogCoshTarget):
        order = max(80, min(160, points // 8))
        state = _logcosh_reference(target, order)
        quadrature = GaussHermiteLogCoshExpectation(order=min(200, 2 * order))
        objective_value, expectation = objective(target, state, quadrature)
        certificate = residuals(state, expectation)
        return ReferenceSolution(
            state,
            objective_value,
            certificate.fisher_rao_squared,
            certificate.bures_wasserstein_squared,
            {"backend": "gauss_hermite_foc", "order": order},
        )
    # Nonseparable targets are handled on one fixed quadrature design that is
    # shared with the deterministic updates and with the objective evaluation.
    # The reference is then the exact minimizer of the *same* discretized
    # problem the algorithms solve, so deterministic objective gaps are exact
    # rather than being contaminated by a design mismatch.  The residual of the
    # reference against an independent, larger design is reported separately as
    # the honest quadrature-transfer error.
    if engine is None:
        engine = FixedNormalExpectation.qmc(target.dimension, points, seed)
    state, metadata = _qmc_reference(target, initial, engine, seed=seed, certify=certify)
    surrogate_objective, expectation = objective(target, state, engine)
    certificate = residuals(state, expectation)
    if not certify:
        return ReferenceSolution(
            state,
            surrogate_objective,
            certificate.fisher_rao_squared,
            certificate.bures_wasserstein_squared,
            metadata,
        )
    # Non-negative gaps: measure against the smaller of the FOC objective and
    # the directly minimized surrogate objective.
    objective_value = float(min(metadata["foc_objective"], metadata["surrogate_objective"]))
    independent = FixedNormalExpectation.qmc(target.dimension, 4 * engine.normals.shape[0], seed + 104729)
    independent_objective, independent_expectation = objective(target, state, independent)
    independent_certificate = residuals(state, independent_expectation)
    metadata = {
        **metadata,
        "design_transfer_objective_difference": float(independent_objective - objective_value),
        "design_transfer_fisher_rao_residual_squared": independent_certificate.fisher_rao_squared,
        "design_transfer_bures_wasserstein_residual_squared": (
            independent_certificate.bures_wasserstein_squared
        ),
        "independent_points": int(4 * engine.normals.shape[0]),
    }
    return ReferenceSolution(
        state,
        objective_value,
        certificate.fisher_rao_squared,
        certificate.bures_wasserstein_squared,
        metadata,
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

