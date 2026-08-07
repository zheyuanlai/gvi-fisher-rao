"""Reference Gaussian-VI solutions and non-iterative Laplace baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.diagnostics.core import objective, residuals
from fr_gvi.expectations.core import (
    ExactGaussianExpectation,
    ExpectationEngine,
    FixedNormalExpectation,
    GaussHermiteLogCoshExpectation,
    LogisticExactExpectation,
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


def logcosh_marginal_optimizer(
    nu_values: FloatArray, rho_value: float, offsets: FloatArray, order: int = 32
) -> tuple[FloatArray, FloatArray]:
    """Per-coordinate Gaussian-VI optimizer of the separable log-cosh marginals.

    Returns the optimizing means and variances of the ``z`` coordinates, which are
    the diagonal of the optimizer in any affine image ``x = T z + s``.
    """

    # The stationarity conditions are integrated with the *same* panelled rule the
    # expectation engine uses.  A single Gauss--Legendre panel over
    # [m - 12 s, m + 12 s] places its nodes proportionally to the marginal width, so
    # in the small-curvature coordinates -- where the marginal is several units wide
    # but the tanh and sech^2 structure lives in a window of unit width -- it misses
    # the peak.  That is what made the reference residual on the wide-marginal cells
    # five orders of magnitude worse than on the rest of the grid.
    rule = GaussHermiteLogCoshExpectation(order=order)
    dimension = int(np.asarray(nu_values).size)
    means = np.zeros(dimension, dtype=np.float64)
    variances = np.ones(dimension, dtype=np.float64)
    for index in range(dimension):
        nu = float(np.asarray(nu_values)[index])
        rho = float(rho_value)
        offset = float(np.asarray(offsets)[index])

        def conditions(parameters: FloatArray) -> FloatArray:
            mean = parameters[0]
            variance = np.exp(parameters[1])
            samples, weights = rule._rule(mean, np.sqrt(variance), offset)
            tanh = np.tanh(samples - offset)
            expected_gradient = float(weights @ (nu * samples + rho * tanh))
            expected_hessian = float(weights @ (nu + rho * (1.0 - tanh**2)))
            return np.asarray([expected_gradient, variance * expected_hessian - 1.0])

        solution = least_squares(conditions, np.asarray([0.0, -np.log(nu + rho)]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if not solution.success or np.linalg.norm(solution.fun) > 1.0e-11:
            raise RuntimeError(f"log-cosh reference solve failed in coordinate {index}: {solution.message}")
        means[index] = solution.x[0]
        variances[index] = np.exp(solution.x[1])
    return means, variances


def _logcosh_reference(target: ShiftedLogCoshTarget, order: int) -> GaussianState:
    means, variances = logcosh_marginal_optimizer(
        target.nu, target.rho, target.offset, order
    )
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


def _foc_residual(target: Target, engine: ExpectationEngine, state: GaussianState) -> float:
    """Squared first-order-condition residual at a state, on the fixed design."""

    expectation = engine.evaluate(target, state.mean, state.covariance)
    return float(residuals(state, expectation).fisher_rao_squared)


def _foc_newton(
    target: Target,
    engine: ExpectationEngine,
    initial: GaussianState,
    *,
    iterations: int = 200,
    tolerance: float = 1e-14,
) -> tuple[GaussianState, float]:
    """Solve the fixed-design first-order conditions by damped Newton.

    The stationarity conditions of the Gaussian variational problem are

        E_q[grad V] = 0,        C^{-1} = E_q[grad^2 V],

    and the map ``(m, C) -> (m - H^{-1} g, H^{-1})`` with ``H = E_q[grad^2 V]``
    is the corresponding Newton step when third derivatives are neglected.  It
    is used purely as a solver for the reference; the resulting point is
    certified afterwards by *both* the Fisher--Rao and Bures--Wasserstein
    residuals, so no geometry is privileged by the choice of solver.

    The undamped iteration diverges on near-separable logistic posteriors with a
    weak prior, where one full Newton step throws the mean hundreds of units away
    and the iteration then cycles.  Each step is therefore accepted only if it
    decreases the residual, and backtracked along the segment to the proposal
    otherwise.  The achieved residual is returned so the caller can fall back to
    direct minimization when the solve does not converge.
    """

    state = initial
    residual = _foc_residual(target, engine, state)
    for _ in range(iterations):
        expectation = engine.evaluate(target, state.mean, state.covariance)
        hessian = np.asarray(expectation.hessian, dtype=np.float64)
        proposal_mean = state.mean - spd_solve(hessian, np.asarray(expectation.grad))
        proposal_covariance = spd_inverse(hessian)

        accepted = None
        step = 1.0
        for _ in range(30):
            try:
                candidate = GaussianState(
                    (1.0 - step) * state.mean + step * proposal_mean,
                    (1.0 - step) * state.covariance + step * proposal_covariance,
                )
                candidate_residual = _foc_residual(target, engine, candidate)
            except (ValueError, np.linalg.LinAlgError):
                step *= 0.5
                continue
            if candidate_residual < residual:
                accepted = (candidate, candidate_residual)
                break
            step *= 0.5
        if accepted is None:
            break
        movement = max(
            float(np.max(np.abs(accepted[0].mean - state.mean))),
            float(np.max(np.abs(accepted[0].covariance - state.covariance))),
        )
        state, residual = accepted
        if movement <= tolerance or residual <= 1e-24:
            break
    return state, residual


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
    newton_state, newton_residual = _foc_newton(target, engine, initial)
    if not certify:
        # Curvature constants only need the reference state, so the surrogate
        # minimization used for the resolution floor is skipped.
        return newton_state, {
            "backend": "fixed_qmc_foc_newton",
            "points": points,
            "certified": False,
        }
    # The Newton point is normally already stationary, so a single polish from it
    # is enough; the extra starts exist as a guard when it stalls.
    candidates: list[FloatArray] = [_state_to_parameter(newton_state)]
    _, newton_gradient = value_gradient(candidates[0])
    if float(np.linalg.norm(newton_gradient)) > 1.0e-3 or newton_residual > 1.0e-8:
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
    # If the damped Newton solve did not reach a stationary point -- which happens
    # on near-separable logistic posteriors with a weak prior -- the directly
    # minimized surrogate is used as the reference state instead, and the switch
    # is recorded.  The state is certified by its residuals either way.
    surrogate_state = _parameter_to_state(np.asarray(best.x, dtype=np.float64), dimension)
    surrogate_residual = _foc_residual(target, engine, surrogate_state)
    if surrogate_residual < newton_residual:
        newton_state, newton_residual = surrogate_state, surrogate_residual
        solver = "fixed_qmc_surrogate_lbfgs"
    else:
        solver = "fixed_qmc_foc_newton"
    newton_objective, _ = value_gradient(_state_to_parameter(newton_state))
    metadata = {
        "backend": solver,
        "foc_residual_squared": newton_residual,
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
        return ReferenceSolution(
            state, objective_value, 0.0, 0.0, {"backend": "exact", "analytic": True}
        )
    if isinstance(target, ShiftedLogCoshTarget):
        order = max(32, min(64, points // 32))
        state = _logcosh_reference(target, order)
        quadrature = GaussHermiteLogCoshExpectation(order=min(128, 2 * order))
        objective_value, expectation = objective(target, state, quadrature)
        certificate = residuals(state, expectation)
        return ReferenceSolution(
            state,
            objective_value,
            certificate.fisher_rao_squared,
            certificate.bures_wasserstein_squared,
            {"backend": "gauss_hermite_foc", "order": order},
        )
    if isinstance(target, LogisticRegressionTarget):
        # The logistic expectations are exact, so the stationarity conditions
        #     E_q[grad V] = 0,     C^{-1} = E_q[grad^2 V]
        # can be solved directly.  Under strong logconcavity the objective has a
        # unique stationary point which is its minimizer, so there is no surrogate
        # to minimize separately and no design to transfer between: the reference
        # is the true Gaussian variational optimum up to the residual reported here.
        exact = LogisticExactExpectation(96)
        state, residual = _foc_newton(target, exact, initial)
        objective_value, expectation = objective(target, state, exact)
        certificate = residuals(state, expectation)
        return ReferenceSolution(
            state,
            objective_value,
            certificate.fisher_rao_squared,
            certificate.bures_wasserstein_squared,
            {
                "backend": "logistic_exact_foc_newton",
                "order": exact.order,
                "foc_residual_squared": residual,
            },
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

