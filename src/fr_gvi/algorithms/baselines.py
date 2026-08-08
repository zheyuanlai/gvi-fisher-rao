"""Modern natural-gradient and parameter-space baselines.

The three methods here exist so that the manuscript's comparison separates the
*geometry* from the *gradient estimator*.  Restricted to ``FR--R``, ``FR--KL``
and ``FB--GVI``, the campaign can only say that Fisher--Rao differs from
Bures--Wasserstein; it cannot say which part of the difference is geometric.
Adding a natural-gradient scheme with a cheaper retraction, and two
parameter-space schemes that differ from each other only in their estimator,
turns that into a two-way comparison:

===================  ====================  ==================  =================
                     Fisher--Rao           Bures--Wasserstein  Parameter space
===================  ====================  ==================  =================
deterministic        FR--R, FR--KL         FB--GVI             Sq--NGVI
Price/Hessian        FR--R--STL,           S--FB--GVI          Price--BBVI
                     FR--KL--STL
gradient-only        --                    --                  BBVI--STL
===================  ====================  ==================  =================

All three are implemented as published, including their published safeguards.
Each is written against a Cholesky factor ``C`` of the covariance, lower
triangular with a strictly positive diagonal, so that ``Sigma = C C^T``.  The
campaign carries ``(m, Sigma)``, and ``C`` is recovered exactly by a Cholesky
factorization at the start of every step: on the set these methods are defined
on, the factor with positive diagonal is unique, so the round trip is exact
rather than a reparameterization.  A step that leaves that set is a genuine
breakdown of the method and is raised as a failure, never repaired -- see
``_positive_diagonal``.

References
----------
Sq--NGVI
    Kumar, Moellenhoff, Khan and Lucchi, *Optimization Guarantees for
    Square-Root Natural-Gradient Variational Inference*, TMLR 2025,
    Algorithm 1 (SR-VN) and Equation (14).
Price--BBVI
    Kim, Fu, Ma, Gardner and Campbell, *Stochastic Gradient Variational
    Inference with Price's Gradient Estimator from Bures--Wasserstein to
    Parameter Space*, ICML 2026: stochastic proximal gradient descent (SPGD) in
    the parameterization of Assumption 2.2, with the Bonnet--Price estimator of
    Section 3.2 and the closed-form entropy proximal operator of Section 2.3.
BBVI--STL
    Kim, Ma and Gardner, *Linear Convergence of Black-Box Variational
    Inference: Should We Stick the Landing?*, AISTATS 2024: projected SGD on the
    triangular scale parameterization, with the sticking-the-landing estimator
    of Definition 5 and the projection of Proposition 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

FloatArray = NDArray[np.float64]


class FactorBreakdown(ValueError):
    """The square-root factor left the set the method is defined on."""


def cholesky_factor(covariance: FloatArray) -> FloatArray:
    """The unique lower-triangular factor with positive diagonal."""

    try:
        return np.asarray(np.linalg.cholesky(covariance), dtype=np.float64)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded upstream
        raise FactorBreakdown(f"covariance is not positive definite: {exc}") from exc


def _positive_diagonal(factor: FloatArray, name: str) -> FloatArray:
    """Reject a factor that has left the positive-diagonal triangular set.

    ``Sigma = C C^T`` stays positive definite when a diagonal entry of ``C``
    changes sign, so a naive implementation would sail past this point and keep
    iterating on a state its own analysis does not cover.  It is also exactly the
    point at which recovering ``C`` from ``Sigma`` would silently flip the sign
    back.  Both make it a failure to record, not a state to continue from.
    """

    diagonal = np.diag(factor)
    if not np.all(np.isfinite(factor)):
        raise FactorBreakdown(f"{name}: square-root factor is not finite")
    if np.any(diagonal <= 0.0):
        raise FactorBreakdown(
            f"{name}: square-root factor lost its positive diagonal, "
            f"min diagonal = {float(np.min(diagonal)):.6e}"
        )
    return factor


def _tril_half_diagonal(matrix: FloatArray) -> FloatArray:
    """``tril[A]`` of Kumar et al. (2025), Equation (10): halved diagonal."""

    lower = np.tril(matrix)
    np.fill_diagonal(lower, 0.5 * np.diag(matrix))
    return np.asarray(lower, dtype=np.float64)


def _tril_full_diagonal(matrix: FloatArray) -> FloatArray:
    """Gradient with respect to a lower-triangular parameter: plain lower part.

    For ``x = m + C u`` with ``C`` triangular, the free parameters are the
    entries on and below the diagonal, so the parameter gradient is the plain
    lower triangle of the ambient gradient.  This differs from
    :func:`_tril_half_diagonal`, which carries the factor of one half from the
    derivative of the Cholesky map itself.
    """

    return np.asarray(np.tril(matrix), dtype=np.float64)


def square_root_natural_gradient_step(
    mean: FloatArray,
    covariance: FloatArray,
    expected_gradient: FloatArray,
    hessian: FloatArray,
    step_size: float,
) -> tuple[FloatArray, FloatArray]:
    """One SR-VN step: Kumar et al. (2025), Algorithm 1 with ``gamma = 1``.

    With ``g_t`` the expected gradient and ``H_t`` the expected Hessian of the
    potential,

        C_{t+1} = C_t - rho C_t tril[C_t^T H_t C_t - I],
        m_{t+1} = m_t - rho C_t C_t^T g_t.

    The mean update coincides with the Fisher--Rao mean update, since
    ``C_t C_t^T = Sigma_t``.  The methods differ only in how the covariance is
    advanced: ``FR--R`` exponentiates, ``FR--KL`` inverts a resolvent, and
    SR-VN takes a forward Euler step on the factor.  That is exactly the
    comparison the panel is for, so the shared mean update is retained rather
    than perturbed.
    """

    factor = _positive_diagonal(cholesky_factor(covariance), "Sq--NGVI")
    dimension = factor.shape[0]
    whitened = factor.T @ hessian @ factor - np.eye(dimension, dtype=np.float64)
    updated = factor - step_size * factor @ _tril_half_diagonal(whitened)
    updated = _positive_diagonal(updated, "Sq--NGVI")
    mean_new = mean - step_size * (factor @ (factor.T @ expected_gradient))
    return np.asarray(mean_new, dtype=np.float64), np.asarray(updated @ updated.T)


def _entropy_proximal_diagonal(factor: FloatArray, step_size: float) -> FloatArray:
    """``prox`` of the negative entropy on a triangular factor.

    The entropy of ``N(m, C C^T)`` contributes ``-sum_i log C_ii``, whose
    proximal operator is separable over the diagonal and solves
    ``c' - c - gamma / c' = 0``, that is
    ``c' = (c + sqrt(c^2 + 4 gamma)) / 2`` -- Kim et al. (2026), Section 2.3.
    It returns a strictly positive diagonal for any real input, so
    ``Price--BBVI`` cannot leave its parameter set; this is the published
    algorithm's own proximal step, not a repair added here.
    """

    updated = np.array(factor, dtype=np.float64, copy=True)
    diagonal = np.diag(factor)
    np.fill_diagonal(
        updated,
        0.5 * (diagonal + np.sqrt(diagonal * diagonal + 4.0 * step_size)),
    )
    return updated


def price_bbvi_step(
    mean: FloatArray,
    covariance: FloatArray,
    sampled_gradient: FloatArray,
    sampled_hessian: FloatArray,
    step_size: float,
) -> tuple[FloatArray, FloatArray]:
    """One SPGD step with the Bonnet--Price estimator, Kim et al. (2026).

    In the parameterization ``Sigma = C C^T`` the energy gradients are
    ``grad_m E = E[grad V]`` and, by Price's theorem,
    ``grad_C E = E[grad^2 V] C``; restricting to the triangular free parameters
    gives the lower triangle.  The energy step is followed by the closed-form
    entropy proximal step, which acts on the diagonal alone.

    This is the parameter-space arm of the estimator-versus-geometry
    comparison: it shares its second-order estimator with ``S--FB--GVI`` and its
    parameter space with ``BBVI--STL``.
    """

    factor = _positive_diagonal(cholesky_factor(covariance), "Price--BBVI")
    energy_gradient = _tril_full_diagonal(sampled_hessian @ factor)
    half = factor - step_size * energy_gradient
    updated = _entropy_proximal_diagonal(half, step_size)
    updated = _positive_diagonal(updated, "Price--BBVI")
    mean_new = mean - step_size * sampled_gradient
    return np.asarray(mean_new, dtype=np.float64), np.asarray(updated @ updated.T)


def sticking_the_landing_parameter_gradient(
    factor: FloatArray,
    normals: FloatArray,
    gradients: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The STL estimator of Kim et al. (2024), Definition 5.

    Stopping the gradient through the density leaves the path derivative of
    ``log q_nu(T_lambda(u)) - log pi(T_lambda(u))``.  For ``x = m + C u`` the
    score is ``grad_x log q(x) = -C^{-T} u``, so with ``r = grad V(x) - C^{-T}u``
    the estimator is ``(r, r u^T)`` restricted to the triangular parameters.

    At a Gaussian target evaluated at its own optimizer, ``grad V(x)`` equals
    ``Sigma_*^{-1}(x - m_*) = C^{-T} u`` and ``r`` vanishes for every sample,
    which is the pathwise cancellation the estimator is named for.
    """

    scores = solve_triangular(factor.T, normals.T, lower=False).T
    residual = np.asarray(gradients, dtype=np.float64) - scores
    mean_gradient = np.mean(residual, axis=0)
    factor_gradient = _tril_full_diagonal(
        residual.T @ np.asarray(normals, dtype=np.float64) / normals.shape[0]
    )
    return np.asarray(mean_gradient, dtype=np.float64), factor_gradient


def bbvi_stl_step(
    mean: FloatArray,
    covariance: FloatArray,
    normals: FloatArray,
    gradients: FloatArray,
    step_size: float,
    projection_floor: float,
) -> tuple[FloatArray, FloatArray, int]:
    """One projected-SGD step with the STL estimator, Kim et al. (2024).

    The projection onto ``Lambda_S = {sigma_min(C) >= 1 / sqrt(S)}`` is
    elementwise on the diagonal (their Proposition 1) and costs ``Theta(d)``.
    It is part of the published method and of the domain its guarantee is
    stated on, so it is applied rather than omitted; the number of iterations at
    which it actually binds is counted and reported, so a reader can see when the
    safeguard is doing work.  ``S`` is the log-smoothness constant of the target
    in the original coordinates, which every family in the campaign supplies in
    closed form from the model alone.
    """

    factor = _positive_diagonal(cholesky_factor(covariance), "BBVI--STL")
    mean_gradient, factor_gradient = sticking_the_landing_parameter_gradient(
        factor, normals, gradients
    )
    half = factor - step_size * factor_gradient
    diagonal = np.diag(half)
    activations = int(np.count_nonzero(diagonal < projection_floor))
    updated = np.array(half, dtype=np.float64, copy=True)
    np.fill_diagonal(updated, np.maximum(diagonal, projection_floor))
    updated = _positive_diagonal(updated, "BBVI--STL")
    mean_new = mean - step_size * mean_gradient
    return np.asarray(mean_new, dtype=np.float64), np.asarray(updated @ updated.T), activations
