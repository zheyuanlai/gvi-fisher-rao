"""Exact curvature constants in original and optimizer-whitened coordinates.

The manuscript's rates are stated in the optimizer-whitened constants

    alpha_star = inf_x lambda_min(C_star^{1/2} grad^2 V(x) C_star^{1/2}),
    beta_star  = sup_x lambda_max(C_star^{1/2} grad^2 V(x) C_star^{1/2}),
    kappa_star = beta_star / alpha_star,

together with the whitened initial-covariance bounds ``lambda_{0,star}`` and
``lambda_{0,star}^max`` and the ceiling ``lambda_max_star``.  For each of the
three target families used in the campaign these suprema are available in closed
form, so no optimization or sampling is involved.

The eigenvalue identity used throughout is that for any factorization
``C_star = R R^T`` the matrices ``C_star^{1/2} M C_star^{1/2}`` and ``R^T M R``
are similar, hence share their spectrum.  This lets us avoid the symmetric
square root when a structured factor is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.targets.core import (
    GaussianTarget,
    LogisticRegressionTarget,
    ShiftedLogCoshTarget,
    Target,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CurvatureConstants:
    alpha: float
    beta: float
    condition: float
    alpha_star: float
    beta_star: float
    kappa_star: float
    exact: bool
    source: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def _whitened_spectrum_bounds(
    factor: FloatArray, lower: FloatArray, upper: FloatArray
) -> tuple[float, float]:
    """Extreme whitened eigenvalues over the Hessian interval [lower, upper]."""

    low = np.linalg.eigvalsh(factor.T @ lower @ factor)
    high = np.linalg.eigvalsh(factor.T @ upper @ factor)
    return float(low[0]), float(high[-1])


def curvature_constants(target: Target, optimum: GaussianState) -> CurvatureConstants:
    if isinstance(target, GaussianTarget):
        eigenvalues = np.linalg.eigvalsh(target.precision)
        alpha, beta = float(eigenvalues[0]), float(eigenvalues[-1])
        # The whitened Hessian of a Gaussian target is exactly the identity.
        return CurvatureConstants(
            alpha, beta, beta / alpha, 1.0, 1.0, 1.0, True, "gaussian_exact"
        )

    if isinstance(target, ShiftedLogCoshTarget):
        # V(x) = sum_i [nu_i z_i^2 / 2 + rho logcosh(z_i - b_i)], z = T^{-1}(x - s).
        # The separable Hessian lies in diag([nu_i, nu_i + rho]) and the optimizer
        # is C_star = T diag(sigma^2) T^T, so R = T diag(sigma) is a factor and
        # R^T grad^2 V R = diag(sigma_i^2 D_ii).  Both extremes are attained.
        inverse = target.inverse_transform
        lower_diagonal = target.nu
        upper_diagonal = target.nu + target.rho
        alpha = float(np.linalg.eigvalsh(inverse.T @ np.diag(lower_diagonal) @ inverse)[0])
        beta = float(np.linalg.eigvalsh(inverse.T @ np.diag(upper_diagonal) @ inverse)[-1])
        whitened = np.linalg.solve(target.transform, optimum.covariance)
        whitened = np.linalg.solve(target.transform, whitened.T).T
        variances = np.clip(np.diag(whitened), 0.0, None)
        alpha_star = float(np.min(variances * lower_diagonal))
        beta_star = float(np.max(variances * upper_diagonal))
        return CurvatureConstants(
            alpha,
            beta,
            beta / alpha,
            alpha_star,
            beta_star,
            beta_star / alpha_star,
            True,
            "logcosh_separable_exact",
        )

    if isinstance(target, LogisticRegressionTarget):
        # grad^2 V = lambda I + sum_i w_i x_i x_i^T with w_i in (0, 1/4].  The
        # bounds below are the exact infimum and supremum over the parameter.
        features = target.features
        gram = features.T @ features
        identity = np.eye(target.dimension)
        lower = target.prior_precision * identity
        upper = target.prior_precision * identity + 0.25 * gram
        alpha = float(np.linalg.eigvalsh(lower)[0])
        beta = float(np.linalg.eigvalsh(upper)[-1])
        factor = np.linalg.cholesky(optimum.covariance)
        alpha_star, beta_star = _whitened_spectrum_bounds(factor, lower, upper)
        return CurvatureConstants(
            alpha,
            beta,
            beta / alpha,
            alpha_star,
            beta_star,
            beta_star / alpha_star,
            True,
            "logistic_interval_exact",
        )

    raise TypeError(f"no curvature formula for target type {type(target).__name__}")


@dataclass(frozen=True)
class WhitenedInitialization:
    lambda_0_star: float
    lambda_0_star_max: float
    lambda_max_star: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def whitened_initialization(
    initial: GaussianState, optimum: GaussianState, alpha_star: float
) -> WhitenedInitialization:
    factor = np.linalg.cholesky(optimum.covariance)
    whitened = np.linalg.solve(factor, np.linalg.solve(factor, initial.covariance).T).T
    values = np.linalg.eigvalsh((whitened + whitened.T) / 2.0)
    minimum, maximum = float(values[0]), float(values[-1])
    return WhitenedInitialization(minimum, maximum, max(maximum, 1.0 / alpha_star))


def certified_step_sizes(
    constants: CurvatureConstants, initialization: WhitenedInitialization
) -> dict[str, float]:
    """Largest stepsizes admitted by the theorems, per method.

    Fisher--Rao bounds are the global conditions of manuscript Theorems 2.9 and
    2.14; the FB--GVI bound is ``eta <= 1/beta`` from Diao et al., Corollary D.2.
    """

    ceiling = constants.beta_star * initialization.lambda_max_star
    return {
        "FR--R": 1.0 / (2.0 * ceiling),
        "FR--KL": 1.0 / ceiling,
        "FR--R--STL": 1.0 / constants.kappa_star,
        "FR--KL--STL": 1.0 / (8.0 * constants.kappa_star),
        "FB--GVI": 1.0 / constants.beta,
        "S--FB--GVI": 1.0 / constants.beta,
    }
