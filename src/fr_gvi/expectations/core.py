"""Deterministic and stochastic Gaussian expectation engines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import expit, ndtri
from scipy.stats import qmc

from fr_gvi.linear_algebra.spd import spd_sqrt
from fr_gvi.targets.core import (
    GaussianTarget,
    LogisticRegressionTarget,
    ShiftedLogCoshTarget,
    Target,
    mean_hessian,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExpectationResult:
    value: float
    grad: FloatArray
    hessian: FloatArray


class ExpectationEngine(Protocol):
    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult: ...


@dataclass(frozen=True)
class ExactGaussianExpectation:
    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult:
        if not isinstance(target, GaussianTarget):
            raise TypeError("exact Gaussian backend only supports GaussianTarget")
        delta = mean - target.mean
        value = 0.5 * (
            float(delta @ target.precision @ delta) + float(np.trace(target.precision @ covariance))
        )
        return ExpectationResult(value, target.precision @ delta, target.precision.copy())


@dataclass(frozen=True)
class FixedNormalExpectation:
    normals: FloatArray
    weights: FloatArray
    backend: str
    seed: int

    @classmethod
    def qmc(cls, dimension: int, points: int, seed: int) -> "FixedNormalExpectation":
        exponent = int(np.ceil(np.log2(points)))
        uniforms = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(exponent)[:points]
        epsilon = np.finfo(np.float64).eps
        normals = ndtri(np.clip(uniforms, epsilon, 1.0 - epsilon)).astype(np.float64)
        weights = np.full(points, 1.0 / points, dtype=np.float64)
        return cls(normals, weights, "scrambled_sobol", seed)

    @classmethod
    def iid(cls, dimension: int, points: int, seed: int) -> "FixedNormalExpectation":
        normals = np.random.default_rng(seed).standard_normal((points, dimension), dtype=np.float64)
        weights = np.full(points, 1.0 / points, dtype=np.float64)
        return cls(normals, weights, "iid_normal", seed)

    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult:
        root = spd_sqrt(covariance)
        samples = mean + self.normals @ root.T
        values = np.asarray(target.value(samples), dtype=np.float64)
        gradients = np.asarray(target.grad(samples), dtype=np.float64)
        return ExpectationResult(
            float(self.weights @ values),
            np.asarray(np.einsum("s,si->i", self.weights, gradients), dtype=np.float64),
            mean_hessian(target, samples, self.weights),
        )


@lru_cache(maxsize=32)
def _legendre_rule(count: int) -> tuple[FloatArray, FloatArray]:
    """Cached Gauss--Legendre nodes and weights; the solve is O(n^2)."""

    nodes, weights = leggauss(count)
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _panelled_normal_rule(
    means: FloatArray,
    deviations: FloatArray,
    *,
    order: int = 48,
    truncation: float = 12.0,
    transition: float = 20.0,
) -> tuple[FloatArray, FloatArray]:
    """Vectorized panelled Gauss--Legendre rules against ``N(mean_i, dev_i^2)``.

    One rule per row, all sharing the same panel structure: ``[m - L s, m + L s]``
    split at the edges of a fixed transition window around the origin, with the
    middle panel resolved finely.  This is the rule of
    :class:`GaussHermiteLogCoshExpectation`, lifted to act on a whole batch of
    marginals at once, and it exists for the same reason: the integrand's
    structure lives in a window of unit width while the marginal may be many units
    wide, so any rule whose nodes scale with the marginal misses it.
    """

    means = np.atleast_1d(np.asarray(means, dtype=np.float64))
    deviations = np.atleast_1d(np.asarray(deviations, dtype=np.float64))
    low = means - truncation * deviations
    high = means + truncation * deviations
    inner_low = np.clip(-transition, low, high)
    inner_high = np.clip(transition, low, high)
    nodes: list[FloatArray] = []
    weights: list[FloatArray] = []
    for left, right, count in (
        (low, inner_low, 2 * order),
        (inner_low, inner_high, 6 * order),
        (inner_high, high, 2 * order),
    ):
        raw_nodes, raw_weights = _legendre_rule(max(16, int(count)))
        half = 0.5 * (right - left)[:, None]
        centre = 0.5 * (right + left)[:, None]
        points = centre + half * raw_nodes[None, :]
        density = np.exp(
            -0.5 * ((points - means[:, None]) / deviations[:, None]) ** 2
        ) / (deviations[:, None] * np.sqrt(2.0 * np.pi))
        nodes.append(points)
        weights.append(raw_weights[None, :] * half * density)
    return np.concatenate(nodes, axis=1), np.concatenate(weights, axis=1)


def expected_bernoulli_probabilities(
    features: FloatArray, mean: FloatArray, covariance: FloatArray, order: int = 48
) -> FloatArray:
    """``E_q[sigma(x_i . theta)]`` for each row of ``features``, without sampling.

    The posterior predictive probability of a logistic model under a Gaussian
    ``q`` is a one-dimensional integral per point, for the same reason the
    training expectations are: the model sees ``theta`` only through
    ``x_i . theta``.  Computing it exactly keeps Monte Carlo out of the reported
    held-out metrics.
    """

    features = np.asarray(features, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    predictor_mean = features @ mean
    predictor_variance = np.einsum("nd,de,ne->n", features, covariance, features)
    deviations = np.maximum(
        np.sqrt(np.maximum(predictor_variance, 0.0)), np.sqrt(np.finfo(np.float64).eps)
    )
    nodes, weights = _panelled_normal_rule(predictor_mean, deviations, order=order)
    return np.asarray(np.einsum("nq,nq->n", weights, expit(nodes)), dtype=np.float64)


@dataclass(frozen=True)
class LogisticExactExpectation:
    """Exact Gaussian expectations for logistic regression, to quadrature precision.

    The potential depends on ``theta`` only through the linear predictors
    ``z_i = x_i . theta``, and under ``theta ~ N(m, C)`` each ``z_i`` is the scalar
    Gaussian ``N(x_i . m, x_i^T C x_i)``.  The objective, the expected gradient and
    the expected Hessian therefore reduce to ``n`` independent one-dimensional
    integrals,

        E[V]      = (lambda/2)(|m|^2 + tr C) + sum_i (E[log(1+e^{z_i})] - y_i mu_i),
        E[grad V] = lambda m + sum_i (E[sigma(z_i)] - y_i) x_i,
        E[hess V] = lambda I + sum_i E[sigma(z_i)(1-sigma(z_i))] x_i x_i^T,

    with no Monte Carlo anywhere.  This removes the design-mismatch question from
    the logistic experiment entirely: the updates, the objective evaluation and the
    reference solve all see the same exact population quantities, so a reported gap
    is a gap to the true Gaussian variational optimum rather than to the minimizer
    of a finite-sample surrogate.  It is also far cheaper, ``O(n Q + n d^2)``
    instead of ``O(S n d)``, and far more accurate: on the cells used here a
    scrambled Sobol design of 4096 points misplaces the objective by about
    ``1e-1``, while a single marginal of this rule agrees with an adaptive
    integrator to ``4e-14`` and the summed objective changes by ``2e-11`` between
    orders 48 and 96.
    """

    order: int = 48

    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult:
        if not isinstance(target, LogisticRegressionTarget):
            raise TypeError("exact logistic backend supports LogisticRegressionTarget")
        features, labels = target.features, target.labels
        prior = target.prior_precision
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)

        predictor_mean = features @ mean
        predictor_variance = np.einsum("nd,de,ne->n", features, covariance, features)
        deviations = np.sqrt(np.maximum(predictor_variance, 0.0))
        # A degenerate marginal would make the panel construction singular; the
        # Gaussian collapses to a point mass there, which the rule cannot express.
        floor = np.sqrt(np.finfo(np.float64).eps)
        nodes, weights = _panelled_normal_rule(
            predictor_mean, np.maximum(deviations, floor), order=self.order
        )
        probabilities = expit(nodes)
        expected_probability = np.einsum("nq,nq->n", weights, probabilities)
        expected_curvature = np.einsum(
            "nq,nq->n", weights, probabilities * (1.0 - probabilities)
        )
        expected_softplus = np.einsum("nq,nq->n", weights, np.logaddexp(0.0, nodes))

        value = 0.5 * prior * (float(mean @ mean) + float(np.trace(covariance)))
        value += float(np.sum(expected_softplus - labels * predictor_mean))
        gradient = prior * mean + features.T @ (expected_probability - labels)
        gram = features.T @ (features * expected_curvature[:, None])
        hessian = 0.5 * (gram + gram.T) + prior * np.eye(mean.size, dtype=np.float64)
        return ExpectationResult(value, np.asarray(gradient), np.asarray(hessian))


@dataclass(frozen=True)
class GaussHermiteLogCoshExpectation:
    """Separable one-dimensional quadrature for affine images of log-cosh targets.

    The integrands carry two very different length scales: the ``tanh`` and
    ``sech^2`` structure of the potential varies over a window of unit width
    around the offset, while the marginal standard deviation reaches 7 and more
    in the small-curvature coordinates of the ill-conditioned cells.  Any rule
    whose nodes are spaced proportionally to the marginal width therefore fails
    to see the peak.  Gauss--Hermite is such a rule: on the widest cell here it
    is still wrong by ``5e-4`` at order 200, and raising the order moves the
    answer instead of converging it, which silently shifted the fixed point of
    the deterministic schemes.

    The rule used instead is panelled Gauss--Legendre against the Gaussian
    density: ``[m - L s, m + L s]`` is split at the edges of the transition
    window and the middle panel is resolved finely, so accuracy does not degrade
    as the marginal widens.  Measured against an adaptive integrator it holds
    ``1e-13`` relative accuracy over five orders of magnitude in the variance.
    Truncation at ``L = 12`` standard deviations is below floating-point
    relevance for a Gaussian.

    ``order`` scales the node counts.  The class name is kept for continuity with
    the manifests already written against it.
    """

    order: int = 32
    truncation: float = 12.0
    transition: float = 20.0

    def _rule(self, mean: float, deviation: float, offset: float) -> tuple[FloatArray, FloatArray]:
        """Panelled Gauss--Legendre nodes and weights against ``N(mean, deviation^2)``.

        The ``tanh`` and ``sech^2`` structure of the potential lives in a window of
        fixed width around ``offset``, while the Gaussian spreads over ``deviation``.
        A single rule scaled to either scale fails when they differ: this splits
        ``[mean - L s, mean + L s]`` at the edges of the transition window and
        resolves the middle panel finely, so the accuracy does not degrade as the
        marginal widens.
        """

        low = mean - self.truncation * deviation
        high = mean + self.truncation * deviation
        inner_low = float(np.clip(offset - self.transition, low, high))
        inner_high = float(np.clip(offset + self.transition, low, high))
        panels = [
            (low, inner_low, 8 * self.order),
            (inner_low, inner_high, 32 * self.order),
            (inner_high, high, 8 * self.order),
        ]
        nodes: list[FloatArray] = []
        weights: list[FloatArray] = []
        for left, right, count in panels:
            if right - left <= 0.0:
                continue
            raw_nodes, raw_weights = _legendre_rule(max(32, int(count)))
            half = 0.5 * (right - left)
            centre = 0.5 * (right + left)
            points = centre + half * raw_nodes
            density = np.exp(-0.5 * ((points - mean) / deviation) ** 2) / (
                deviation * np.sqrt(2.0 * np.pi)
            )
            nodes.append(points)
            weights.append(raw_weights * half * density)
        return np.concatenate(nodes), np.concatenate(weights)

    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult:
        if not isinstance(target, ShiftedLogCoshTarget):
            raise TypeError("separable quadrature backend supports ShiftedLogCoshTarget")
        inverse = target.inverse_transform
        mean_z = inverse @ (mean - target.shift)
        covariance_z = inverse @ covariance @ inverse.T
        value = 0.0
        grad_z = np.zeros(target.dimension, dtype=np.float64)
        hessian_diagonal = np.zeros(target.dimension, dtype=np.float64)
        for index in range(target.dimension):
            deviation = np.sqrt(max(float(covariance_z[index, index]), 0.0))
            samples, weights = self._rule(
                float(mean_z[index]), deviation, float(target.offset[index])
            )
            shifted = samples - target.offset[index]
            logcosh = target._logcosh(shifted)
            tanh = np.tanh(shifted)
            value += float(
                weights
                @ (0.5 * target.nu[index] * samples**2 + target.rho * logcosh)
            )
            grad_z[index] = float(
                weights @ (target.nu[index] * samples + target.rho * tanh)
            )
            hessian_diagonal[index] = float(
                weights @ (target.nu[index] + target.rho * (1.0 - tanh**2))
            )
        grad = inverse.T @ grad_z
        hessian = inverse.T @ np.diag(hessian_diagonal) @ inverse
        return ExpectationResult(value, grad, hessian)

