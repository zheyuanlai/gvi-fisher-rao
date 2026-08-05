"""Deterministic and stochastic Gaussian expectation engines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from fr_gvi.linear_algebra.spd import spd_sqrt
from fr_gvi.targets.core import GaussianTarget, ShiftedLogCoshTarget, Target, mean_hessian

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

