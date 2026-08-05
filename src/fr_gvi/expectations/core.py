"""Deterministic and stochastic Gaussian expectation engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from fr_gvi.linear_algebra.spd import spd_sqrt
from fr_gvi.targets.core import GaussianTarget, ShiftedLogCoshTarget, Target

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
        hessians = np.asarray(target.hessian(samples), dtype=np.float64)
        return ExpectationResult(
            float(self.weights @ values),
            np.asarray(np.einsum("s,si->i", self.weights, gradients), dtype=np.float64),
            np.asarray(np.einsum("s,sij->ij", self.weights, hessians), dtype=np.float64),
        )


@dataclass(frozen=True)
class GaussHermiteLogCoshExpectation:
    order: int = 32

    def evaluate(self, target: Target, mean: FloatArray, covariance: FloatArray) -> ExpectationResult:
        if not isinstance(target, ShiftedLogCoshTarget):
            raise TypeError("separable Gauss--Hermite backend supports ShiftedLogCoshTarget")
        inverse = target.inverse_transform
        mean_z = inverse @ (mean - target.shift)
        covariance_z = inverse @ covariance @ inverse.T
        nodes, raw_weights = hermgauss(self.order)
        nodes = np.sqrt(2.0) * nodes
        weights = raw_weights / np.sqrt(np.pi)
        value = 0.0
        grad_z = np.zeros(target.dimension, dtype=np.float64)
        hessian_diagonal = np.zeros(target.dimension, dtype=np.float64)
        for index in range(target.dimension):
            samples = mean_z[index] + np.sqrt(covariance_z[index, index]) * nodes
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

