"""Potential functions used by the experiment campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from fr_gvi.linear_algebra.spd import ensure_spd, spd_inverse

FloatArray = NDArray[np.float64]


class Target(Protocol):
    dimension: int

    def value(self, x: FloatArray) -> FloatArray: ...

    def grad(self, x: FloatArray) -> FloatArray: ...

    def hessian(self, x: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class GaussianTarget:
    mean: FloatArray
    precision: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        precision, _ = ensure_spd(np.asarray(self.precision, dtype=np.float64))
        if precision.shape != (mean.size, mean.size):
            raise ValueError("precision shape does not match target mean")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "precision", precision)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    @property
    def covariance(self) -> FloatArray:
        return spd_inverse(self.precision)

    def value(self, x: FloatArray) -> FloatArray:
        delta = np.asarray(x, dtype=np.float64) - self.mean
        return np.asarray(0.5 * np.einsum("...i,ij,...j->...", delta, self.precision, delta))

    def grad(self, x: FloatArray) -> FloatArray:
        delta = np.asarray(x, dtype=np.float64) - self.mean
        return np.asarray(np.einsum("ij,...j->...i", self.precision, delta), dtype=np.float64)

    def hessian(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return self.precision.copy()
        return np.broadcast_to(self.precision, x.shape[:-1] + self.precision.shape).copy()


@dataclass(frozen=True)
class ShiftedLogCoshTarget:
    """Affine image of a separable shifted log-cosh target.

    The coordinate convention is x = transform @ z + shift.
    """

    nu: FloatArray
    rho: float
    offset: FloatArray
    transform: FloatArray
    shift: FloatArray

    def __post_init__(self) -> None:
        nu = np.asarray(self.nu, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        transform = np.asarray(self.transform, dtype=np.float64)
        shift = np.asarray(self.shift, dtype=np.float64)
        dimension = nu.size
        if np.any(nu <= 0.0) or self.rho < 0.0:
            raise ValueError("log-cosh target requires nu > 0 and rho >= 0")
        if offset.shape != (dimension,) or shift.shape != (dimension,):
            raise ValueError("offset and shift must match nu")
        if transform.shape != (dimension, dimension):
            raise ValueError("transform shape must be d x d")
        if abs(float(np.linalg.det(transform))) < np.finfo(np.float64).eps:
            raise ValueError("affine transform must be invertible")
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "shift", shift)

    @classmethod
    def base(
        cls,
        dimension: int,
        *,
        nu: float | FloatArray = 1.0,
        rho: float = 1.0,
        offset: float | FloatArray = 0.5,
    ) -> "ShiftedLogCoshTarget":
        nu_array = np.broadcast_to(np.asarray(nu, dtype=np.float64), (dimension,)).copy()
        offset_array = np.broadcast_to(np.asarray(offset, dtype=np.float64), (dimension,)).copy()
        return cls(nu_array, rho, offset_array, np.eye(dimension), np.zeros(dimension))

    @property
    def dimension(self) -> int:
        return int(self.nu.size)

    @property
    def inverse_transform(self) -> FloatArray:
        return np.asarray(np.linalg.solve(self.transform, np.eye(self.dimension)), dtype=np.float64)

    @staticmethod
    def _logcosh(x: FloatArray) -> FloatArray:
        absolute = np.abs(x)
        return absolute + np.log1p(np.exp(-2.0 * absolute)) - np.log(2.0)

    def _z(self, x: FloatArray) -> FloatArray:
        delta = np.asarray(x, dtype=np.float64) - self.shift
        return np.asarray(np.einsum("ij,...j->...i", self.inverse_transform, delta), dtype=np.float64)

    def value(self, x: FloatArray) -> FloatArray:
        z = self._z(x)
        return np.asarray(
            np.sum(0.5 * self.nu * z * z + self.rho * self._logcosh(z - self.offset), axis=-1),
            dtype=np.float64,
        )

    def grad(self, x: FloatArray) -> FloatArray:
        z = self._z(x)
        grad_z = self.nu * z + self.rho * np.tanh(z - self.offset)
        return np.asarray(np.einsum("ji,...j->...i", self.inverse_transform, grad_z), dtype=np.float64)

    def hessian(self, x: FloatArray) -> FloatArray:
        z = self._z(x)
        diagonal = self.nu + self.rho * (1.0 - np.tanh(z - self.offset) ** 2)
        inverse = self.inverse_transform
        return np.asarray(np.einsum("ki,...k,kj->...ij", inverse, diagonal, inverse), dtype=np.float64)


@dataclass(frozen=True)
class LogisticRegressionTarget:
    features: FloatArray
    labels: FloatArray
    prior_precision: float

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.float64)
        if features.ndim != 2 or labels.shape != (features.shape[0],):
            raise ValueError("features must be n x d and labels length n")
        if self.prior_precision < 0.0:
            raise ValueError("prior precision cannot be negative")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    def value(self, theta: FloatArray) -> FloatArray:
        theta = np.asarray(theta, dtype=np.float64)
        logits = np.einsum("nd,...d->...n", self.features, theta)
        likelihood = np.sum(np.logaddexp(0.0, logits) - self.labels * logits, axis=-1)
        prior = 0.5 * self.prior_precision * np.sum(theta * theta, axis=-1)
        return np.asarray(prior + likelihood, dtype=np.float64)

    def grad(self, theta: FloatArray) -> FloatArray:
        theta = np.asarray(theta, dtype=np.float64)
        logits = np.einsum("nd,...d->...n", self.features, theta)
        residual = expit(logits) - self.labels
        return np.asarray(
            self.prior_precision * theta + np.einsum("...n,nd->...d", residual, self.features),
            dtype=np.float64,
        )

    def hessian(self, theta: FloatArray) -> FloatArray:
        theta = np.asarray(theta, dtype=np.float64)
        logits = np.einsum("nd,...d->...n", self.features, theta)
        probabilities = expit(logits)
        weights = probabilities * (1.0 - probabilities)
        hessian = np.einsum("...n,ni,nj->...ij", weights, self.features, self.features)
        identity = np.eye(self.dimension, dtype=np.float64)
        return np.asarray(hessian + self.prior_precision * identity, dtype=np.float64)

