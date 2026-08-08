"""The algorithms admitted by the scientific protocol.

The Fisher--Rao schemes and the Bures--Wasserstein baseline live here; the three
external comparators that work through a square-root factor are in
:mod:`fr_gvi.algorithms.baselines` and are dispatched from :func:`step`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.baselines import (
    FactorBreakdown,
    bbvi_stl_step,
    cholesky_factor,
    price_bbvi_step,
    square_root_natural_gradient_step,
)
from fr_gvi.expectations.core import ExpectationEngine
from fr_gvi.linear_algebra.spd import (
    SPDValidationError,
    ensure_spd,
    jko_entropy_eigenvalue_map,
    spd_exp,
    spd_inverse,
    spd_solve,
    spd_sqrt,
    symmetrize,
)
from fr_gvi.targets.core import Target, mean_hessian
from fr_gvi.utils.accounting import OperationCounts

FloatArray = NDArray[np.float64]


class AlgorithmFailure(RuntimeError):
    """Expected wrapper for a genuine algorithmic numerical failure."""


class Method(StrEnum):
    FR_R = "FR--R"
    FR_KL = "FR--KL"
    FR_R_STL = "FR--R--STL"
    FR_KL_STL = "FR--KL--STL"
    FB_GVI = "FB--GVI"
    S_FB_GVI = "S--FB--GVI"
    SQ_NGVI = "Sq--NGVI"
    PRICE_BBVI = "Price--BBVI"
    BBVI_STL = "BBVI--STL"
    LAPLACE = "Laplace"

    @property
    def stochastic(self) -> bool:
        return self in {
            self.FR_R_STL,
            self.FR_KL_STL,
            self.S_FB_GVI,
            self.PRICE_BBVI,
            self.BBVI_STL,
        }

    @property
    def geometry(self) -> str:
        """Which geometry the update is a descent step in.

        The manuscript's comparison is two-way -- geometry against estimator --
        so both axes are properties of the method rather than facts a plotting
        script has to re-derive from its name.
        """

        if self in {self.FR_R, self.FR_KL, self.FR_R_STL, self.FR_KL_STL}:
            return "fisher-rao"
        if self in {self.FB_GVI, self.S_FB_GVI}:
            return "bures-wasserstein"
        if self is self.SQ_NGVI:
            return "fisher-rao-square-root"
        if self in {self.PRICE_BBVI, self.BBVI_STL}:
            return "parameter-space"
        return "none"

    @property
    def estimator(self) -> str:
        """Which oracle the covariance direction is built from."""

        if self in {self.FR_R_STL, self.FR_KL_STL, self.S_FB_GVI, self.PRICE_BBVI}:
            return "price-hessian"
        if self is self.BBVI_STL:
            return "reparameterization"
        if self is self.LAPLACE:
            return "none"
        return "exact"

    @property
    def uses_hessian(self) -> bool:
        return self.estimator in {"price-hessian", "exact"} and self is not self.LAPLACE


@dataclass(frozen=True)
class GaussianState:
    mean: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance, _ = ensure_spd(np.asarray(self.covariance, dtype=np.float64))
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("covariance shape does not match mean")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class StepDiagnostics:
    repair: dict[str, float] | None
    gradient: FloatArray
    hessian: FloatArray
    # Cost accounting for the scaling study.  ``oracle_seconds`` is the time
    # spent inside the target's own gradient and Hessian evaluation, which is
    # model dependent and identical across methods at equal batch size;
    # ``linear_algebra_seconds`` is the remaining dense work, which is what the
    # matrix exponential, the resolvent solve and the Cholesky update actually
    # differ in.  Reporting only their sum hides which of the two a method's
    # per-iteration cost is made of.
    oracle_seconds: float = 0.0
    linear_algebra_seconds: float = 0.0
    projection_activations: int = 0


def _validated_state(
    mean: FloatArray,
    covariance: FloatArray,
    counts: OperationCounts,
) -> tuple[GaussianState, dict[str, float] | None]:
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise AlgorithmFailure("NaN or Inf encountered in iterate")
    try:
        covariance, repair = ensure_spd(covariance)
    except SPDValidationError as exc:
        raise AlgorithmFailure(str(exc)) from exc
    repair_dict: dict[str, float] | None = None
    if repair is not None:
        counts.roundoff_repairs += 1
        repair_dict = {
            "minimum_eigenvalue": repair.minimum_eigenvalue,
            "tolerance": repair.tolerance,
            "replacement": repair.replacement,
        }
    return GaussianState(mean, covariance), repair_dict


def quadratic_rescue(target: Target, input_mean: FloatArray) -> GaussianState:
    gradient = np.asarray(target.grad(input_mean), dtype=np.float64)
    hessian = np.asarray(target.hessian(input_mean), dtype=np.float64)
    mean = input_mean - spd_solve(hessian, gradient)
    covariance = spd_inverse(hessian)
    return GaussianState(mean, covariance)


def _population(
    target: Target,
    state: GaussianState,
    engine: ExpectationEngine,
    counts: OperationCounts,
    timer: "_OracleTimer",
) -> tuple[FloatArray, FloatArray]:
    with timer:
        result = engine.evaluate(target, state.mean, state.covariance)
    counts.expectation_evaluations += 1
    counts.gradient_evaluations += 1
    counts.hessian_evaluations += 1
    counts.oracle_pairs += 1
    return np.asarray(result.grad, dtype=np.float64), np.asarray(result.hessian, dtype=np.float64)


class _OracleTimer:
    """Accumulates the time spent inside target gradient/Hessian evaluation."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self._started = 0.0

    def __enter__(self) -> "_OracleTimer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.seconds += time.perf_counter() - self._started


@dataclass(frozen=True)
class _Batch:
    normals: FloatArray
    samples: FloatArray
    gradients: FloatArray
    hessian: FloatArray | None
    factor: FloatArray


def _draw(
    target: Target,
    state: GaussianState,
    rng: np.random.Generator,
    batch_size: int,
    *,
    counts: OperationCounts,
    timer: _OracleTimer,
    triangular_root: bool,
    need_hessian: bool,
) -> _Batch:
    """One Monte Carlo batch, with the root the method is defined against.

    The Fisher--Rao and Bures--Wasserstein schemes sample through the symmetric
    root, the parameter-space schemes through the Cholesky factor they optimize.
    Both give ``X ~ N(m, C)``, so the estimators remain unbiased for the same
    population quantities; the factor is returned because the parameter-space
    updates need the very same one they sampled with.
    """

    if triangular_root:
        root = cholesky_factor(state.covariance)
        counts.cholesky_factorizations += 1
    else:
        root = spd_sqrt(state.covariance)
        counts.matrix_square_roots += 1
        counts.eigendecompositions += 1
    normals = rng.standard_normal((batch_size, state.mean.size), dtype=np.float64)
    samples = state.mean + normals @ root.T
    with timer:
        gradients = np.asarray(target.grad(samples), dtype=np.float64)
        hessian = mean_hessian(target, samples) if need_hessian else None
    counts.gradient_evaluations += batch_size
    if need_hessian:
        counts.hessian_evaluations += batch_size
        counts.oracle_pairs += batch_size
    return _Batch(normals, samples, gradients, hessian, root)


def _sampled(
    target: Target,
    state: GaussianState,
    rng: np.random.Generator,
    batch_size: int,
    *,
    stl: bool,
    counts: OperationCounts,
    timer: _OracleTimer | None = None,
) -> tuple[FloatArray, FloatArray]:
    batch = _draw(
        target,
        state,
        rng,
        batch_size,
        counts=counts,
        timer=timer if timer is not None else _OracleTimer(),
        triangular_root=False,
        need_hessian=True,
    )
    assert batch.hessian is not None
    if stl:
        centered = batch.samples - state.mean
        score = spd_solve(state.covariance, centered.T).T
        counts.cholesky_factorizations += 1
        counts.cholesky_solves += 1
        mean_direction = np.mean(-batch.gradients + score, axis=0)
    else:
        mean_direction = np.mean(batch.gradients, axis=0)
    return np.asarray(mean_direction), batch.hessian


def step(
    method: Method,
    target: Target,
    state: GaussianState,
    step_size: float,
    *,
    engine: ExpectationEngine | None,
    rng: np.random.Generator,
    batch_size: int,
    counts: OperationCounts,
    raw_mean_ablation: bool = False,
    projection_floor: float | None = None,
) -> tuple[GaussianState, StepDiagnostics]:
    if step_size <= 0.0:
        raise ValueError("step size must be positive")
    if method is Method.BBVI_STL and projection_floor is None:
        raise ValueError(
            "BBVI--STL is defined on Lambda_S and needs its projection floor "
            "1/sqrt(S); pass the target's log-smoothness constant"
        )
    covariance = state.covariance
    identity = np.eye(state.mean.size, dtype=np.float64)
    timer = _OracleTimer()
    started = time.perf_counter()
    activations = 0

    if method in {Method.FR_R, Method.FR_KL, Method.FB_GVI, Method.SQ_NGVI}:
        if engine is None:
            raise ValueError("deterministic method requires an expectation engine")
        expected_gradient, hessian = _population(target, state, engine, counts, timer)
        if method in {Method.FR_R, Method.FR_KL}:
            mean_direction = -expected_gradient
        else:
            mean_direction = expected_gradient
    elif method in {Method.FR_R_STL, Method.FR_KL_STL}:
        mean_direction, hessian = _sampled(
            target,
            state,
            rng,
            batch_size,
            stl=not raw_mean_ablation,
            counts=counts,
            timer=timer,
        )
        if raw_mean_ablation:
            mean_direction = -mean_direction
        expected_gradient = -mean_direction
    elif method == Method.S_FB_GVI:
        mean_direction, hessian = _sampled(
            target, state, rng, batch_size, stl=False, counts=counts, timer=timer
        )
        expected_gradient = mean_direction
    elif method in {Method.PRICE_BBVI, Method.BBVI_STL}:
        try:
            batch = _draw(
                target,
                state,
                rng,
                batch_size,
                counts=counts,
                timer=timer,
                triangular_root=True,
                need_hessian=method is Method.PRICE_BBVI,
            )
        except FactorBreakdown as exc:
            raise AlgorithmFailure(f"{method} sampling failed: {exc}") from exc
        expected_gradient = np.mean(batch.gradients, axis=0)
        mean_direction = expected_gradient
        hessian = batch.hessian if batch.hessian is not None else identity
    else:
        raise ValueError(f"{method} does not have an iterative step")

    try:
        if method in {Method.FR_R, Method.FR_R_STL}:
            mean_new = state.mean + step_size * covariance @ mean_direction
            root = spd_sqrt(covariance)
            tangent = symmetrize(identity - root @ hessian @ root)
            covariance_new = symmetrize(root @ spd_exp(step_size * tangent) @ root)
            counts.matrix_square_roots += 1
            counts.matrix_exponentials += 1
            counts.eigendecompositions += 2
        elif method in {Method.FR_KL, Method.FR_KL_STL}:
            mean_new = state.mean + step_size * covariance @ mean_direction
            precision = spd_inverse(covariance)
            covariance_new = (1.0 + step_size) * spd_solve(
                precision + step_size * hessian, identity
            )
            covariance_new = symmetrize(covariance_new)
            counts.cholesky_factorizations += 2
            counts.cholesky_solves += 2
        elif method is Method.SQ_NGVI:
            mean_new, covariance_new = square_root_natural_gradient_step(
                state.mean, covariance, expected_gradient, hessian, step_size
            )
            counts.cholesky_factorizations += 1
        elif method is Method.PRICE_BBVI:
            mean_new, covariance_new = price_bbvi_step(
                state.mean, covariance, expected_gradient, hessian, step_size
            )
        elif method is Method.BBVI_STL:
            assert projection_floor is not None
            mean_new, covariance_new, activations = bbvi_stl_step(
                state.mean,
                covariance,
                batch.normals,
                batch.gradients,
                step_size,
                projection_floor,
            )
            counts.cholesky_solves += 1
            counts.projection_activations += activations
        else:
            mean_new = state.mean - step_size * mean_direction
            forward_map = identity - step_size * hessian
            half_covariance = symmetrize(forward_map @ covariance @ forward_map)
            values, vectors = np.linalg.eigh(half_covariance)
            scale = max(float(np.max(np.abs(values))), 1.0)
            tolerance = 100.0 * np.finfo(np.float64).eps * scale
            if float(values[0]) < -tolerance:
                raise SPDValidationError(
                    f"FB--GVI forward covariance invalid: lambda_min={values[0]:.6e}"
                )
            values = np.maximum(values, 0.0)
            updated_values = jko_entropy_eigenvalue_map(values, step_size)
            covariance_new = symmetrize((vectors * updated_values) @ vectors.T)
            counts.eigendecompositions += 1
            counts.matrix_square_roots += 1
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, SPDValidationError) as exc:
        raise AlgorithmFailure(f"{method} update failed: {exc}") from exc

    counts.iterations += 1
    next_state, repair = _validated_state(mean_new, covariance_new, counts)
    total = time.perf_counter() - started
    counts.oracle_seconds += timer.seconds
    counts.linear_algebra_seconds += max(total - timer.seconds, 0.0)
    return next_state, StepDiagnostics(
        repair,
        expected_gradient,
        hessian,
        oracle_seconds=timer.seconds,
        linear_algebra_seconds=max(total - timer.seconds, 0.0),
        projection_activations=activations,
    )

