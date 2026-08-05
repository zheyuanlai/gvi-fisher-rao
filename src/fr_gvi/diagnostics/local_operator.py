"""Finite-dimensional assembly of the manuscript's local operator (Eq. 3.2.7)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fr_gvi.algorithms.core import GaussianState
from fr_gvi.linear_algebra.spd import spd_sqrt
from fr_gvi.targets.core import Target

FloatArray = NDArray[np.float64]


def symmetric_star_basis(dimension: int) -> list[FloatArray]:
    basis: list[FloatArray] = []
    for row in range(dimension):
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        matrix[row, row] = np.sqrt(2.0)
        basis.append(matrix)
        for column in range(row):
            matrix = np.zeros((dimension, dimension), dtype=np.float64)
            matrix[row, column] = 1.0
            matrix[column, row] = 1.0
            basis.append(matrix)
    return basis


def assemble_local_operator(
    target: Target,
    optimum: GaussianState,
    normals: FloatArray,
) -> FloatArray:
    """Assemble L*(u,X)=(u+T[X]/2, X+T*[u]+S[X]/2)."""

    dimension = optimum.mean.size
    root = spd_sqrt(optimum.covariance)
    samples = optimum.mean + normals @ root.T
    original_hessians = np.asarray(target.hessian(samples), dtype=np.float64)
    hessians = np.einsum("ij,sjk,kl->sil", root, original_hessians, root)
    covariance_basis = symmetric_star_basis(dimension)
    size = dimension + len(covariance_basis)
    operator = np.zeros((size, size), dtype=np.float64)

    def inner(left_u: FloatArray, left_x: FloatArray, right_u: FloatArray, right_x: FloatArray) -> float:
        return float(left_u @ right_u + 0.5 * np.trace(left_x @ right_x))

    input_basis: list[tuple[FloatArray, FloatArray]] = []
    for index in range(dimension):
        unit = np.zeros(dimension, dtype=np.float64)
        unit[index] = 1.0
        input_basis.append((unit, np.zeros((dimension, dimension), dtype=np.float64)))
    input_basis.extend((np.zeros(dimension, dtype=np.float64), matrix) for matrix in covariance_basis)

    for column, (u, x) in enumerate(input_basis):
        trace_h_x = np.einsum("sij,ji->s", hessians, x)
        t_x = np.mean(trace_h_x[:, None] * normals, axis=0)
        t_star_u = np.mean((normals @ u)[:, None, None] * hessians, axis=0)
        centered_quadratic = np.einsum("si,ij,sj->s", normals, x, normals) - np.trace(x)
        s_x = np.mean(centered_quadratic[:, None, None] * hessians, axis=0)
        output_u = u + 0.5 * t_x
        output_x = x + t_star_u + 0.5 * s_x
        for row, (test_u, test_x) in enumerate(input_basis):
            operator[row, column] = inner(test_u, test_x, output_u, output_x)
    return np.asarray((operator + operator.T) * 0.5, dtype=np.float64)

