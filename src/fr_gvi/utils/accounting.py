from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OperationCounts:
    iterations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    oracle_pairs: int = 0
    expectation_evaluations: int = 0
    matrix_exponentials: int = 0
    matrix_square_roots: int = 0
    eigendecompositions: int = 0
    cholesky_factorizations: int = 0
    cholesky_solves: int = 0
    roundoff_repairs: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

