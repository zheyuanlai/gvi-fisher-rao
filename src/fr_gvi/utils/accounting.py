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
    # Activations of a published safeguard that is part of a baseline's own
    # definition -- currently only the BBVI--STL projection onto Lambda_S.  It is
    # counted rather than hidden so that a reader can tell how often the
    # comparison rested on it; no method of ours has one.
    projection_activations: int = 0
    # Wall-clock split of the update.  The oracle share is model dependent and
    # equal across methods at equal batch size; the remainder is the dense
    # algebra the schemes actually differ in.
    oracle_seconds: float = 0.0
    linear_algebra_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

