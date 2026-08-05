from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregateSeries:
    job: str
    method: str
    variant: str
    x: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    replicates: int


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan


def aggregate_series(
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]],
    y_key: str,
    *,
    x_key: str = "iteration",
) -> list[AggregateSeries]:
    collections: dict[tuple[str, str, str], list[list[dict[str, str]]]] = defaultdict(list)
    for (job, method, _seed, variant), trajectory in groups.items():
        collections[(job, method, variant)].append(trajectory)
    output: list[AggregateSeries] = []
    for (job, method, variant), trajectories in sorted(collections.items()):
        maps = [
            {_number(row, x_key): _number(row, y_key) for row in trajectory}
            for trajectory in trajectories
        ]
        common = sorted(set.intersection(*(set(values) for values in maps)))
        matrix = np.asarray([[values[x] for x in common] for values in maps], dtype=np.float64)
        output.append(
            AggregateSeries(
                job=job,
                method=method,
                variant=variant,
                x=np.asarray(common, dtype=np.float64),
                median=np.nanmedian(matrix, axis=0),
                lower=np.nanpercentile(matrix, 10.0, axis=0),
                upper=np.nanpercentile(matrix, 90.0, axis=0),
                replicates=len(trajectories),
            )
        )
    return output

