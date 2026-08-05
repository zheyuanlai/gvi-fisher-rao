from __future__ import annotations

import numpy as np

from fr_gvi.plotting.aggregation import aggregate_series


def test_seed_trajectories_aggregate_to_median_and_bands() -> None:
    groups = {
        ("job", "FR--R--STL", "1", "variant"): [
            {"iteration": "0", "objective_gap": "1"},
            {"iteration": "1", "objective_gap": "0.2"},
        ],
        ("job", "FR--R--STL", "2", "variant"): [
            {"iteration": "0", "objective_gap": "3"},
            {"iteration": "1", "objective_gap": "0.4"},
        ],
    }
    series = aggregate_series(groups, "objective_gap")
    assert len(series) == 1
    assert series[0].replicates == 2
    np.testing.assert_allclose(series[0].median, [2.0, 0.3])
    np.testing.assert_allclose(series[0].lower, [1.2, 0.22])
    np.testing.assert_allclose(series[0].upper, [2.8, 0.38])

