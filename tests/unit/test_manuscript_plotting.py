"""The plotting helpers that decide what is convergence and what is roundoff."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fr_gvi.plotting.manuscript_figures import (
    PLATEAU_THRESHOLD,
    _markevery,
    _truncate_at_floor,
    reference_resolution_floor,
)


def test_plateau_is_dropped() -> None:
    """A trajectory that converges and then fluctuates is cut at the plateau."""

    decay = 2.0 * 0.1 ** np.arange(12)
    plateau = np.array([1e-12, 1.4e-12, 0.9e-12, 1.2e-12])
    values = np.concatenate([decay, plateau])
    truncated, level = _truncate_at_floor(values)
    drawn = np.isfinite(truncated)
    assert drawn[: len(decay)].all()
    assert not drawn[len(decay) + 1 :].any()
    assert level <= decay[-1]


def test_early_transient_is_kept() -> None:
    """A non-monotone step early on is part of the trajectory, not a floor.

    At a practical stepsize the objective can rise once before it settles.  A rule
    that cut at the first non-decrease would truncate this after one iteration.
    """

    values = np.array([2.0, 0.70, 0.85, 0.15, 0.13, 3.3e-2, 2.1e-2, 8e-3, 4e-3])
    truncated, _ = _truncate_at_floor(values)
    assert np.isfinite(truncated).all()


def test_analytic_floor_is_a_second_cut() -> None:
    values = np.array([1.0, 1e-3, 1e-6, 1e-9, 1e-12])
    truncated, level = _truncate_at_floor(values, floor=1e-8)
    assert np.isfinite(truncated[:3]).all()
    assert not np.isfinite(truncated[3:]).any()
    assert level == pytest.approx(1e-6)


def test_plateau_threshold_is_relative_to_the_start() -> None:
    """The depth condition scales with the initial value, not with an absolute size."""

    start = 1.0e6
    values = np.array([start, start * 1e-10, start * 1e-10])
    truncated, _ = _truncate_at_floor(values)
    assert np.isfinite(truncated[:2]).all()
    assert not np.isfinite(truncated[2])
    assert PLATEAU_THRESHOLD > 0.0


def test_markevery_phases_by_series() -> None:
    values = np.arange(60, dtype=np.float64)
    first = _markevery(values, index=0)
    second = _markevery(values, index=1)
    assert first[1] == second[1]
    assert first[0] != second[0]
    assert second[0] < second[1]


def test_reference_resolution_floor_uses_the_negative_excursion() -> None:
    """A gap that goes negative measures how far the reference actually resolves."""

    cell = pd.DataFrame(
        {"objective_gap": [1.0, 1e-3, 1e-6, -4.0e-7], "objective": [3.0e2] * 4}
    )
    assert reference_resolution_floor(cell) == pytest.approx(4.0e-7)


def test_reference_resolution_floor_falls_back_to_machine_precision() -> None:
    cell = pd.DataFrame({"objective_gap": [1.0, 1e-3, 1e-9], "objective": [3.0e2] * 3})
    floor = reference_resolution_floor(cell)
    assert 0.0 < floor < 1e-12
