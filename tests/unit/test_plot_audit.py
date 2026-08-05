from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fr_gvi.plotting.audit import audit_figure_pair


def test_plot_audit_accepts_exact_width_pair(tmp_path: Path) -> None:
    base = tmp_path / "experiment_test"
    figure = plt.figure(figsize=(7.0, 2.0), dpi=220)
    figure.text(0.5, 0.5, "embedded text")
    figure.savefig(base.with_suffix(".png"))
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)
    result = audit_figure_pair(base.with_suffix(".png"), 7.0)
    assert result["errors"] == []

