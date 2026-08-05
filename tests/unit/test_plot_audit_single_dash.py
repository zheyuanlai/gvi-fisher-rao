from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from fr_gvi.plotting.audit import audit_figure_pair


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext is unavailable")
def test_plot_audit_rejects_double_dash_pdf_text(tmp_path: Path) -> None:
    base = tmp_path / "experiment_double_dash"
    figure = plt.figure(figsize=(7.0, 2.0), dpi=220)
    figure.text(0.5, 0.5, "FR--R")
    figure.savefig(base.with_suffix(".png"))
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)

    result = audit_figure_pair(base.with_suffix(".png"), 7.0)

    assert any("double-dash figure text" in error for error in result["errors"])
