"""Shared publication style, colours and data loading for the figures.

Colours are the Okabe--Ito colourblind-safe palette; every method also carries a
distinct linestyle and marker so the figures survive greyscale printing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"
MANIFESTS = ROOT / "results" / "manifests"

# Manuscript text width in inches (article class, 1in margins on US letter).
TEXT_WIDTH = 6.5

COLORS = {
    "FR--R": "#0072B2",
    "FR--KL": "#009E73",
    "FR--R--STL": "#56B4E9",
    "FR--KL--STL": "#66C2A5",
    "FB--GVI": "#D55E00",
    "S--FB--GVI": "#E69F00",
    "Laplace": "#555555",
}
LINESTYLES = {
    "FR--R": "-",
    "FR--KL": "--",
    "FR--R--STL": "-",
    "FR--KL--STL": "--",
    "FB--GVI": "-.",
    "S--FB--GVI": (0, (3, 1, 1, 1)),
    "Laplace": ":",
}
MARKERS = {
    "FR--R": "o",
    "FR--KL": "s",
    "FR--R--STL": "^",
    "FR--KL--STL": "v",
    "FB--GVI": "D",
    "S--FB--GVI": "P",
    "Laplace": "X",
}
METHOD_ORDER = [
    "FR--R",
    "FR--KL",
    "FR--R--STL",
    "FR--KL--STL",
    "FB--GVI",
    "S--FB--GVI",
    "Laplace",
]

REFERENCE_GREY = "#4D4D4D"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_style(method: str, index: int = 0) -> dict[str, object]:
    return {
        "color": COLORS.get(method, f"C{index % 10}"),
        "linestyle": LINESTYLES.get(method, "-"),
        "marker": MARKERS.get(method, "o"),
    }


def load_experiment(experiment: str, tier: str = "full") -> pd.DataFrame:
    """Concatenate every raw trajectory CSV of one experiment."""

    directory = RAW / tier / experiment
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.rglob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(ROOT))
        frame["variant"] = path.stem.rsplit("_seed", 1)[0]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def display_label(row: pd.Series | dict) -> str:
    """Human-readable label including the quadratic-rescue and ablation tags."""

    method = str(row.get("method", ""))
    variant = str(row.get("variant", ""))
    if "-qr" in variant:
        method += "+QR"
    if "-raw" in variant:
        method += " (raw score)"
    return figure_text(method)


def figure_text(value: object) -> str:
    """Return manuscript-visible text with single-hyphen compound names.

    Experiment data and plotting styles deliberately retain their established
    double-hyphen method identifiers. Normalising only display text keeps those
    identifiers stable while enforcing the manuscript typographic convention.
    """

    return str(value).replace("--", "-")


def normalize_figure_text(figure: plt.Figure) -> None:
    """Apply the display-text convention to every text artist in the figure."""

    for artist in figure.findobj(match=Text):
        artist.set_text(figure_text(artist.get_text()))


def positive(values) -> np.ndarray:
    """Clamp a series for log axes without inventing data.

    Non-positive entries become NaN so they are simply absent from the line
    rather than being silently floored to a fake value.
    """

    array = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(array) & (array > 0.0), array, np.nan)


def save_figure(figure: plt.Figure, name: str, caption: str, data: pd.DataFrame | None = None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    base = FIGURES / name
    normalize_figure_text(figure)
    caption = figure_text(caption)
    figure.savefig(base.with_suffix(".pdf"), metadata={"Creator": "fr-gvi"})
    figure.savefig(base.with_suffix(".png"))
    plt.close(figure)
    lines = [f"# {name} caption draft", "", caption, ""]
    if data is not None and not data.empty:
        path = PROCESSED / f"{name}.csv"
        data.to_csv(path, index=False)
        lines.append(f"Underlying data: `{path.relative_to(ROOT)}`.")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def band(axis: plt.Axes, x, lower, upper, color) -> None:
    axis.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0.0)


def seed_summary(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Median and 10--90% band of ``y`` against ``x`` across seeds."""

    grouped = frame.groupby(x)[y]
    return pd.DataFrame(
        {
            x: sorted(frame[x].unique()),
            "median": grouped.median().to_numpy(),
            "lower": grouped.quantile(0.10).to_numpy(),
            "upper": grouped.quantile(0.90).to_numpy(),
            "replicates": grouped.count().to_numpy(),
        }
    )


def terminal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Last recorded iteration of every individual trajectory."""

    keys = ["job_id", "method", "variant", "seed"]
    keys = [key for key in keys if key in frame.columns]
    return frame.sort_values("iteration").groupby(keys, as_index=False).last()


def panel_letter(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(figure_text(f"({letter}) {title}"), loc="left", fontsize=9.0)


def tidy_log_axes(*axes: plt.Axes) -> None:
    """Suppress log minor-tick labels, which collide on narrow decades."""

    from matplotlib.ticker import LogLocator, NullFormatter

    for axis in axes:
        for scale, which in ((axis.get_xscale(), axis.xaxis), (axis.get_yscale(), axis.yaxis)):
            if scale == "log":
                which.set_minor_formatter(NullFormatter())
                which.set_major_locator(LogLocator(numticks=6))
