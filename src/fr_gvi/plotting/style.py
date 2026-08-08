"""Shared publication style, colours and data loading for the figures.

Colours are the Okabe--Ito colourblind-safe palette; every method also carries a
distinct linestyle and marker so the figures survive greyscale printing.
"""

from __future__ import annotations

import re
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

# Hue encodes geometry -- blues and greens for Fisher--Rao, oranges for
# Bures--Wasserstein, purple and pink for parameter space -- so a reader can see
# the two-way comparison in the legend before reading a single label.  Every
# entry is from the Okabe--Ito palette or an equally separable extension of it.
COLORS = {
    "FR--R": "#0072B2",
    "FR--KL": "#009E73",
    "FR--R--STL": "#56B4E9",
    "FR--KL--STL": "#66C2A5",
    "FB--GVI": "#D55E00",
    "S--FB--GVI": "#E69F00",
    "Sq--NGVI": "#332288",
    "Price--BBVI": "#CC79A7",
    "BBVI--STL": "#882255",
    "Laplace": "#555555",
}
LINESTYLES = {
    "FR--R": "-",
    "FR--KL": "--",
    "FR--R--STL": "-",
    "FR--KL--STL": "--",
    "FB--GVI": "-.",
    "S--FB--GVI": (0, (3, 1, 1, 1)),
    "Sq--NGVI": (0, (5, 1)),
    "Price--BBVI": (0, (1, 1)),
    "BBVI--STL": (0, (4, 1, 1, 1, 1, 1)),
    "Laplace": ":",
}
MARKERS = {
    "FR--R": "o",
    "FR--KL": "s",
    "FR--R--STL": "^",
    "FR--KL--STL": "v",
    "FB--GVI": "D",
    "S--FB--GVI": "P",
    "Sq--NGVI": "*",
    "Price--BBVI": "h",
    "BBVI--STL": "<",
    "Laplace": "X",
}
METHOD_ORDER = [
    "FR--R",
    "FR--KL",
    "FR--R--STL",
    "FR--KL--STL",
    "FB--GVI",
    "S--FB--GVI",
    "Sq--NGVI",
    "Price--BBVI",
    "BBVI--STL",
    "Laplace",
]

REFERENCE_GREY = "#4D4D4D"


def configure_style() -> None:
    plt.rcParams.update(
        {
            # Computer Modern, matching the manuscript's article class.  DejaVu's
            # math glyphs are visibly wrong for the symbols this paper leans on:
            # it draws kappa as a heavy near-upright letter and sets beta and
            # lambda far too dark against the surrounding text.  The "cm" fontset
            # ships with matplotlib, so this needs no LaTeX installation; the
            # local toolchain has no dvipng, and requiring one would put a system
            # dependency in the path of every figure regeneration.
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # cmr10 has no ASCII hyphen-minus, so tick labels must be typeset as
            # mathtext for their signs and exponents to come out right.
            "axes.formatter.use_mathtext": True,
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


# Display names.  The stored identifiers stay as they are so raw data, configs and
# manifests remain stable; only what a reader sees is renamed.
#
# ``FB--GVI`` is Diao et al.'s name for their algorithm, but it names only the
# discretization, while the two Fisher--Rao entries name their geometry.  Setting
# the baseline as ``BW-FB`` makes the three labels symmetric -- geometry first,
# then discretization -- so the comparison the figures make is legible from the
# legend alone.  It is deliberately not plain ``BW``: that would name the geometry
# rather than the algorithm, and would suggest the comparison covers the
# Bures--Wasserstein methods this study excludes by design.
DISPLAY_NAMES = {
    "FB--GVI": "BW-FB",
    "FB-GVI": "BW-FB",
    "S--FB--GVI": "S-BW-FB",
    "S-FB-GVI": "S-BW-FB",
}


# The dimension is ``N_\theta`` throughout the manuscript, but the configs, job
# identifiers and most of the plotting code call it ``d``.  A figure that labels
# an axis ``d`` while the surrounding text says ``N_\theta`` makes the reader do
# the translation, so the substitution happens here rather than at eighteen call
# sites -- and cannot be forgotten at the nineteenth.
#
# The rewrite applies inside math mode only.  Prose is full of things that look
# like a lone ``d`` to a regex and are not the dimension: the panel letter ``(d)``,
# the abbreviation ``s.d.``, a job identifier's ``_d10``.  Guarding against each
# in turn is a losing game -- two of those three were found only after they had
# been rendered into a figure -- whereas "it is the dimension when it is a math
# identifier" is the actual rule and needs no exceptions.
#
# Within math, ``d`` still has to stand alone: not part of ``\mathrm{cond}``, not
# a differential ``dx``, not ``\delta``.
_MATH_SEGMENT = re.compile(r"(?<![A-Za-z\\_{])d(?![A-Za-z])")


def _rewrite_dimension(text: str) -> str:
    """Spell the dimension as ``N_\\theta`` in every math span of ``text``."""

    # Odd-indexed pieces of a split on "$" are the math spans.  An unbalanced "$"
    # leaves the tail outside math, which is the safe direction to fail.
    pieces = text.split("$")
    for index in range(1, len(pieces), 2):
        # ``nd^2`` is a product, so its ``d`` follows a letter and the standalone
        # rule will not see it.
        piece = pieces[index].replace("nd^2", r"nN_\theta^2")
        pieces[index] = _MATH_SEGMENT.sub(r"N_\\theta", piece)
    return "$".join(pieces)


def figure_text(value: object) -> str:
    """Return manuscript-visible text with the repository's typographic fixes.

    Four normalisations, all display-only so the underlying identifiers stay put:

    * compound method names are set with single hyphens, while experiment data and
      plotting styles keep their established double-hyphen identifiers;
    * the Bures--Wasserstein entries are named for their geometry, matching the
      geometry-first naming of the Fisher--Rao entries;
    * the dimension is spelled as the manuscript spells it;
    * every ``\\star`` is braced.  TeX classes ``\\star`` as a binary operator, and
      mathtext keeps that operator spacing wherever it appears, which opens a
      visible gap in both ``\\kappa_\\star`` and ``\\lambda_{0,\\star}``.  Bracing
      makes it an ordinary atom and closes the gap.  The braced form is identical
      LaTeX, so caption drafts stay valid when pasted into the manuscript.
    """

    text = str(value)
    # Longest key first so ``S--FB--GVI`` is not partly rewritten by ``FB--GVI``.
    for identifier in sorted(DISPLAY_NAMES, key=len, reverse=True):
        text = text.replace(identifier, DISPLAY_NAMES[identifier])
    text = text.replace("--", "-")
    text = _rewrite_dimension(text)
    # Unbrace first so the rule is idempotent over repeated normalisation.
    return text.replace(r"{\star}", r"\star").replace(r"\star", r"{\star}")


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
    """Across-seed spread, kept subordinate to the median it surrounds.

    Five or more overlapping bands at the previous 0.16 read as a single wash on
    the benchmark panels, and where they crossed, the stacked alpha was darker
    than some of the lines.  The band is context; the median is the measurement.
    """

    axis.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0.0)


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

    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

    for axis in axes:
        for scale, which in ((axis.get_xscale(), axis.xaxis), (axis.get_yscale(), axis.yaxis)):
            if scale != "log":
                continue
            which.set_minor_formatter(NullFormatter())
            # A panel that set its own ticks meant them: the batch-size axis is
            # sampled at powers of two, and a decade locator would label one of
            # the seven points it actually has.
            if isinstance(which.get_major_locator(), FixedLocator):
                continue
            which.set_major_locator(LogLocator(numticks=6))
