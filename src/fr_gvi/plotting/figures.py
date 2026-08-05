from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fr_gvi.plotting.aggregation import aggregate_series

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"

COLORS = {
    "FR--R": "#0072B2",
    "FR--KL": "#009E73",
    "FR--R--STL": "#56B4E9",
    "FR--KL--STL": "#CC79A7",
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
    "S--FB--GVI": ":",
    "Laplace": ":",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_rows(raw_root: Path) -> dict[str, list[dict[str, str]]]:
    by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in sorted(raw_root.rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["source_file"] = str(path.resolve().relative_to(ROOT))
                by_experiment[row["experiment"]].append(row)
    return by_experiment


def number(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def write_processed(experiment: str, rows: list[dict[str, str]]) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"experiment_{experiment}.csv"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def group_trajectories(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("job_id", ""),
            row.get("method", ""),
            row.get("seed", ""),
            Path(row.get("source_file", "")).stem.rsplit("_seed", 1)[0],
        )
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: number(row, "iteration", 0.0))
    return groups


def positive(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite_positive = array[np.isfinite(array) & (array > 0.0)]
    floor = max(float(finite_positive.min()) * 0.1, 1.0e-16) if finite_positive.size else 1.0e-16
    return np.where(np.isfinite(array), np.maximum(array, floor), np.nan)


def label_for(key: tuple[str, str, str, str], multiple_jobs: bool) -> str:
    job, method, seed, variant = key
    label = method
    if "-qr" in variant:
        label += "+QR"
    if multiple_jobs:
        label += f" ({job.removesuffix('_core').removesuffix('_smoke')})"
    if seed not in {"", "0"}:
        label += f", seed={seed}"
    return label


def line_panel(
    axis: plt.Axes,
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]],
    y_key: str,
    ylabel: str,
    *,
    log_y: bool = True,
) -> None:
    series_values = aggregate_series(groups, y_key)
    multiple_jobs = len({series.job for series in series_values}) > 1
    for index, series in enumerate(series_values):
        median = positive(list(series.median)) if log_y else series.median
        lower = positive(list(series.lower)) if log_y else series.lower
        upper = positive(list(series.upper)) if log_y else series.upper
        color = COLORS.get(series.method, f"C{index % 10}")
        key = (series.job, series.method, "", series.variant)
        if series.replicates > 1:
            axis.fill_between(series.x, lower, upper, color=color, alpha=0.18, linewidth=0.0)
        axis.plot(
            series.x,
            median,
            label=label_for(key, multiple_jobs),
            color=color,
            linestyle=LINESTYLES.get(series.method, "-"),
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(1, len(series.x) // 7),
            markersize=3.2,
        )
    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")


def make_figure(experiment: str, rows: list[dict[str, str]]) -> tuple[plt.Figure, str]:
    groups = group_trajectories(rows)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    caption = ""
    if experiment == "A":
        line_panel(axes[0], groups, "objective_gap", "Objective gap")
        line_panel(axes[1], groups, "covariance_min_eigenvalue", r"$\lambda_{\min}(C_n)$")
        caption = "Fisher--Rao covariance burn-in: objective localization and entry of the smallest covariance eigenvalue into a stable band."
    elif experiment == "B":
        line_panel(axes[0], groups, "objective_gap", "Exact Gaussian KL gap")
        line_panel(axes[1], groups, "equivariance_error_covariance", "Relative covariance equivariance error")
        caption = "Affine-equivariance diagnostic. Fisher--Rao errors measure direct transformed-iterate agreement; FB--GVI is a geometry-specific comparison."
    elif experiment == "I":
        axes[1].set_visible(False)
        series_values = aggregate_series(
            groups, "stl_raw_variance_ratio", x_key="optimizer_relative_distance"
        )
        multiple_jobs = len({series.job for series in series_values}) > 1
        for index, series in enumerate(series_values):
            median = positive(list(series.median))
            lower = positive(list(series.lower))
            upper = positive(list(series.upper))
            color = f"C{index % 10}"
            if series.replicates > 1:
                axes[0].fill_between(series.x, lower, upper, color=color, alpha=0.18, linewidth=0.0)
            key = (series.job, series.method, "", series.variant)
            axes[0].plot(
                series.x,
                median,
                marker=MARKERS[index % len(MARKERS)],
                color=color,
                label=label_for(key, multiple_jobs),
            )
        axes[0].set_xlabel("Optimizer-relative state distance")
        axes[0].set_ylabel("STL/raw intrinsic variance ratio")
        axes[0].set_yscale("log")
        caption = "Estimator ablation: Fisher--Rao intrinsic conditional mean-estimator variance for STL relative to the raw score estimator."
    elif experiment == "K":
        line_panel(axes[0], groups, "objective_gap", "Expected objective gap")
        series_values = aggregate_series(groups, "objective_gap")
        multiple_jobs = len({series.job for series in series_values}) > 1
        for index, series in enumerate(series_values):
            gap = positive(list(series.median))
            lower = np.maximum(series.x, 1.0) * positive(list(series.lower))
            upper = np.maximum(series.x, 1.0) * positive(list(series.upper))
            color = COLORS.get(series.method, f"C{index % 10}")
            if series.replicates > 1:
                axes[1].fill_between(series.x, lower, upper, color=color, alpha=0.18, linewidth=0.0)
            key = (series.job, series.method, "", series.variant)
            axes[1].plot(
                series.x,
                np.maximum(series.x, 1.0) * gap,
                label=label_for(key, multiple_jobs),
                color=color,
                linestyle=LINESTYLES.get(series.method, "-"),
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(series.x) // 7),
                markersize=3.2,
            )
        axes[1].set_xlabel("Iteration N")
        axes[1].set_ylabel(r"$N\,\Delta_N$")
        caption = "Manuscript decreasing-step schedule: objective gap and the rescaled N times gap diagnostic for the predicted O(1/N) regime."
    elif experiment == "L":
        line_panel(axes[0], groups, "objective_gap", "Gaussian-VI objective gap")
        line_panel(axes[1], groups, "predictive_nll", "Predictive NLL", log_y=False)
        caption = "Bayesian logistic-regression pilot with a proper Gaussian prior. Laplace is a noniterative approximation-quality baseline."
    else:
        line_panel(axes[0], groups, "objective_gap", "Objective gap")
        line_panel(axes[1], groups, "w2_squared", r"Gaussian $W_2^2$ error")
        captions = {
            "C": "Diao et al. anisotropic Gaussian benchmark using the exact target spectrum and exact expectations.",
            "D": "Strongly log-concave shifted log-cosh global comparison under common deterministic expectation points.",
            "F": "Exact standard-Gaussian local-region trajectories for the Fisher--Rao discretizations and FB--GVI.",
            "G": "Near-Gaussian local spectral-rate pilot using the manuscript score-operator definition.",
            "H": "Gaussian STL cancellation with paired samples. This compares complete algorithms and native estimators, not geometry alone.",
            "J": "Minibatch residual-floor pilot showing medians and 10--90% bands while retaining every paired trajectory in the raw data.",
        }
        caption = captions.get(experiment, f"Experiment {experiment} numerical trajectories.")
    visible_axes = [axis for axis in axes if axis.get_visible()]
    handles: list[Any] = []
    labels: list[str] = []
    for axis in visible_axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        if len(labels) > 6:
            figure.legend(handles, labels, loc="outside lower center", ncols=min(4, len(labels)), frameon=False)
        else:
            visible_axes[0].legend(handles, labels, loc="best", frameon=False)
    figure.suptitle(f"Experiment {experiment}", fontsize=11)
    return figure, caption


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--experiments", nargs="*", default=None)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    configure_style()
    by_experiment = read_rows(args.raw_root)
    selected = set(args.experiments) if args.experiments else set(by_experiment)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for experiment in sorted(selected):
        rows = by_experiment.get(experiment, [])
        if not rows:
            continue
        available_tiers = {row.get("tier", "") for row in rows}
        preferred_tier = next(
            tier for tier in ("full", "core", "smoke", "appendix") if tier in available_tiers
        )
        rows = [row for row in rows if row.get("tier", "") == preferred_tier]
        if experiment == "C":
            rows = [row for row in rows if row.get("quadratic_rescue", "False") != "True"]
        if experiment == "F":
            rows = [row for row in rows if row.get("method", "") != "FB--GVI"]
        processed = write_processed(experiment, rows)
        figure, caption = make_figure(experiment, rows)
        base = FIGURES / f"experiment_{experiment}"
        figure.savefig(base.with_suffix(".png"))
        figure.savefig(base.with_suffix(".pdf"), metadata={"Creator": "fr-gvi academic plotting"})
        plt.close(figure)
        base.with_suffix(".md").write_text(
            f"# Experiment {experiment} caption draft\n\n{caption}\n\nUnderlying data: `{processed.relative_to(ROOT)}`.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

