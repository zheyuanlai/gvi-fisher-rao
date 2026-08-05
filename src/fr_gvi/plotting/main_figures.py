from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fr_gvi.plotting.aggregation import AggregateSeries, aggregate_series
from fr_gvi.plotting.main_processed import write_main_processed
from fr_gvi.plotting.figures import (
    COLORS,
    LINESTYLES,
    MARKERS,
    FIGURES,
    RAW,
    ROOT,
    configure_style,
    group_trajectories,
    positive,
    read_rows,
)

METHOD_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "FR--R",
            "FR--KL",
            "FR--R--STL",
            "FR--KL--STL",
            "FB--GVI",
            "S--FB--GVI",
            "Laplace",
        )
    )
}


def _number(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _core(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("tier") == "core"]


def _series(
    rows: list[dict[str, str]], y_key: str, *, x_key: str = "iteration"
) -> list[AggregateSeries]:
    values = aggregate_series(group_trajectories(rows), y_key, x_key=x_key)
    return sorted(values, key=lambda item: (METHOD_ORDER.get(item.method, 99), item.job))


def _method_label(item: AggregateSeries) -> str:
    return item.method


def _plot_series(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    y_key: str,
    *,
    x_key: str = "iteration",
    xlabel: str,
    ylabel: str,
    log_y: bool = True,
    log_x: bool = False,
    laplace_line: bool = False,
) -> None:
    for index, item in enumerate(_series(rows, y_key, x_key=x_key)):
        color = COLORS.get(item.method, f"C{index % 10}")
        y = positive(list(item.median)) if log_y else np.asarray(item.median)
        lower = positive(list(item.lower)) if log_y else np.asarray(item.lower)
        upper = positive(list(item.upper)) if log_y else np.asarray(item.upper)
        finite = np.isfinite(item.x) & np.isfinite(y)
        if laplace_line and item.method == "Laplace" and np.any(finite):
            axis.axhline(
                float(y[finite][-1]), color=color, linestyle=LINESTYLES[item.method],
                linewidth=1.5, label="Laplace",
            )
            continue
        if item.replicates > 1:
            axis.fill_between(item.x, lower, upper, color=color, alpha=0.18, linewidth=0.0)
        axis.plot(
            item.x,
            y,
            color=color,
            linestyle=LINESTYLES.get(item.method, "-"),
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(1, len(item.x) // 6),
            markersize=3.0,
            label=_method_label(item),
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    if log_x:
        axis.set_xscale("log")


def _deduplicated_legend(figure: plt.Figure, axes: Iterable[plt.Axes], ncols: int = 4) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        current_handles, current_labels = axis.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels, strict=False):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=min(ncols, len(labels)),
            frameon=False,
        )


def _save(figure: plt.Figure, number: int, caption: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / f"main_figure_{number}"
    figure.savefig(base.with_suffix(".png"))
    figure.savefig(base.with_suffix(".pdf"), metadata={"Creator": "fr-gvi academic plotting"})
    plt.close(figure)
    base.with_suffix(".md").write_text(
        f"# Main Figure {number} caption draft\n\n{caption}\n",
        encoding="utf-8",
    )


def _panel_titles(axes: Iterable[plt.Axes], titles: Iterable[str]) -> None:
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, loc="left")


def figure_1(by_experiment: dict[str, list[dict[str, str]]]) -> None:
    figure, axes_array = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    axes = axes_array.ravel()

    burn_records: list[tuple[str, float, float]] = []
    for path in sorted((ROOT / "results" / "manifests" / "core").glob("A_burnin_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("burn_in_iteration") is None:
            continue
        config = record["config"]
        scale = float(config["target"]["initial_covariance_scale"])
        beta = float(config["target"]["condition"])
        step = float(record["method_specification"]["step_size"])
        burn_records.append(
            (
                str(record["method_specification"]["name"]),
                float(np.log(1.0 / (beta * scale))),
                float(record["burn_in_iteration"]) * step,
            )
        )
    for index, method in enumerate(("FR--R", "FR--KL")):
        values = sorted((x, y) for name, x, y in burn_records if name == method)
        if values:
            x, y = np.asarray(values).T
            axes[0].plot(
                x, y, marker=MARKERS[index], color=COLORS[method],
                linestyle=LINESTYLES[method], label=method,
            )
    if burn_records:
        limit = np.asarray([min(value[1] for value in burn_records), max(value[1] for value in burn_records)])
        axes[0].plot(limit, limit, color="#444444", linestyle=":", label="log envelope")
    axes[0].set_xlabel(r"$\log[1/(\beta\lambda_0)]$")
    axes[0].set_ylabel(r"$N_{\rm cov}h$")

    hard_rows = [
        row
        for row in _core(by_experiment.get("A", []))
        if "lam1e12" in row.get("job_id", "") and row.get("method") == "FR--R"
    ]
    hard_rows.sort(key=lambda row: _number(row, "iteration"))
    if hard_rows:
        n = np.asarray([_number(row, "iteration") for row in hard_rows])
        state = np.sqrt(
            np.asarray([_number(row, "mean_error") for row in hard_rows]) ** 2
            + np.asarray([_number(row, "covariance_error") for row in hard_rows]) ** 2
        )
        trajectories = (
            ("relative_covariance_min_eigenvalue", r"$\lambda_{\min}(C_\star^{-1/2}C_nC_\star^{-1/2})$"),
            ("objective_gap", r"$\Delta_n$"),
            ("fisher_rao_residual_squared", r"$\|\operatorname{grad}\mathcal{E}\|_{\rm FR}$"),
        )
        for index, (key, label) in enumerate(trajectories):
            values = np.asarray([_number(row, key) for row in hard_rows])
            if key.endswith("squared"):
                values = np.sqrt(np.maximum(values, 0.0))
            axes[1].plot(n, positive(list(values)), label=label, marker=MARKERS[index],
                         markevery=max(1, len(n) // 7), markersize=2.8)
        axes[1].plot(n, positive(list(state)), label=r"$r_\star$", color="#CC79A7", linestyle="--")
        markers = [(207, "cov. band")]
        for threshold, label in ((1.0, r"$r_\star\leq1$"), (0.1, r"$r_\star\leq0.1$")):
            indices = np.flatnonzero(state <= threshold)
            if indices.size:
                markers.append((int(n[indices[0]]), label))
        for location, label in markers:
            axes[1].axvline(location, color="#777777", linewidth=0.8, linestyle=":")
            axes[1].text(location, 0.03, label, rotation=90, transform=axes[1].get_xaxis_transform(),
                         ha="right", va="bottom", fontsize=6.5, color="#555555")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Diagnostic value")

    _plot_series(
        axes[2], _core(by_experiment.get("B", [])), "equivariance_error_covariance",
        xlabel="Iteration", ylabel="Relative covariance discrepancy",
    )
    _plot_series(
        axes[3], [row for row in _core(by_experiment.get("C", [])) if row.get("quadratic_rescue") != "True"],
        "objective_gap", xlabel="Oracle pairs", x_key="oracle_pairs", ylabel="Exact Gaussian KL gap",
    )
    _panel_titles(
        axes,
        (
            "(a) Logarithmic covariance burn-in",
            "(b) Hard-start three-stage diagnostics",
            "(c) Affine-equivariance error, $K=10^4$",
            "(d) Diao anisotropic Gaussian pilot",
        ),
    )
    _deduplicated_legend(figure, axes)
    _save(
        figure,
        1,
        "Gaussian geometry and initialization. (a) Recorded covariance-band entry times follow the logarithmic envelope. "
        "(b) The hardest core initialization shows covariance, objective, Fisher--Rao residual, and optimizer-relative stages; "
        "the two radius thresholds are operational diagnostics, not theorem constants. (c) Direct transformed-iterate errors "
        "test affine equivariance. (d) The competitive Diao benchmark excludes the one-query quadratic-rescue verification.",
    )


def figure_2(by_experiment: dict[str, list[dict[str, str]]]) -> None:
    rows = _core(by_experiment.get("D", []))
    figure, axes_array = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    axes = axes_array.ravel()
    _plot_series(axes[0], rows, "objective_gap", xlabel="Iteration", ylabel="Objective gap")
    _plot_series(
        axes[1], rows, "objective_gap", x_key="wall_time_seconds",
        xlabel="Wall time (s)", ylabel="Objective gap",
    )
    for index, item in enumerate(_series(rows, "mean_error")):
        color = COLORS.get(item.method, f"C{index}")
        axes[2].plot(item.x, positive(list(item.median)), color=color,
                     linestyle="-", label=f"{item.method}: mean")
    for index, item in enumerate(_series(rows, "covariance_error")):
        color = COLORS.get(item.method, f"C{index}")
        axes[2].plot(item.x, positive(list(item.median)), color=color,
                     linestyle=":", label=f"{item.method}: covariance")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Optimizer-relative error")

    terminal: dict[str, tuple[float, float, float]] = {}
    for item in _series(rows, "objective_gap"):
        initial = float(item.median[0])
        final = float(item.median[-1])
        step_rows = [row for row in rows if row.get("method") == item.method]
        step = _number(step_rows[-1], "normalized_step_size") if step_rows else np.nan
        terminal[item.method] = (initial, final, step)
    methods = [name for name in ("FR--R", "FR--KL", "FB--GVI") if name in terminal]
    status = np.asarray(
        [[2 if terminal[name][1] <= 1.0e-6 else 1 if terminal[name][1] < terminal[name][0] else 0] for name in methods]
    )
    from matplotlib.colors import ListedColormap

    axes[3].imshow(status, aspect="auto", vmin=0, vmax=2,
                   cmap=ListedColormap(["#D55E00", "#F0E442", "#009E73"]))
    axes[3].set_xticks([0], ["theory-scale core cell"])
    axes[3].set_yticks(range(len(methods)), methods)
    for row_index, name in enumerate(methods):
        axes[3].text(0, row_index, f"stable; h={terminal[name][2]:g}", ha="center", va="center", fontsize=7)
    axes[3].text(0.02, -0.18, "Full logarithmic stepsize grid: pending full tier",
                 transform=axes[3].transAxes, fontsize=6.5, color="#555555")
    _panel_titles(
        axes,
        (
            "(a) Objective versus iteration",
            "(b) Objective versus wall time",
            "(c) Mean and covariance errors",
            "(d) Stepsize-status pilot",
        ),
    )
    _deduplicated_legend(figure, axes[:3], ncols=3)
    _save(
        figure,
        2,
        "Global shifted log-cosh core pilot (d=10, intrinsic condition 100). Solid and dotted curves in (c) "
        "separate mean and covariance error. Panel (d) reports only the completed theory-scale cell per method; "
        "the planned logarithmic stability sweep belongs to the unrun full tier.",
    )


def _observed_rate(rows: list[dict[str, str]]) -> tuple[float, float] | None:
    ordered = sorted(rows, key=lambda row: _number(row, "iteration"))
    if len(ordered) < 8:
        return None
    time = np.asarray([_number(row, "iteration") * _number(row, "step_size") for row in ordered])
    gap = np.asarray([_number(row, "objective_gap") for row in ordered])
    finite = np.isfinite(time) & np.isfinite(gap) & (gap > 0.0)
    time, gap = time[finite], gap[finite]
    start = len(time) // 2
    if len(time[start:]) < 4:
        return None
    slope = np.polyfit(time[start:], np.log(gap[start:]), deg=1)[0]
    gamma = _number(ordered[-1], "local_gamma")
    return 2.0 * gamma, -float(slope)


def figure_3(by_experiment: dict[str, list[dict[str, str]]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.4), constrained_layout=True)
    axes[0].axis("off")
    axes[0].text(
        0.5,
        0.57,
        "Bump-train experiment\nnot executed",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    axes[0].text(
        0.5,
        0.34,
        "The manuscript and supplied plan do not\nspecify the theorem's potential formula.\nNo surrogate was invented.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    _plot_series(
        axes[1], [row for row in _core(by_experiment.get("F", [])) if row.get("method") != "FB--GVI"],
        "objective_gap", xlabel="Iteration", ylabel="Exact Gaussian KL gap",
    )
    rates: list[tuple[str, float, float]] = []
    grouped = group_trajectories(_core(by_experiment.get("G", [])))
    for (_job, method, _seed, _variant), trajectory in grouped.items():
        value = _observed_rate(trajectory)
        if value is not None:
            rates.append((method, *value))
    for index, (method, prediction, observed) in enumerate(sorted(rates, key=lambda item: METHOD_ORDER.get(item[0], 99))):
        axes[2].scatter(prediction, observed, s=40, marker=MARKERS[index],
                        color=COLORS.get(method, f"C{index}"), label=method, zorder=3)
    if rates:
        bounds = np.asarray([min(value for _, predicted, observed in rates for value in (predicted, observed)),
                             max(value for _, predicted, observed in rates for value in (predicted, observed))])
        padding = max(0.05, 0.08 * float(np.ptp(bounds)))
        bounds = np.asarray([bounds[0] - padding, bounds[1] + padding])
        axes[2].plot(bounds, bounds, color="#555555", linestyle=":", label="observed = predicted")
        axes[2].set_xlim(bounds)
        axes[2].set_ylim(bounds)
    axes[2].set_xlabel(r"Predicted $2\gamma_\star$")
    axes[2].set_ylabel("Fitted objective rate")
    _panel_titles(
        axes,
        (
            r"(a) $N_{1/2}$ versus $\kappa$",
            "(b) Exact Gaussian local trajectory",
            "(c) Spectral-rate pilot",
        ),
    )
    _deduplicated_legend(figure, axes, ncols=3)
    _save(
        figure,
        3,
        "Sharp-mechanism evidence. The bump-train cell is explicitly blocked because its defining potential is absent "
        "from the supplied sources. Panel (b) uses a nontrivial covariance perturbation and excludes FB--GVI because "
        "the mechanism is Fisher--Rao specific. Panel (c) fits the final half of each objective trajectory in algorithmic time.",
    )


def _terminal_floors(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float, float, float]]]:
    samples: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (_job, method, _seed, _variant), trajectory in group_trajectories(rows).items():
        ordered = sorted(trajectory, key=lambda row: _number(row, "iteration"))
        tail = ordered[max(0, int(0.8 * len(ordered))):]
        batch = _number(ordered[-1], "batch_size")
        values = positive([_number(row, "objective_gap") for row in tail])
        samples[(method, batch)].append(float(np.nanmedian(values)))
    output: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for (method, batch), values in samples.items():
        output[method].append(
            (batch, float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90)))
        )
    return output


def figure_4(by_experiment: dict[str, list[dict[str, str]]]) -> None:
    figure, axes_array = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    axes = axes_array.ravel()
    variance = _core(by_experiment.get("I", []))
    for index, item in enumerate(_series(variance, "stl_raw_variance_ratio", x_key="optimizer_relative_distance")):
        axes[0].fill_between(item.x, positive(list(item.lower)), positive(list(item.upper)),
                             alpha=0.18, color="#0072B2", linewidth=0.0)
        axes[0].plot(item.x, positive(list(item.median)), color="#0072B2", marker=MARKERS[index],
                     label="STL/raw")
    axes[0].axhline(1.0, color="#555555", linestyle=":", label="equal variance")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Optimizer-relative state distance")
    axes[0].set_ylabel("Intrinsic variance ratio")

    _plot_series(
        axes[1], _core(by_experiment.get("H", [])), "objective_gap",
        xlabel="Iteration", ylabel="Exact Gaussian KL gap",
    )

    floor_data = _terminal_floors(_core(by_experiment.get("J", [])))
    for index, method in enumerate(sorted(floor_data, key=lambda name: METHOD_ORDER.get(name, 99))):
        values = sorted(floor_data[method])
        batch = np.asarray([value[0] for value in values])
        median = np.asarray([value[1] for value in values])
        lower = np.asarray([value[2] for value in values])
        upper = np.asarray([value[3] for value in values])
        color = COLORS.get(method, f"C{index}")
        axes[2].fill_between(batch, lower, upper, alpha=0.18, color=color, linewidth=0.0)
        axes[2].plot(batch, median, marker=MARKERS[index], color=color,
                     linestyle=LINESTYLES.get(method, "-"), label=method)
    if floor_data:
        anchor_method = min(floor_data, key=lambda name: METHOD_ORDER.get(name, 99))
        anchor = sorted(floor_data[anchor_method])[0]
        batch = np.asarray(sorted({value[0] for values in floor_data.values() for value in values}))
        axes[2].plot(batch, anchor[1] * anchor[0] / batch, color="#555555", linestyle=":", label=r"$B^{-1}$")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Minibatch size B")
    axes[2].set_ylabel("Tail median objective gap")

    decreasing = _core(by_experiment.get("K", []))
    _plot_series(
        axes[3], decreasing, "objective_gap", xlabel="Iteration N", ylabel=r"$\mathbb{E}[\Delta_N]$",
        log_x=True,
    )
    reference_series = _series(decreasing, "objective_gap")
    if reference_series:
        item = reference_series[0]
        mask = item.x > 0
        x = item.x[mask]
        y = positive(list(item.median[mask]))
        if len(x):
            axes[3].plot(x, y[0] * x[0] / x, color="#555555", linestyle=":", label=r"$N^{-1}$")
    _panel_titles(
        axes,
        (
            "(a) Raw versus STL variance",
            "(b) Gaussian pathwise cancellation",
            "(c) Minibatch residual floor",
            "(d) Decreasing-step diagnostic",
        ),
    )
    _deduplicated_legend(figure, axes)
    _save(
        figure,
        4,
        "Stochastic core evidence. Bands are 10--90 percentiles across paired seeds. Gaussian-matched FR--STL "
        "trajectories coincide pathwise, whereas S--FB--GVI retains sampling variation. Floors are tail medians; "
        "the decreasing-step panel overlays the predicted inverse-iteration slope.",
    )


def figure_5(by_experiment: dict[str, list[dict[str, str]]]) -> None:
    rows = _core(by_experiment.get("L", []))
    figure, axes_array = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    axes = axes_array.ravel()
    _plot_series(
        axes[0], rows, "objective_gap", x_key="oracle_pairs",
        xlabel="Gradient--Hessian oracle pairs", ylabel="Objective gap", laplace_line=True,
    )
    _plot_series(
        axes[1], rows, "objective_gap", x_key="wall_time_seconds",
        xlabel="Wall time (s)", ylabel="Objective gap", laplace_line=True,
    )
    _plot_series(
        axes[2], rows, "predictive_nll", x_key="wall_time_seconds",
        xlabel="Wall time (s)", ylabel="Test predictive NLL", log_y=False, laplace_line=True,
    )
    terminal: list[tuple[str, float, float, float]] = []
    for method in sorted({row.get("method", "") for row in rows}, key=lambda name: METHOD_ORDER.get(name, 99)):
        method_rows = [row for row in rows if row.get("method") == method]
        per_seed: list[float] = []
        for trajectory in group_trajectories(method_rows).values():
            ordered = sorted(trajectory, key=lambda row: _number(row, "iteration"))
            per_seed.append(_number(ordered[-1], "predictive_nll"))
        terminal.append(
            (method, float(np.nanmedian(per_seed)), float(np.nanpercentile(per_seed, 10)),
             float(np.nanpercentile(per_seed, 90)))
        )
    positions = np.arange(len(terminal))
    medians = np.asarray([value[1] for value in terminal])
    errors = np.asarray([[value[1] - value[2] for value in terminal], [value[3] - value[1] for value in terminal]])
    axes[3].errorbar(positions, medians, yerr=errors, fmt="none", ecolor="#555555", capsize=2)
    for index, (method, median, _lower, _upper) in enumerate(terminal):
        axes[3].scatter(index, median, marker=MARKERS[index % len(MARKERS)], s=32,
                        color=COLORS.get(method, f"C{index}"), zorder=3)
    axes[3].set_xticks(positions, [value[0] for value in terminal], rotation=35, ha="right")
    axes[3].set_ylabel("Final test predictive NLL")
    axes[3].text(0.02, 0.04, r"Core pilot: $\operatorname{cond}(\Sigma_x)=100$; full grid pending",
                 transform=axes[3].transAxes, fontsize=6.5, color="#555555")
    _panel_titles(
        axes,
        (
            "(a) Objective versus oracle pairs",
            "(b) Objective versus wall time",
            "(c) Predictive loss versus wall time",
            "(d) Final NLL at condition 100",
        ),
    )
    _deduplicated_legend(figure, axes[:3])
    _save(
        figure,
        5,
        "Theory-aligned Bayesian logistic-regression core pilot (d=10, n=100, proper Gaussian prior). Laplace "
        "is a noniterative horizontal quality reference. Stochastic bands use three paired seeds. Panel (d) is the "
        "completed feature-condition cell; the planned conditioning sweep remains in the full tier.",
    )


def main() -> int:
    configure_style()
    by_experiment = read_rows(RAW)
    write_main_processed(by_experiment)
    figure_1(by_experiment)
    figure_2(by_experiment)
    figure_3(by_experiment)
    figure_4(by_experiment)
    figure_5(by_experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
