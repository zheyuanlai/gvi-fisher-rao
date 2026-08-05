"""Per-experiment figures built from the full-tier grids.

Each experiment gets a figure designed around the mechanism it tests rather than
a generic trajectory plot, and every figure is accompanied by the exact
processed CSV that produced it plus a caption draft.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fr_gvi.plotting.style import (
    COLORS,
    FIGURES,
    LINESTYLES,
    MANIFESTS,
    MARKERS,
    METHOD_ORDER,
    REFERENCE_GREY,
    ROOT,
    TEXT_WIDTH,
    band,
    configure_style,
    display_label,
    load_experiment,
    method_style,
    panel_letter,
    positive,
    save_figure,
    terminal_rows,
    tidy_log_axes,
)

FAILURE_MARKER = "x"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def failed_runs(experiment: str) -> pd.DataFrame:
    """Every recorded algorithm failure, so instability is plotted, not hidden."""

    records = []
    for path in sorted((MANIFESTS / "full").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("config", {}).get("experiment") != experiment:
            continue
        if manifest.get("status") != "failed":
            continue
        specification = manifest.get("method_specification", {})
        records.append(
            {
                "job_id": manifest["config"]["id"],
                "method": specification.get("name", ""),
                "normalized_step_size": specification.get("normalized_step_size", np.nan),
                "step_size": specification.get("step_size", np.nan),
                "batch_size": specification.get("batch_size", np.nan),
                "failure_reason": manifest.get("failure_reason", ""),
                **{f"grid_{k}": v for k, v in manifest["config"].get("grid", {}).items()},
            }
        )
    return pd.DataFrame(records)


def ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"].unique())
    return [method for method in METHOD_ORDER if method in present]


def plot_median_band(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    label: str,
    style: dict,
    log_y: bool = True,
    markevery: int | None = None,
) -> None:
    grouped = frame.groupby(x)[y]
    xs = np.asarray(sorted(frame[x].unique()), dtype=np.float64)
    median = grouped.median().to_numpy()
    lower = grouped.quantile(0.10).to_numpy()
    upper = grouped.quantile(0.90).to_numpy()
    if log_y:
        median, lower, upper = positive(median), positive(lower), positive(upper)
    if grouped.count().max() > 1:
        band(axis, xs, lower, upper, style["color"])
    axis.plot(
        xs,
        median,
        label=label,
        markevery=markevery or max(1, len(xs) // 8),
        **style,
    )
    if log_y:
        axis.set_yscale("log")


def legend_below(figure: plt.Figure, axes, columns: int = 4, tidy: bool = True) -> None:
    if tidy:
        tidy_log_axes(*np.atleast_1d(axes).ravel())
    handles: list = []
    labels: list[str] = []
    for axis in np.atleast_1d(axes).ravel():
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=min(columns, len(labels)),
            frameon=False,
        )


# --------------------------------------------------------------------------
# Experiment A: covariance burn-in
# --------------------------------------------------------------------------


def figure_a(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    burn_in = []
    for (job, method, seed), trajectory in frame.groupby(["job_id", "method", "seed"]):
        trajectory = trajectory.sort_values("iteration")
        beta_star = float(trajectory["beta_star"].iloc[0])
        threshold = 1.0 / (2.0 * beta_star)
        entered = trajectory[trajectory["relative_covariance_min_eigenvalue"] >= threshold]
        if entered.empty:
            continue
        step = float(trajectory["step_size"].iloc[-1])
        lambda_0 = float(trajectory["lambda_0_star"].iloc[0])
        burn_in.append(
            {
                "job_id": job,
                "method": method,
                "condition": float(trajectory["grid_condition"].iloc[0]),
                "lambda0": float(trajectory["grid_lambda0"].iloc[0]),
                "lambda_0_star": lambda_0,
                "beta_star": beta_star,
                "step_size": step,
                "burn_in_iteration": int(entered["iteration"].iloc[0]),
                "burn_in_time": int(entered["iteration"].iloc[0]) * step,
                "predicted_time": float(np.log(1.0 / (beta_star * lambda_0))),
            }
        )
    table = pd.DataFrame(burn_in)

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    # (a) measured entry time against the theoretical logarithmic envelope
    for method in ordered_methods(table.rename(columns={"method": "method"})):
        subset = table[table["method"] == method].sort_values("predicted_time")
        style = method_style(method)
        axes[0].plot(
            subset["predicted_time"],
            subset["burn_in_time"],
            linestyle="none",
            label=method,
            **{k: v for k, v in style.items() if k != "linestyle"},
        )
    limits = [0.0, float(table["predicted_time"].max()) * 1.05]
    axes[0].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label=r"$\log[1/(\beta_\star\lambda_{0,\star})]$")
    axes[0].set_xlabel(r"$\log[1/(\beta_\star\lambda_{0,\star})]$")
    axes[0].set_ylabel(r"measured $N_{\mathrm{cov}}\,\Delta t$")
    panel_letter(axes[0], "a", "Logarithmic burn-in")

    # (b) independence of the original-coordinate condition number
    table["ratio"] = table["burn_in_time"] / table["predicted_time"]
    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        summary = subset.groupby("condition")["ratio"]
        axes[1].errorbar(
            summary.median().index,
            summary.median().to_numpy(),
            yerr=[
                summary.median().to_numpy() - summary.min().to_numpy(),
                summary.max().to_numpy() - summary.median().to_numpy(),
            ],
            label=method,
            capsize=2.5,
            **method_style(method),
        )
    axes[1].axhline(1.0, color=REFERENCE_GREY, linestyle=":", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 2.0)
    axes[1].set_xlabel(r"original-coordinate condition number $\kappa$")
    axes[1].set_ylabel("measured / predicted")
    panel_letter(axes[1], "b", r"Independent of $\kappa$")

    # (c) three-stage trajectory for the hardest start
    hardest = frame[frame["grid_lambda0"] == frame["grid_lambda0"].min()]
    hardest = hardest[hardest["grid_condition"] == hardest["grid_condition"].max()]
    hardest = hardest[hardest["method"] == "FR--R"].sort_values("iteration")
    axes[2].plot(
        hardest["iteration"], positive(hardest["relative_covariance_min_eigenvalue"]),
        color="#0072B2", linestyle="-",
        label=r"$\lambda_{\min}(\widetilde C_n)$",
    )
    axes[2].plot(
        hardest["iteration"], positive(hardest["objective_gap"]),
        color="#E69F00", linestyle="--", label=r"$\Delta(a_n)$",
    )
    axes[2].plot(
        hardest["iteration"], positive(hardest["fisher_rao_residual_squared"]),
        color="#CC79A7", linestyle="-.", label=r"$\|\mathrm{grad}\,\mathcal{E}\|_{a_n}^2$",
    )
    axes[2].axhline(0.5, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                    label=r"band $1/(2\beta_\star)$")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("iteration $n$")
    axes[2].set_ylabel("diagnostic value")
    panel_letter(axes[2], "c", r"FR--R, $\lambda_0=10^{-12}$, $\kappa=10^3$")

    legend_below(figure, axes, columns=4)
    caption = (
        "Fisher--Rao covariance burn-in on Gaussian targets with $d=20$, "
        "$\\kappa\\in\\{10,10^2,10^3\\}$ and $\\lambda_0\\in\\{10^{-2},\\dots,10^{-12}\\}$, "
        "step $\\Delta t=0.1$. (a) The measured entry time of the optimizer-whitened "
        "covariance band $\\lambda_{\\min}(\\widetilde C_n)\\ge 1/(2\\beta_\\star)$ follows the "
        "predicted $\\log[1/(\\beta_\\star\\lambda_{0,\\star})]$ envelope of "
        "Theorem 2.9. (b) The entry time is independent of the original-coordinate "
        "condition number $\\kappa$, as affine invariance requires. (c) The hardest start "
        "shows the covariance reaching the band before the energy gap contracts."
    )
    return figure, caption, table


# --------------------------------------------------------------------------
# Experiment B: affine equivariance
# --------------------------------------------------------------------------


def figure_b(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    frame = frame.copy()
    frame["affine_condition"] = frame["grid_affine_condition"]
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)

    hardest = frame[frame["affine_condition"] == frame["affine_condition"].max()]
    for method in ordered_methods(hardest):
        trajectory = hardest[hardest["method"] == method].sort_values("iteration")
        axes[0].plot(
            trajectory["iteration"],
            positive(trajectory["equivariance_error_covariance"]),
            label=method,
            **method_style(method),
            markevery=max(1, len(trajectory) // 8),
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel("relative covariance equivariance error")
    panel_letter(axes[0], "a", r"$K=10^8$")

    summary = (
        frame.groupby(["method", "affine_condition"])[
            ["equivariance_error_mean", "equivariance_error_covariance"]
        ]
        .max()
        .reset_index()
    )
    for method in ordered_methods(summary):
        subset = summary[summary["method"] == method].sort_values("affine_condition")
        axes[1].plot(
            subset["affine_condition"],
            positive(subset["equivariance_error_covariance"]),
            label=method,
            **method_style(method),
        )
    axes[1].axhline(
        1e-12, color=REFERENCE_GREY, linestyle=":", linewidth=1.0, label="roundoff scale"
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"conditioning $K$ of the coordinate change")
    axes[1].set_ylabel("max relative equivariance error")
    panel_letter(axes[1], "b", "Across conditioning")

    legend_below(figure, axes)
    caption = (
        "Affine equivariance (Proposition 2.3). A Gaussian target and its initialization "
        "are transformed by an invertible affine map of conditioning $K$, and the "
        "transformed iterates are compared with the iterates of the untransformed run. "
        "Both Fisher--Rao schemes agree with the transported reference to within roundoff "
        "across eight decades of $K$. Their residual grows in proportion to the "
        "conditioning of the change of variables, from $10^{-15}$ at $K=1$ to $10^{-9}$ at "
        "$K=10^8$, which is floating-point amplification through an ill-conditioned map "
        "rather than a loss of equivariance. FB--GVI, whose Bures--Wasserstein geometry is "
        "not affine invariant, departs by an $O(1)$ amount as soon as the map is "
        "non-orthogonal. This is an iterate-level identity test, not a cross-geometry "
        "performance comparison."
    )
    return figure, caption, summary


# --------------------------------------------------------------------------
# Experiments C, D, L: stepsize-swept comparisons
# --------------------------------------------------------------------------


def stability_table(frame: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Largest stable normalized step and best gap at a fixed oracle budget."""

    initial = frame[frame["iteration"] == 0].groupby(["job_id", "method"])["objective_gap"].median()
    terminal = terminal_rows(frame)
    records = []
    for (job, method), subset in terminal.groupby(["job_id", "method"]):
        start = float(initial.get((job, method), np.inf))
        # "Stable" means the run neither failed nor ended above where it started;
        # a step that merely avoids overflow while making no progress is useless.
        progressed = subset[subset["objective_gap"] < start]
        stable = float(progressed["normalized_step_size"].max()) if not progressed.empty else np.nan
        failed = failures[(failures["job_id"] == job) & (failures["method"] == method)]
        smallest_failure = (
            float(failed["normalized_step_size"].min()) if not failed.empty else np.nan
        )
        best = subset.loc[subset["objective_gap"].idxmin()]
        records.append(
            {
                "job_id": job,
                "method": method,
                "largest_stable_normalized_step": stable,
                "smallest_failing_normalized_step": smallest_failure,
                "stability_censored": bool(
                    np.isnan(smallest_failure)
                    and stable == float(subset["normalized_step_size"].max())
                ),
                "best_terminal_gap": float(best["objective_gap"]),
                "best_normalized_step": float(best["normalized_step_size"]),
                "oracle_pairs_at_best": float(best["oracle_pairs"]),
                "kappa_star": float(subset["kappa_star"].iloc[0]),
                "failures": int(len(failed)),
            }
        )
    return pd.DataFrame(records)


def figure_c(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    failures = failed_runs("C")
    competitive = frame[~frame["variant"].str.contains("rescue", na=False)]
    rescue = frame[frame["variant"].str.contains("rescue", na=False)]
    table = stability_table(competitive, failures)

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    terminal = terminal_rows(competitive)
    for method in ordered_methods(competitive):
        best_step = table.loc[table["method"] == method, "best_normalized_step"].iloc[0]
        subset = competitive[
            (competitive["method"] == method)
            & np.isclose(competitive["normalized_step_size"], best_step)
        ]
        label = rf"{method} ($\times{best_step:g}$)"
        plot_median_band(
            axes[0], subset, "iteration", "exact_gaussian_kl",
            label=label, style=method_style(method),
        )
        clock = subset.groupby("iteration")[["wall_time_seconds", "exact_gaussian_kl"]].median()
        axes[1].plot(
            clock["wall_time_seconds"], positive(clock["exact_gaussian_kl"]),
            label=label, **method_style(method),
            markevery=max(1, len(clock) // 8),
        )
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel(r"exact KL$(\rho_{a_n}\,\|\,\rho_{\mathrm{post}})$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("wall-clock time (s)")
    axes[1].set_ylabel("exact KL gap")
    panel_letter(axes[0], "a", "Best step per method")
    panel_letter(axes[1], "b", "Per wall-clock second")

    for method in ordered_methods(terminal):
        subset = terminal[terminal["method"] == method]
        grouped = subset.groupby("normalized_step_size")["exact_gaussian_kl"].median()
        axes[2].plot(grouped.index, positive(grouped.to_numpy()), **method_style(method))
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    bottom, top = axes[2].get_ylim()
    for method in ordered_methods(terminal):
        failed = failures[failures["method"] == method]
        if failed.empty:
            continue
        steps = np.sort(failed["normalized_step_size"].unique())
        axes[2].plot(
            steps, np.full(steps.size, top * 0.6), linestyle="none",
            marker=FAILURE_MARKER, markersize=5, color=COLORS.get(method),
            label=f"{method} diverged",
        )
    axes[2].set_xlabel(r"step $\Delta t$ / certified step")
    axes[2].set_ylabel("terminal KL gap")
    panel_letter(axes[2], "c", "Stepsize sweep")

    legend_below(figure, axes, columns=4)
    rescue_terminal = terminal_rows(rescue)
    caption = (
        "Anisotropic Gaussian benchmark of Diao et al. ($d=10$, "
        "$\\Sigma^{-1}=U\\mathrm{diag}(10^{-9},\\dots,1)U^\\top$, $q_0=\\mathcal N(0,I)$), "
        "10 random rotations, 500 iterations. Each method is swept over multiples of its "
        "own certified step, so no geometry is forced onto another's stepsize scale. "
        "The optimizer-whitened condition number is $\\kappa_\\star=1$ here while the "
        "original-coordinate $\\kappa=10^9$, which is exactly the regime the affine-invariant "
        "rate predicts the Fisher--Rao schemes to be insensitive to. Quadratic-rescued "
        "Fisher--Rao runs recover the target in one oracle query and are reported "
        f"separately (median gap {rescue_terminal['exact_gaussian_kl'].median():.2e} after "
        f"{rescue_terminal['oracle_pairs'].median():.0f} query); they are excluded from the "
        "competitive curves."
    )
    return figure, caption, table


def figure_d(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    failures = failed_runs("D")
    table = stability_table(frame, failures)
    table = table.merge(
        frame.groupby("job_id")[["grid_dimension", "grid_condition", "grid_rho"]].first(),
        on="job_id",
    )

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    representative = frame[frame["job_id"] == "D_logcosh_d10_k100_rho1"]
    for method in ordered_methods(representative):
        best_step = table.loc[
            (table["job_id"] == "D_logcosh_d10_k100_rho1") & (table["method"] == method),
            "best_normalized_step",
        ]
        if best_step.empty:
            continue
        subset = representative[
            (representative["method"] == method)
            & np.isclose(representative["normalized_step_size"], float(best_step.iloc[0]))
        ]
        plot_median_band(
            axes[0], subset, "oracle_pairs", "objective_gap",
            label=method, style=method_style(method),
        )
    axes[0].set_xlabel("gradient--Hessian oracle pairs")
    axes[0].set_ylabel(r"objective gap $\Delta(a_n)$")
    panel_letter(axes[0], "a", r"$d=10$, $\kappa_{\rm base}=10^2$, $\rho=1$")

    sizes = {2: 12.0, 10: 22.0, 50: 38.0}
    for method in ordered_methods(table):
        subset = table[table["method"] == method].sort_values("kappa_star")
        axes[1].scatter(
            subset["kappa_star"], positive(subset["best_terminal_gap"]),
            s=[sizes.get(int(d), 20.0) for d in subset["grid_dimension"]],
            facecolors="none", edgecolors=COLORS.get(method), linewidths=0.9,
            marker=MARKERS.get(method), label=method,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\kappa_\star$")
    axes[1].set_ylabel("best gap at fixed budget")
    panel_letter(axes[1], "b", r"27 cells, size $\propto d$")

    for method in ordered_methods(table):
        subset = table[table["method"] == method].sort_values("kappa_star")
        axes[2].scatter(
            subset["kappa_star"], subset["largest_stable_normalized_step"],
            s=22.0, facecolors="none", edgecolors=COLORS.get(method), linewidths=0.9,
            marker=MARKERS.get(method), label=method,
        )
        censored = subset[subset["stability_censored"]]
        if not censored.empty:
            axes[2].scatter(
                censored["kappa_star"], censored["largest_stable_normalized_step"],
                s=22.0, color=COLORS.get(method), marker="^", alpha=0.5,
                label=f"{method} censored at grid top",
            )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\kappa_\star$")
    axes[2].set_ylabel("largest stable step / certified")
    panel_letter(axes[2], "c", "Stability margin")

    legend_below(figure, axes)
    caption = (
        "Strongly log-concave shifted log-cosh comparison over the full grid "
        "$d\\in\\{2,10,50\\}$, $\\kappa_{\\rm base}\\in\\{1,10,10^2\\}$, "
        "$\\rho\\in\\{0.1,1,5\\}$, each with five random affine copies of conditioning $10$ "
        "and a nine-point sweep around each method's certified step. "
        "(a) A representative cell at each method's best step. (b) Best terminal gap at a "
        "fixed oracle budget against the affine-invariant condition number $\\kappa_\\star$. "
        "(c) Largest stable step as a multiple of the certified step; certified steps are "
        "conservative for all three methods, and every divergent run is recorded as a "
        "failure rather than stabilized."
    )
    return figure, caption, table


def figure_l(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    failures = failed_runs("L")
    iterative = frame[frame["method"] != "Laplace"]
    laplace = frame[frame["method"] == "Laplace"]
    table = stability_table(iterative, failures)
    table = table.merge(
        frame.groupby("job_id")[
            ["grid_dimension", "grid_prior_precision", "grid_feature_condition"]
        ].first(),
        on="job_id",
    )

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    representative = frame[frame["job_id"] == "L_logistic_d50_lam1_fc1e2"]
    if representative.empty:
        representative = frame[frame["job_id"] == frame["job_id"].iloc[0]]
    job = representative["job_id"].iloc[0]

    for method in ordered_methods(representative[representative["method"] != "Laplace"]):
        best = table[(table["job_id"] == job) & (table["method"] == method)]
        if best.empty:
            continue
        subset = representative[
            (representative["method"] == method)
            & np.isclose(representative["normalized_step_size"],
                         float(best["best_normalized_step"].iloc[0]))
        ]
        plot_median_band(
            axes[0], subset, "oracle_pairs", "objective_gap",
            label=method, style=method_style(method),
        )
        plot_median_band(
            axes[1], subset, "oracle_pairs", "predictive_nll",
            label=method, style=method_style(method), log_y=False,
        )
    laplace_cell = laplace[laplace["job_id"] == job]
    if not laplace_cell.empty:
        axes[0].axhline(
            float(laplace_cell["objective_gap"].iloc[-1]),
            color=COLORS["Laplace"], linestyle=":", linewidth=1.1, label="Laplace",
        )
        axes[1].axhline(
            float(laplace_cell["predictive_nll"].iloc[-1]),
            color=COLORS["Laplace"], linestyle=":", linewidth=1.1, label="Laplace",
        )
    axes[0].set_xlabel("gradient--Hessian oracle pairs")
    axes[0].set_ylabel("Gaussian-VI objective gap")
    panel_letter(axes[0], "a", job.replace("_", " "))
    axes[1].set_xlabel("gradient--Hessian oracle pairs")
    axes[1].set_ylabel("held-out predictive NLL")
    # The early transient is orders of magnitude above the converged level, so the
    # axis is clipped to the region where the methods can actually be told apart.
    converged = representative[
        (representative["iteration"] >= 0.5 * representative["iteration"].max())
    ]["predictive_nll"].dropna()
    if len(converged):
        low, high = float(converged.min()), float(converged.max())
        margin = max(0.02 * abs(low), 1.5 * (high - low), 1e-3)
        axes[1].set_ylim(low - margin, high + margin)
    panel_letter(axes[1], "b", "Predictive quality")

    for method in ordered_methods(table):
        subset = table[table["method"] == method].sort_values("kappa_star")
        axes[2].plot(
            subset["kappa_star"], positive(subset["best_terminal_gap"]),
            linestyle="none", label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\kappa_\star$")
    axes[2].set_ylabel("best terminal gap")
    panel_letter(axes[2], "c", "All logistic cells")

    legend_below(figure, axes)
    caption = (
        "Bayesian logistic regression with a proper Gaussian prior, "
        "$d\\in\\{10,50,100\\}$, $n=10d$, $\\lambda\\in\\{0.1,1\\}$ and feature-covariance "
        "conditioning in $\\{1,10^2,10^4\\}$, using exact full-data derivatives. "
        "Each cell uses one fixed quadrature design shared by the updates, the objective "
        "evaluation and the reference, so deterministic gaps are exact for the discretized "
        "problem; the quadrature resolution floor of each cell is recorded in its reference "
        "manifest. Laplace appears only as a non-iterative approximation-quality baseline."
    )
    return figure, caption, table


# --------------------------------------------------------------------------
# Experiment F: exact Gaussian local region
# --------------------------------------------------------------------------


def figure_f(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    frame = frame.copy()
    frame["distance_squared"] = frame["mean_error"] ** 2 + 0.5 * frame["covariance_error"] ** 2

    records = []
    for (job, method), subset in frame.groupby(["job_id", "method"]):
        subset = subset.sort_values("iteration").reset_index(drop=True)
        step = float(subset["step_size"].iloc[-1])
        values = subset["distance_squared"].to_numpy()
        rates = subset["gaussian_core_rate"].to_numpy()
        relative_minimum = subset["relative_covariance_min_eigenvalue"].to_numpy()
        usable = np.isfinite(values) & (values > 1e-20)
        # Instantaneous measured rate from one step to the next, compared with the
        # exact Gaussian-core rate q_G at the same state.
        for index in range(len(values) - 1):
            if not (usable[index] and usable[index + 1]):
                continue
            measured = -float(np.log(values[index + 1] / values[index])) / step
            records.append(
                {
                    "job_id": job,
                    "method": method,
                    "dimension": int(subset["grid_dimension"].iloc[0]),
                    "rate_parameter": float(subset["grid_rate"].iloc[0]),
                    "initial_eigenvalue": float(subset["grid_initial_eigenvalue"].iloc[0]),
                    "iteration": int(subset["iteration"].iloc[index]),
                    "step_size": step,
                    "measured_rate": measured,
                    "core_rate": float(rates[index]),
                    "two_lambda_min": 2.0 * float(relative_minimum[index]),
                    "initial_energy_gap": float(subset["objective_gap"].iloc[0]),
                }
            )
    table = pd.DataFrame(records)

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    representative = frame[frame["grid_dimension"] == 10]
    for rate, subset in representative.groupby("grid_rate"):
        first = subset[subset["method"] == "FR--R"].sort_values("iteration")
        step = float(first["step_size"].iloc[-1])
        times = first["iteration"].to_numpy() * step
        axes[0].plot(
            times, positive(first["distance_squared"]),
            color=COLORS["FR--R"], linestyle="-", linewidth=1.1,
            label="FR--R" if rate == representative["grid_rate"].min() else None,
        )
        second = subset[subset["method"] == "FR--KL"].sort_values("iteration")
        axes[0].plot(
            second["iteration"].to_numpy() * step, positive(second["distance_squared"]),
            color=COLORS["FR--KL"], linestyle="--", linewidth=1.1,
            label="FR--KL" if rate == representative["grid_rate"].min() else None,
        )
        start = float(first["distance_squared"].iloc[0])
        axes[0].plot(
            times, start * np.exp(-rate * times),
            color=REFERENCE_GREY, linestyle=":", linewidth=0.9,
            label=r"bound $e^{-2\ell_\delta t}$" if rate == representative["grid_rate"].min() else None,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"time $t=n\,\Delta t$")
    axes[0].set_ylabel(r"$\|a_n-a_\star\|_\star^2$")
    panel_letter(axes[0], "a", r"$d=10$, four $\ell_\delta$")

    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        subset = subset[np.isfinite(subset["core_rate"]) & np.isfinite(subset["measured_rate"])]
        axes[1].plot(
            subset["core_rate"], subset["measured_rate"], linestyle="none",
            markersize=2.0, alpha=0.5, label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    finite = table["core_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    limits = [float(finite.min()) * 0.95, float(finite.max()) * 1.05]
    axes[1].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label="identity")
    axes[1].set_xlabel(r"exact core rate $q_{\mathrm{G}}(a_n)$")
    axes[1].set_ylabel("measured instantaneous rate")
    panel_letter(axes[1], "b", "Lemma 2.24 identity")

    initial = table[table["iteration"] == 0].drop_duplicates(
        subset=["job_id", "method"]
    )
    initial = initial.assign(
        predicted_threshold=0.5
        * (
            initial["initial_eigenvalue"]
            - 1.0
            - np.log(initial["initial_eigenvalue"])
        )
    )
    for method in ordered_methods(initial):
        subset = initial[initial["method"] == method]
        axes[2].plot(
            subset["predicted_threshold"], subset["initial_energy_gap"],
            linestyle="none", label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    limits = [0.0, float(initial["predicted_threshold"].max()) * 1.1]
    axes[2].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label="identity")
    axes[2].set_xlabel(r"$\Delta_{\mathrm{G}}^{\sharp}(\rho)$")
    axes[2].set_ylabel(r"measured $\Delta(a_0)$")
    panel_letter(axes[2], "c", "Sharp threshold")

    legend_below(figure, axes)
    caption = (
        "Exact Gaussian local region. The target is $\\mathcal N(0,I)$ and the "
        "initialization is $C_0=\\mathrm{diag}(c,1,\\dots,1)$ with $c=\\rho/2$ for "
        "$\\rho\\in\\{0.25,0.5,1,1.5\\}$ and $d\\in\\{2,10,100\\}$. (a) Trajectories stay "
        "below the uniform bound $e^{-2\\ell_\\delta t}$ of Corollary 2.26; the bound is "
        "not tight late in the trajectory because $\\ell_\\delta$ grows toward $1$ as the "
        "covariance approaches $C_\\star$. (b) The measured instantaneous decay rate of "
        "$\\|a_n-a_\\star\\|_\\star^2$ coincides with the exact Gaussian-core rate "
        "$q_{\\mathrm G}(a_n)$ of Lemma 2.24 along every trajectory and dimension. "
        "(c) The energy of the extremal initialization matches the sharp threshold "
        "$\\Delta_{\\mathrm G}^{\\sharp}(\\rho)=\\frac12(\\rho/2-1-\\log(\\rho/2))$."
    )
    return figure, caption, table


# --------------------------------------------------------------------------
# Experiment G: near-Gaussian local spectral rate
# --------------------------------------------------------------------------


def figure_g(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    frame = frame.copy()
    frame["distance_squared"] = frame["mean_error"] ** 2 + 0.5 * frame["covariance_error"] ** 2
    records = []
    for (job, method), subset in frame.groupby(["job_id", "method"]):
        subset = subset.sort_values("iteration")
        step = float(subset["step_size"].iloc[-1])
        times = subset["iteration"].to_numpy() * step
        values = subset["distance_squared"].to_numpy()
        # The whitened error saturates once it reaches the relative precision of
        # C_star^{-1/2} C C_star^{-1/2}, which puts the squared distance on a
        # roundoff plateau near 1e-26.  Everything below 1e-22 is discarded.
        usable = np.isfinite(values) & (values > 1e-22)
        tail = usable & (times >= 0.6 * times[usable].max()) if usable.any() else usable
        fitted = np.nan
        if tail.sum() >= 4:
            fitted = float(-np.polyfit(times[tail], np.log(values[tail]), 1)[0])
        records.append(
            {
                "job_id": job,
                "method": method,
                "dimension": int(subset["grid_dimension"].iloc[0]),
                "rho": float(subset["grid_rho"].iloc[0]),
                "step_size": step,
                "gamma_star": float(subset["local_gamma"].iloc[0]),
                "gamma_kl": float(subset["local_kl_gamma"].iloc[0]),
                "lambda_star": float(subset["local_lambda"].iloc[0]),
                "kappa_star": float(subset["kappa_star"].iloc[0]),
                "fitted_rate": fitted,
                "continuous_rate": 2.0 * float(subset["local_gamma"].iloc[0]),
                "discrete_rate": float(subset["local_discrete_rate"].iloc[0]),
                "fast_rate": float(
                    -2.0 * np.log1p(-step * float(subset["local_lambda"].iloc[0])) / step
                ),
            }
        )
    table = pd.DataFrame(records)

    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    # (a) The per-step contraction rate along a representative trajectory,
    # converging onto the rate predicted by the spectral gap.  A single window
    # fit would mix in the faster modes, which decay only a little quicker.
    representative = frame[(frame["grid_dimension"] == 3) & (frame["grid_rho"] == 0.01)]
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method].sort_values("iteration")
        step = float(subset["step_size"].iloc[-1])
        times = subset["iteration"].to_numpy() * step
        distance = (
            subset["mean_error"] ** 2 + 0.5 * subset["covariance_error"] ** 2
        ).to_numpy()
        usable = np.flatnonzero(np.isfinite(distance) & (distance > 1e-22))
        rates = -np.diff(np.log(distance[usable])) / np.diff(times[usable])
        style = method_style(method)
        axes[0].plot(times[usable][:-1], rates, label=method, **style,
                     markevery=max(1, len(rates) // 8))
        predicted = float(subset["local_discrete_rate"].iloc[0])
        axes[0].axhline(predicted, color=style["color"], linestyle=":", linewidth=0.9)
    axes[0].set_xlabel(r"time $t=n\,\Delta t$")
    axes[0].set_ylabel("per-step contraction rate")
    panel_letter(axes[0], "a", r"$d=3$, $\rho=10^{-2}$")

    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        axes[1].vlines(
            subset["discrete_rate"], subset["discrete_rate"], subset["fast_rate"],
            color=COLORS.get(method), alpha=0.25, linewidth=2.5,
        )
        axes[1].plot(
            subset["discrete_rate"], subset["fitted_rate"], linestyle="none", label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    finite = table["discrete_rate"].dropna()
    limits = [float(finite.min()) * 0.95, float(finite.max()) * 1.05]
    axes[1].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label="identity")
    axes[1].set_xlabel("predicted slowest one-step rate")
    axes[1].set_ylabel("measured rate")
    panel_letter(axes[1], "b", "Inside the spectral bracket")

    spectra = table.groupby(["dimension", "rho"])[["gamma_star", "lambda_star"]].first().reset_index()
    dimension_markers = {2: "o", 3: "s", 5: "^"}
    for dimension, subset in spectra.groupby("dimension"):
        subset = subset.sort_values("rho")
        axes[2].plot(subset["rho"], subset["gamma_star"], color="#0072B2",
                     marker=dimension_markers.get(int(dimension), "o"),
                     label=rf"$\gamma_\star$" if dimension == 2 else None)
        axes[2].plot(subset["rho"], subset["lambda_star"], color="#D55E00", linestyle="--",
                     marker=dimension_markers.get(int(dimension), "o"),
                     label=rf"$\Lambda_\star$" if dimension == 2 else None)
    for dimension, marker in dimension_markers.items():
        axes[2].plot([], [], color=REFERENCE_GREY, linestyle="none", marker=marker,
                     label=f"$d={dimension}$")
    axes[2].axhline(1.0, color=REFERENCE_GREY, linestyle=":", linewidth=0.9)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"non-Gaussianity $\rho$")
    axes[2].set_ylabel(r"spectrum of $\mathcal{L}_\star$")
    panel_letter(axes[2], "c", "Spectral bracket")

    legend_below(figure, axes, columns=4)
    caption = (
        "Local spectral rate on the shifted log-cosh family, $d\\in\\{2,3,5\\}$ and "
        "$\\rho\\in[10^{-3},2]$. The linearized Fisher--Rao generator $\\mathcal L_\\star$ of "
        "Proposition 3.5 is assembled numerically from its score operators "
        "$\\mathcal T$, $\\mathcal T^\\ast$ and $\\mathcal S$ in optimizer-whitened "
        "coordinates, and the KL/Bregman gap $\\gamma_{\\mathrm{KL},\\Delta t}$ of "
        "Theorem 3.7(iii) from the same operator under the weighted inner product. "
        "(a) The per-step contraction rate along a representative trajectory settles onto "
        "the exact one-step prediction $-2\\log(1-\\Delta t\\,\\gamma)/\\Delta t$ "
        "(dotted). (b) Measured rates against that prediction, with the shaded segments "
        "showing the full spectral bracket from the slowest mode $\\gamma$ to the fastest "
        "$\\Lambda_\\star$. Every measurement lies inside its bracket. Where it sits above "
        "the slow end -- FR--KL at small $\\rho$, whose two extreme modes differ by only "
        "$5\\%$ -- the faster modes have not yet died out at the largest horizon float64 "
        "allows: the whitened error saturates near $10^{-26}$, capping the usable time at "
        "$t\\approx30$. (c) $\\gamma_\\star$ and $\\Lambda_\\star$ against the "
        "non-Gaussianity, both approaching $1$ as $\\rho\\to0$ where "
        "$\\mathcal L_\\star=\\mathrm{Id}$."
    )
    return figure, caption, table


# --------------------------------------------------------------------------
# Experiment H: Gaussian STL cancellation
# --------------------------------------------------------------------------


def figure_h(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    spread = (
        frame.groupby(["job_id", "method", "iteration"])["objective"]
        .agg(lambda values: float(values.max() - values.min()))
        .reset_index(name="across_seed_spread")
    )
    spread = spread.merge(
        frame.groupby("job_id")["grid_dimension"].first(), on="job_id"
    )
    # A spread of exactly zero -- every seed producing bit-identical iterates --
    # cannot be drawn on a logarithmic axis.  Such points are drawn on a marked
    # floor line and counted in the caption rather than dropped.
    floor = 1.0e-17
    spread["exact_zero"] = spread["across_seed_spread"] == 0.0
    spread["plotted"] = np.maximum(spread["across_seed_spread"], floor)

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)

    representative = spread[spread["grid_dimension"] == 10]
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method].sort_values("iteration")
        axes[0].plot(
            subset["iteration"], subset["plotted"], label=method, **method_style(method),
            markevery=max(1, len(subset) // 8),
        )
    axes[0].axhline(floor, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                    label="bit-identical floor")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-18, None)
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel("across-seed objective spread")
    panel_letter(axes[0], "a", r"$d=10$, $B=1$")

    terminal = spread.sort_values("iteration").groupby(["job_id", "method"], as_index=False).last()
    for method in ordered_methods(terminal):
        subset = terminal[terminal["method"] == method].sort_values("grid_dimension")
        axes[1].plot(subset["grid_dimension"], subset["plotted"], label=method,
                     **method_style(method))
    axes[1].axhline(floor, color=REFERENCE_GREY, linestyle=":", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(3e-18, None)
    axes[1].set_xlabel("dimension $d$")
    axes[1].set_ylabel("terminal across-seed spread")
    panel_letter(axes[1], "b", "Across dimension")

    legend_below(figure, axes)
    zero_fraction = {
        method: float(subset["exact_zero"].mean())
        for method, subset in spread.groupby("method")
    }
    fisher_rao_zero = min(
        zero_fraction.get("FR--R--STL", 0.0), zero_fraction.get("FR--KL--STL", 0.0)
    )
    caption = (
        "Sticking-the-landing cancellation on a Gaussian target initialized with the "
        "matched covariance $C_0=C_\\star$ and a mismatched mean, $B=1$, 30 paired seeds, "
        "$d\\in\\{2,10,50\\}$. The Fisher--Rao STL mean noise is proportional to "
        "$C_n-C_\\star$ and so vanishes pathwise once the covariance is matched "
        "(Corollary 4.20): the two Fisher--Rao schemes produce bit-identical trajectories "
        f"across all 30 seeds at {100 * fisher_rao_zero:.0f}\\% of recorded iterations, and "
        "never differ by more than floating-point roundoff. S--FB--GVI retains its native "
        "estimator noise at $O(1)$. This compares complete algorithms together with their "
        "native estimators, not geometry alone."
    )
    return figure, caption, spread


# --------------------------------------------------------------------------
# Experiment I: raw versus STL estimator variance
# --------------------------------------------------------------------------


def figure_i(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)
    palette = {"gaussian": "#0072B2", "logcosh": "#D55E00"}
    dashes = {1: "-", 4: "--", 16: ":"}
    markers = {1: "o", 4: "s", 16: "^"}

    summaries = []
    for (family, batch), subset in frame.groupby(["grid_family", "grid_batch_size"]):
        summary = (
            subset.groupby("interpolation_level")[
                ["optimizer_relative_distance", "stl_raw_variance_ratio",
                 "measured_tangent_variance", "lemma_variance_bound",
                 "psi_hessian_fluctuation", "fisher_rao_gradient_norm_squared"]
            ]
            .median()
            .reset_index()
            .sort_values("optimizer_relative_distance")
        )
        summary["family"] = family
        summary["batch_size"] = int(batch)
        summaries.append(summary)
        style = {
            "color": palette.get(family, "C0"),
            "linestyle": dashes.get(int(batch), "-"),
            "marker": markers.get(int(batch), "o"),
        }
        label = f"{family}, $B={int(batch)}$"
        # The exact-zero point at the optimizer of the Gaussian target would
        # collapse a log axis; it is reported in the caption instead of plotted.
        finite = summary[summary["optimizer_relative_distance"] > 0.0]
        axes[0].plot(finite["optimizer_relative_distance"],
                     positive(finite["stl_raw_variance_ratio"]), label=label, **style)
        axes[1].plot(finite["optimizer_relative_distance"],
                     positive(finite["measured_tangent_variance"]), label=label, **style)
        axes[1].plot(finite["optimizer_relative_distance"],
                     positive(finite["lemma_variance_bound"]),
                     color=style["color"], linestyle=style["linestyle"], alpha=0.45,
                     marker="", linewidth=2.2)
        axes[2].plot(
            finite["optimizer_relative_distance"],
            positive(finite["measured_tangent_variance"] / finite["lemma_variance_bound"]),
            label=label, **style,
        )
    table = pd.concat(summaries, ignore_index=True)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\|a-a_\star\|_\star$")
    axes[0].set_ylabel("STL / raw variance")
    panel_letter(axes[0], "a", "Estimator ratio")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\|a-a_\star\|_\star$")
    axes[1].set_ylabel("tangent variance")
    panel_letter(axes[1], "b", "Measured vs Lemma 4.7")

    axes[2].axhline(1.0, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                    label="Lemma 4.7 bound")
    axes[2].set_xscale("log")
    axes[2].set_ylim(0.0, 1.15)
    axes[2].set_xlabel(r"$\|a-a_\star\|_\star$")
    axes[2].set_ylabel("measured / bound")
    panel_letter(axes[2], "c", "Bound tightness")

    legend_below(figure, axes, columns=4)
    worst = float((table["measured_tangent_variance"] / table["lemma_variance_bound"]).max())
    caption = (
        "Estimator ablation and the sticking-the-landing variance bound, on a Gaussian and "
        "a shifted log-cosh target with $d=8$ and $B\\in\\{1,4,16\\}$, along a path "
        "interpolating from the initialization to the optimizer. (a) The intrinsic "
        "Fisher--Rao variance of the STL mean estimator relative to the raw-score "
        "estimator collapses as the state approaches $a_\\star$; for the Gaussian target it "
        "is exactly zero at $a_\\star$, which is off a logarithmic axis and so is omitted "
        "from the panel. (b) The measured Fisher--Rao tangent variance (thin lines with "
        "markers) against the Lemma 4.7 bound "
        "$(2\\|\\mathrm{grad}\\,\\mathcal E\\|_a^2 + \\tfrac32\\Psi(a))/B$ (thick pale "
        f"lines). (c) The bound holds at every state and batch size, with a worst-case "
        f"ratio of {worst:.3f}. This is an estimator ablation, not one of the six compared "
        "algorithms."
    )
    return figure, caption, table


# --------------------------------------------------------------------------
# Experiment J: minibatch residual floors
# --------------------------------------------------------------------------


def figure_j(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    frame = frame.copy()
    frame["rescued"] = frame["variant"].str.contains("-qr", na=False)
    frame["swept_step"] = frame["variant"].str.contains("-hx", na=False)
    tail = frame[frame["iteration"] >= 0.75 * frame["iteration"].max()]
    floors = (
        tail.groupby(["job_id", "method", "rescued", "swept_step", "step_size",
                      "normalized_step_size", "grid_dimension", "grid_condition",
                      "grid_rho", "grid_batch_size"])["objective_gap"]
        .median()
        .reset_index(name="terminal_floor")
    )

    figure, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.4), constrained_layout=True)
    base = floors[
        (floors["grid_dimension"] == 8)
        & (floors["grid_condition"] == 10.0)
        & (floors["grid_rho"] == 0.5)
    ]
    default_step = base[~base["swept_step"]]

    # (a) floor against batch size, at each method's own default step
    reference_cell = default_step[
        default_step["rescued"] | (default_step["method"] == "S--FB--GVI")
    ]
    batches = np.asarray(sorted(reference_cell["grid_batch_size"].unique()), dtype=float)
    for method in ordered_methods(reference_cell):
        subset = reference_cell[reference_cell["method"] == method].sort_values("grid_batch_size")
        axes[0, 0].plot(subset["grid_batch_size"], positive(subset["terminal_floor"]),
                        label=method, **method_style(method))
    anchor = float(reference_cell[reference_cell["grid_batch_size"] == 1]["terminal_floor"].median())
    axes[0, 0].plot(batches, anchor / batches, color=REFERENCE_GREY, linestyle=":",
                    linewidth=1.0, label=r"$\propto 1/B$")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(batches)
    axes[0, 0].set_xticklabels([f"{int(b)}" for b in batches])
    axes[0, 0].set_xlabel("batch size $B$")
    axes[0, 0].set_ylabel("terminal gap floor")
    panel_letter(axes[0, 0], "a", r"$d=8$, $\kappa=10$, $\rho=0.5$")

    # (b) floor against stepsize at fixed batch size
    sweep = base[base["rescued"] & base["grid_batch_size"].isin([1, 16])]
    for method in ordered_methods(sweep):
        for batch, subset in sweep[sweep["method"] == method].groupby("grid_batch_size"):
            subset = subset.sort_values("step_size")
            axes[0, 1].plot(
                subset["step_size"], positive(subset["terminal_floor"]),
                color=COLORS.get(method), marker=MARKERS.get(method),
                linestyle="-" if batch == 1 else "--",
                label=f"{method}, $B={int(batch)}$",
            )
    step_exponent = np.nan
    if not sweep.empty:
        steps = np.asarray(sorted(sweep["step_size"].unique()), dtype=float)
        anchor = float(sweep[np.isclose(sweep["step_size"], steps[0])]["terminal_floor"].median())
        axes[0, 1].plot(steps, anchor * steps / steps[0], color=REFERENCE_GREY,
                        linestyle=":", linewidth=1.0, label=r"$\propto \Delta t$")
        axes[0, 1].plot(steps, anchor * (steps / steps[0]) ** 2, color=REFERENCE_GREY,
                        linestyle="--", linewidth=1.0, label=r"$\propto \Delta t^2$")
        fit = sweep.groupby("step_size")["terminal_floor"].median()
        step_exponent = float(
            np.polyfit(np.log(np.asarray(fit.index, dtype=float)), np.log(fit.to_numpy()), 1)[0]
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"step $\Delta t$")
    axes[0, 1].set_ylabel("terminal gap floor")
    panel_letter(axes[0, 1], "b", "Floor scales with the step")

    # (c) trajectories per oracle pair
    cell = frame[
        (frame["grid_dimension"] == 8) & (frame["grid_condition"] == 10.0)
        & (frame["grid_rho"] == 0.5) & (frame["rescued"]) & (~frame["swept_step"])
        & (frame["method"] == "FR--R--STL")
    ]
    for batch, subset in cell.groupby("grid_batch_size"):
        if int(batch) not in {1, 4, 16, 64}:
            continue
        plot_median_band(
            axes[1, 0], subset, "oracle_pairs", "objective_gap",
            label=rf"$B={int(batch)}$",
            style={
                "color": tuple(float(v) for v in plt.get_cmap("viridis")(np.log2(batch) / 6.0)),
                "linestyle": "-",
                "marker": "",
            },
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("gradient--Hessian oracle pairs")
    axes[1, 0].set_ylabel("objective gap")
    panel_letter(axes[1, 0], "c", "FR--R--STL per oracle pair")

    # (d) what the quadratic rescue actually buys: the transient, not the floor
    ablation_rows = []
    cell_all = frame[
        (frame["grid_dimension"] == 8) & (frame["grid_condition"] == 10.0)
        & (frame["grid_rho"] == 0.5) & (~frame["swept_step"])
        & (frame["method"].isin(["FR--R--STL", "FR--KL--STL"]))
    ]
    for (method, rescued, batch), subset in cell_all.groupby(
        ["method", "rescued", "grid_batch_size"]
    ):
        curve = subset.groupby("oracle_pairs")["objective_gap"].median().sort_index()
        floor_value = float(curve.iloc[-max(3, len(curve) // 4):].median())
        reached = curve[curve <= 2.0 * floor_value]
        ablation_rows.append(
            {
                "method": method,
                "rescued": bool(rescued),
                "batch_size": int(batch),
                "floor": floor_value,
                "oracle_pairs_to_floor": float(reached.index[0]) if len(reached) else np.nan,
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    for method in ordered_methods(ablation.rename(columns={"method": "method"})):
        for rescued, subset in ablation[ablation["method"] == method].groupby("rescued"):
            subset = subset.sort_values("batch_size")
            axes[1, 1].plot(
                subset["batch_size"], subset["oracle_pairs_to_floor"],
                color=COLORS.get(method), linestyle="-" if rescued else "--",
                marker=MARKERS.get(method), alpha=1.0 if rescued else 0.55,
                label=f"{method}{'+QR' if rescued else ' (plain start)'}",
            )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(batches)
    axes[1, 1].set_xticklabels([f"{int(b)}" for b in batches])
    axes[1, 1].set_xlabel("batch size $B$")
    axes[1, 1].set_ylabel("oracle pairs to reach the floor")
    panel_letter(axes[1, 1], "d", "Quadratic-rescue ablation")

    legend_below(figure, axes, columns=4, tidy=False)
    tidy_log_axes(axes[0, 1], axes[1, 0])
    caption = (
        "Minibatch residual floors on the shifted log-cosh family over "
        "$d\\in\\{8,32\\}$, $\\kappa\\in\\{10,10^2\\}$, $\\rho\\in\\{0.5,1\\}$ and "
        "$B\\in\\{1,\\dots,64\\}$, quadratic-rescued, 30 paired seeds per cell and 6000 "
        "iterations so the deterministic transient has died out before the floor is read "
        "off. (a) The floor decreases exactly like $1/B$, the "
        "$O(\\mathfrak V_\\bullet/B)$ prediction of Theorems 4.16 and 4.17; the fitted "
        "exponent is within $1\\%$ of $-1$ in all eight cells. (b) Its dependence on the "
        f"step is steeper than linear, with fitted exponent {step_exponent:.2f}, and lies "
        "between the $\\Delta t$ and $\\Delta t^2$ references. That is what the "
        "sticking-the-landing structure predicts: the Gaussian-core part of the STL noise "
        "is itself proportional to $C_n - C_\\star$, whose stationary size is already "
        "$O(\\Delta t)$, so it contributes at order $\\Delta t^2$ while the genuinely "
        "non-Gaussian part contributes at order $\\Delta t$. Floor levels are not comparable "
        "across geometries, because $\\Delta t$ carries different units in each. "
        "(c) The per-oracle-pair cost of increasing $B$. (d) The quadratic rescue leaves the "
        "floor unchanged -- it is a property of the local dynamics, not of the "
        "initialization -- and instead removes the transient, reaching the floor in fewer "
        "oracle pairs."
    )
    return figure, caption, floors


# --------------------------------------------------------------------------
# Experiment K: decreasing stepsizes
# --------------------------------------------------------------------------


def figure_k(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), constrained_layout=True)
    job = "K_decreasing_d8_rho1_B8"
    representative = frame[frame["job_id"] == job]
    if representative.empty:
        job = sorted(frame["job_id"].unique())[0]
        representative = frame[frame["job_id"] == job]
    representative = representative[representative["iteration"] > 0]

    # Theorem 4.21 is a statement about the expectation, so the sample mean with
    # a standard-error band is the faithful summary here, not the median.
    summary_rows = []
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method]
        grouped = subset.groupby("iteration")["objective_gap"]
        iterations = np.asarray(sorted(subset["iteration"].unique()), dtype=np.float64)
        mean = grouped.mean().to_numpy()
        error = (grouped.std(ddof=1) / np.sqrt(grouped.count())).to_numpy()
        style = method_style(method)
        band(axes[0], iterations, positive(mean - error), positive(mean + error), style["color"])
        axes[0].plot(iterations, positive(mean), label=method, **style,
                     markevery=max(1, len(iterations) // 8))
        # A rolling median over recorded iterations removes the sampling noise
        # from the rescaled diagnostic without changing its level.
        rescaled = pd.Series(iterations * mean).rolling(25, center=True, min_periods=5).median()
        axes[1].plot(iterations, rescaled.to_numpy(), label=method, **style,
                     markevery=max(1, len(iterations) // 8))
        summary_rows.append(
            pd.DataFrame(
                {
                    "job_id": job,
                    "method": method,
                    "iteration": iterations,
                    "mean_gap": mean,
                    "standard_error": error,
                    "rescaled_gap": iterations * mean,
                }
            )
        )
    iterations = np.asarray(sorted(representative["iteration"].unique()), dtype=np.float64)
    anchor_index = len(iterations) // 4
    anchor = float(
        representative[representative["iteration"] == iterations[anchor_index]][
            "objective_gap"
        ].mean()
    )
    axes[0].plot(iterations, anchor * iterations[anchor_index] / iterations,
                 color=REFERENCE_GREY, linestyle=":", linewidth=1.0, label=r"slope $-1$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration $N$")
    axes[0].set_ylabel(r"$\mathbb{E}\,\Delta(a_N)$")
    panel_letter(axes[0], "a", "Decreasing schedule")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("iteration $N$")
    axes[1].set_ylabel(r"$N\,\mathbb{E}\,\Delta(a_N)$")
    panel_letter(axes[1], "b", "Rescaled gap")

    # Fitted log-log slope over the final decade, annotated so the claim about
    # the O(1/N) regime is checkable rather than eyeballed.
    tail_slopes = []
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method]
        grouped = subset.groupby("iteration")["objective_gap"].mean()
        iterations = np.asarray(grouped.index, dtype=np.float64)
        values = grouped.to_numpy()
        window = iterations >= 0.1 * iterations.max()
        slope = float(np.polyfit(np.log(iterations[window]), np.log(values[window]), 1)[0])
        tail_slopes.append((method, slope))
    axes[0].text(
        0.03, 0.06,
        "\n".join(f"{m}: slope {s:.2f}" for m, s in tail_slopes),
        transform=axes[0].transAxes, fontsize=6.5, va="bottom",
    )

    legend_below(figure, axes)
    caption = (
        "Decreasing-stepsize schedule $\\Delta t_n = 8\\kappa_\\star/(n+n_0)$ with "
        "$n_0=\\lceil 64\\kappa_\\star^2\\rceil$, exactly as in Theorem 4.21, run for 20000 "
        "iterations with 30 seeds on shifted log-cosh targets with $d\\in\\{4,8\\}$, "
        "$\\rho\\in\\{0.3,1\\}$ and $B\\in\\{1,8\\}$. Curves are sample means over seeds "
        "with standard-error bands. (a) The expected objective gap follows the predicted "
        "$O(1/N)$ decay with no additive stochastic floor. (b) The rescaled quantity "
        "$N\\,\\mathbb E\\,\\Delta(a_N)$, smoothed by a rolling median over recorded "
        "iterations. The dip and subsequent plateau is the crossover from the "
        "deterministic-contraction regime to the noise-dominated regime in which the "
        "$K/(N+n_0)$ bound of the theorem is the binding one; the annotated log--log "
        "slopes are fitted over the final decade."
    )
    return figure, caption, pd.concat(summary_rows, ignore_index=True)


# --------------------------------------------------------------------------
# Experiment M: general affine-invariant metrics of Section 5
# --------------------------------------------------------------------------


def figure_m(frame: pd.DataFrame) -> tuple[plt.Figure, str, pd.DataFrame]:
    summary = (
        frame.groupby(["job_id", "omega", "tau", "dimension", "step_size"])[
            ["predicted_traceless_rate", "fitted_traceless_rate",
             "predicted_trace_rate", "fitted_trace_rate"]
        ]
        .first()
        .reset_index()
    )
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), constrained_layout=True)

    finest = summary[summary["step_size"] == summary["step_size"].min()]
    for dimension, subset in finest.groupby("dimension"):
        axes[0].plot(subset["predicted_traceless_rate"], subset["fitted_traceless_rate"],
                     linestyle="none", marker="o", label=f"$N={int(dimension)}$")
        axes[1].plot(subset["predicted_trace_rate"], subset["fitted_trace_rate"],
                     linestyle="none", marker="s", label=f"$N={int(dimension)}$")
    for axis, predicted, xlabel, title in (
        (axes[0], "predicted_traceless_rate", r"$1/(2\omega)$", "Traceless mode"),
        (axes[1], "predicted_trace_rate", r"$1/(2(\omega+\tau N))$", "Trace mode"),
    ):
        values = finest[predicted]
        limits = [float(values.min()) * 0.8, float(values.max()) * 1.2]
        axis.plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                  label="identity")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"predicted {xlabel}")
        axis.set_ylabel("fitted rate")
    panel_letter(axes[0], "a", "Traceless mode")
    panel_letter(axes[1], "b", "Trace mode")

    refinement = summary.copy()
    refinement["traceless_error"] = (
        refinement["fitted_traceless_rate"] / refinement["predicted_traceless_rate"] - 1.0
    ).abs()
    refinement["trace_error"] = (
        refinement["fitted_trace_rate"] / refinement["predicted_trace_rate"] - 1.0
    ).abs()
    grouped = refinement.groupby("step_size")[["traceless_error", "trace_error"]].median()
    axes[2].plot(grouped.index, positive(grouped["traceless_error"]), marker="o",
                 label="traceless mode")
    axes[2].plot(grouped.index, positive(grouped["trace_error"]), marker="s", linestyle="--",
                 label="trace mode")
    steps = np.asarray(grouped.index, dtype=float)
    axes[2].plot(steps, steps * float(grouped["traceless_error"].iloc[-1]) / steps[-1],
                 color=REFERENCE_GREY, linestyle=":", linewidth=1.0, label=r"$O(\Delta t)$")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"step $\Delta t$")
    axes[2].set_ylabel("relative rate error")
    panel_letter(axes[2], "c", "Step refinement")

    legend_below(figure, axes)
    caption = (
        "Verification of the affine-invariant metric classification of Section 5. For "
        "members $(\\omega,\\tau)$ of the classified family, run through the Riemannian "
        "retraction discretization on Gaussian targets with $N\\in\\{2,5,10\\}$, the "
        "traceless and trace covariance modes decay at the predicted rates $1/(2\\omega)$ "
        "and $1/(2(\\omega+\\tau N))$. Panel (c) shows the residual discrepancy is the "
        "$O(\\Delta t)$ discretization bias of the retraction and vanishes under step "
        "refinement. The Fisher--Rao metric is the member $(\\omega,\\tau)=(1/2,0)$, for "
        "which both rates coincide at $1$."
    )
    return figure, caption, summary


BUILDERS = {
    "A": figure_a,
    "B": figure_b,
    "C": figure_c,
    "D": figure_d,
    "F": figure_f,
    "G": figure_g,
    "H": figure_h,
    "I": figure_i,
    "J": figure_j,
    "K": figure_k,
    "L": figure_l,
    "M": figure_m,
}


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--tier", default="full")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = args.experiments or sorted(BUILDERS)
    for experiment in selected:
        frame = load_experiment(experiment, args.tier)
        if frame.empty:
            print(f"experiment {experiment}: no data")
            continue
        figure, caption, table = BUILDERS[experiment](frame)
        save_figure(figure, f"experiment_{experiment}", caption, table)
        print(f"experiment {experiment}: {len(frame)} rows -> figure written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
