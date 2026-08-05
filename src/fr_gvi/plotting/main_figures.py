"""Composite manuscript figures.

These are the figures intended to appear in the paper.  Each one collects the
panels that carry a single message across several experiments; the per-experiment
figures in ``fr_gvi.plotting.figures`` remain the detailed supporting evidence.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fr_gvi.plotting.figures import ordered_methods, plot_median_band
from fr_gvi.plotting.style import (
    COLORS,
    MARKERS,
    REFERENCE_GREY,
    TEXT_WIDTH,
    configure_style,
    load_experiment,
    method_style,
    panel_letter,
    positive,
    save_figure,
    tidy_log_axes,
)


def _legend(figure: plt.Figure, axes, columns: int = 4) -> None:
    handles: list = []
    labels: list[str] = []
    for axis in np.atleast_1d(axes).ravel():
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles, labels, loc="outside lower center",
            ncols=min(columns, len(labels)), frameon=False,
        )


def _terminal(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("job_id", "method", "variant", "seed") if k in frame.columns]
    return frame.sort_values("iteration").groupby(keys, as_index=False).last()


def main_figure_1() -> None:
    """Affine invariance: burn-in law, equivariance, and the Diao benchmark."""

    burn = load_experiment("A")
    affine = load_experiment("B")
    diao = load_experiment("C")
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)

    rows = []
    for (_, method, _), trajectory in burn.groupby(["job_id", "method", "seed"]):
        trajectory = trajectory.sort_values("iteration")
        beta_star = float(trajectory["beta_star"].iloc[0])
        entered = trajectory[
            trajectory["relative_covariance_min_eigenvalue"] >= 1.0 / (2.0 * beta_star)
        ]
        if entered.empty:
            continue
        step = float(trajectory["step_size"].iloc[-1])
        rows.append(
            {
                "method": method,
                "measured": int(entered["iteration"].iloc[0]) * step,
                "predicted": float(
                    np.log(1.0 / (beta_star * float(trajectory["lambda_0_star"].iloc[0])))
                ),
            }
        )
    burn_table = pd.DataFrame(rows)
    for method in ordered_methods(burn_table):
        subset = burn_table[burn_table["method"] == method]
        axes[0].plot(
            subset["predicted"], subset["measured"], linestyle="none", label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    limits = [0.0, float(burn_table["predicted"].max()) * 1.05]
    axes[0].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label="theory")
    axes[0].set_xlabel(r"$\log[1/(\beta_\star\lambda_{0,\star})]$")
    axes[0].set_ylabel(r"$N_{\mathrm{cov}}\,\Delta t$")
    panel_letter(axes[0], "a", "Covariance burn-in")

    summary = (
        affine.groupby(["method", "grid_affine_condition"])["equivariance_error_covariance"]
        .max()
        .reset_index()
    )
    for method in ordered_methods(summary):
        subset = summary[summary["method"] == method].sort_values("grid_affine_condition")
        axes[1].plot(
            subset["grid_affine_condition"], positive(subset["equivariance_error_covariance"]),
            label=method, **method_style(method),
        )
    conditions = np.asarray(sorted(summary["grid_affine_condition"].unique()), dtype=float)
    axes[1].plot(
        conditions, np.finfo(np.float64).eps * conditions, color=REFERENCE_GREY,
        linestyle=":", linewidth=1.0, label=r"$\varepsilon_{\mathrm{mach}}\,K$",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"conditioning $K$ of the change of variables")
    axes[1].set_ylabel("max equivariance error")
    panel_letter(axes[1], "b", "Affine equivariance")

    competitive = diao[~diao["variant"].str.contains("rescue", na=False)]
    terminal = _terminal(competitive)
    for method in ordered_methods(competitive):
        subset = terminal[terminal["method"] == method]
        best_step = float(
            subset.groupby("normalized_step_size")["exact_gaussian_kl"].median().idxmin()
        )
        trajectory = competitive[
            (competitive["method"] == method)
            & np.isclose(competitive["normalized_step_size"], best_step)
        ]
        plot_median_band(
            axes[2], trajectory, "iteration", "exact_gaussian_kl",
            label=method, style=method_style(method),
        )
    axes[2].set_xlabel("iteration $n$")
    axes[2].set_ylabel("exact KL gap")
    panel_letter(axes[2], "c", r"$\kappa=10^9$, $\kappa_\star=1$")

    tidy_log_axes(*axes)
    _legend(figure, axes)
    save_figure(
        figure,
        "main_figure_1",
        "Affine invariance of the Gaussian Fisher--Rao flow. "
        "(a) The optimizer-whitened covariance band is entered after the predicted "
        "$\\log[1/(\\beta_\\star\\lambda_{0,\\star})]$ of Theorem 2.9, over "
        "$\\kappa\\in\\{10,10^2,10^3\\}$ and $\\lambda_0\\in\\{10^{-2},\\dots,10^{-12}\\}$. "
        "(b) Transformed Fisher--Rao iterates agree with the transported reference to "
        "within roundoff over eight decades of coordinate conditioning: the residual grows "
        "like the conditioning of the change of variables itself, from $10^{-15}$ at "
        "$K=1$ to $10^{-9}$ at $K=10^8$, which is floating-point amplification through an "
        "ill-conditioned map rather than a loss of equivariance. FB--GVI, whose "
        "Bures--Wasserstein geometry is not affine invariant, departs by an $O(1)$ amount "
        "as soon as the map is non-orthogonal. "
        "(c) On the anisotropic Gaussian of Diao et al., where $\\kappa=10^9$ but "
        "$\\kappa_\\star=1$, both Fisher--Rao schemes reach machine precision at their own "
        "best swept step while FB--GVI remains limited by the original-coordinate "
        "conditioning.",
        burn_table,
    )


def main_figure_2() -> None:
    """Local convergence: exact Gaussian identity and the near-Gaussian spectral gap."""

    gaussian = load_experiment("F")
    spectral = load_experiment("G")
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)

    gaussian = gaussian.copy()
    gaussian["distance_squared"] = (
        gaussian["mean_error"] ** 2 + 0.5 * gaussian["covariance_error"] ** 2
    )
    representative = gaussian[gaussian["grid_dimension"] == 10]
    smallest = float(representative["grid_rate"].min())
    for rate, subset in representative.groupby("grid_rate"):
        first = subset[subset["method"] == "FR--R"].sort_values("iteration")
        step = float(first["step_size"].iloc[-1])
        times = first["iteration"].to_numpy() * step
        axes[0].plot(times, positive(first["distance_squared"]), color=COLORS["FR--R"],
                     linewidth=1.1, label="FR--R" if rate == smallest else None)
        start = float(first["distance_squared"].iloc[0])
        axes[0].plot(times, start * np.exp(-rate * times), color=REFERENCE_GREY,
                     linestyle=":", linewidth=0.9,
                     label=r"bound $e^{-2\ell_\delta t}$" if rate == smallest else None)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"time $t=n\,\Delta t$")
    axes[0].set_ylabel(r"$\|a_n-a_\star\|_\star^2$")
    panel_letter(axes[0], "a", "Exact Gaussian region")

    rows = []
    for (_, method), subset in gaussian.groupby(["job_id", "method"]):
        subset = subset.sort_values("iteration").reset_index(drop=True)
        step = float(subset["step_size"].iloc[-1])
        values = subset["distance_squared"].to_numpy()
        core = subset["gaussian_core_rate"].to_numpy()
        for index in range(len(values) - 1):
            if not (values[index] > 1e-20 and values[index + 1] > 1e-20):
                continue
            rows.append(
                {
                    "method": method,
                    "core_rate": float(core[index]),
                    "measured_rate": -float(np.log(values[index + 1] / values[index])) / step,
                }
            )
    identity = pd.DataFrame(rows)
    for method in ordered_methods(identity):
        subset = identity[identity["method"] == method]
        axes[1].plot(
            subset["core_rate"], subset["measured_rate"], linestyle="none",
            markersize=1.8, alpha=0.4, label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    limits = [
        float(identity["core_rate"].min()) * 0.95,
        float(identity["core_rate"].max()) * 1.05,
    ]
    axes[1].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
                 label="identity")
    axes[1].set_xlabel(r"$q_{\mathrm{G}}(a_n)$")
    axes[1].set_ylabel("measured rate")
    panel_letter(axes[1], "b", "Lemma 2.24")

    rows = []
    for (_, method), subset in spectral.groupby(["job_id", "method"]):
        subset = subset.sort_values("iteration")
        step = float(subset["step_size"].iloc[-1])
        times = subset["iteration"].to_numpy() * step
        distance = (subset["mean_error"] ** 2 + 0.5 * subset["covariance_error"] ** 2).to_numpy()
        usable = np.isfinite(distance) & (distance > 1e-22)
        if not usable.any():
            continue
        tail = usable & (times >= 0.55 * times[usable].max())
        if tail.sum() < 4:
            continue
        rows.append(
            {
                "method": method,
                "fitted": float(-np.polyfit(times[tail], np.log(distance[tail]), 1)[0]),
                "predicted": float(subset["local_discrete_rate"].iloc[0]),
                "rho": float(subset["grid_rho"].iloc[0]),
            }
        )
    spectral_table = pd.DataFrame(rows)
    for method in ordered_methods(spectral_table):
        subset = spectral_table[spectral_table["method"] == method]
        axes[2].plot(
            subset["predicted"], subset["fitted"], linestyle="none", label=method,
            **{k: v for k, v in method_style(method).items() if k != "linestyle"},
        )
    limits = [
        float(spectral_table["predicted"].min()) * 0.97,
        float(spectral_table["predicted"].max()) * 1.03,
    ]
    axes[2].plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("predicted one-step rate")
    axes[2].set_ylabel("fitted rate")
    panel_letter(axes[2], "c", r"Near-Gaussian $\gamma_\star$")

    tidy_log_axes(*axes)
    _legend(figure, axes)
    save_figure(
        figure,
        "main_figure_2",
        "Local convergence. (a) For a standard Gaussian target the trajectories obey the "
        "uniform bound $e^{-2\\ell_\\delta t}$ of Corollary 2.26. (b) The measured "
        "instantaneous rate coincides with the exact Gaussian-core rate $q_{\\mathrm G}$ of "
        "Lemma 2.24 over $d\\in\\{2,10,100\\}$ and four initial covariance eigenvalues. "
        "(c) On near-Gaussian log-cosh targets the fitted asymptotic rate of both "
        "discretizations matches the rate predicted by the spectral gap of the linearized "
        "generator $\\mathcal L_\\star$ of Proposition 3.5 and Theorem 3.7.",
        spectral_table,
    )


def main_figure_3() -> None:
    """Stochastic behaviour: STL cancellation, minibatch floors, decreasing steps."""

    cancellation = load_experiment("H")
    floors_frame = load_experiment("J")
    decreasing = load_experiment("K")
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)

    spread = (
        cancellation.groupby(["job_id", "method", "iteration"])["objective"]
        .agg(lambda values: float(values.max() - values.min()))
        .reset_index(name="spread")
        .merge(cancellation.groupby("job_id")["grid_dimension"].first(), on="job_id")
    )
    representative = spread[spread["grid_dimension"] == 10]
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method].sort_values("iteration")
        axes[0].plot(subset["iteration"], positive(subset["spread"]), label=method,
                     **method_style(method), markevery=max(1, len(subset) // 8))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel("across-seed objective spread")
    panel_letter(axes[0], "a", r"STL cancellation, $B=1$")

    floors_frame = floors_frame.copy()
    floors_frame["rescued"] = floors_frame["variant"].str.contains("-qr", na=False)
    tail = floors_frame[floors_frame["iteration"] >= 0.75 * floors_frame["iteration"].max()]
    floors = (
        tail.groupby(["method", "rescued", "grid_dimension", "grid_condition", "grid_rho",
                      "grid_batch_size"])["objective_gap"]
        .median()
        .reset_index(name="floor")
    )
    cell = floors[
        (floors["grid_dimension"] == 8)
        & (floors["grid_condition"] == 10.0)
        & (floors["grid_rho"] == 0.5)
    ]
    cell = cell[cell["rescued"] | (cell["method"] == "S--FB--GVI")]
    for method in ordered_methods(cell):
        subset = cell[cell["method"] == method].sort_values("grid_batch_size")
        axes[1].plot(subset["grid_batch_size"], positive(subset["floor"]), label=method,
                     **method_style(method))
    batches = np.asarray(sorted(cell["grid_batch_size"].unique()), dtype=float)
    anchor = float(cell[cell["grid_batch_size"] == 1]["floor"].median())
    axes[1].plot(batches, anchor / batches, color=REFERENCE_GREY, linestyle=":",
                 linewidth=1.0, label=r"$\propto 1/B$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size $B$")
    axes[1].set_ylabel("terminal gap floor")
    panel_letter(axes[1], "b", "Minibatch floors")

    job = "K_decreasing_d8_rho1_B8"
    representative = decreasing[decreasing["job_id"] == job]
    if representative.empty:
        representative = decreasing[decreasing["job_id"] == sorted(decreasing["job_id"])[0]]
    representative = representative[representative["iteration"] > 0]
    for method in ordered_methods(representative):
        subset = representative[representative["method"] == method]
        plot_median_band(axes[2], subset, "iteration", "objective_gap",
                         label=method, style=method_style(method))
    iterations = np.asarray(sorted(representative["iteration"].unique()), dtype=float)
    anchor_index = len(iterations) // 4
    anchor = float(
        representative[representative["iteration"] == iterations[anchor_index]][
            "objective_gap"
        ].median()
    )
    axes[2].plot(iterations, anchor * iterations[anchor_index] / iterations,
                 color=REFERENCE_GREY, linestyle=":", linewidth=1.0, label=r"slope $-1$")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("iteration $N$")
    axes[2].set_ylabel(r"$\mathbb{E}\,\Delta(a_N)$")
    panel_letter(axes[2], "c", "Decreasing steps")

    tidy_log_axes(axes[0], axes[2])
    axes[1].set_xticks(batches)
    axes[1].set_xticklabels([f"{int(b)}" for b in batches])
    _legend(figure, axes)
    save_figure(
        figure,
        "main_figure_3",
        "Stochastic Fisher--Rao schemes with the Price/Hessian--STL estimator. "
        "(a) With the covariance matched to a Gaussian target the STL mean noise vanishes "
        "pathwise, so the Fisher--Rao trajectories are seed-independent to floating-point "
        "precision while S--FB--GVI retains its native estimator noise; this compares "
        "complete algorithms together with their estimators, not geometry alone. "
        "(b) Terminal objective-gap floors decrease like $1/B$, matching the "
        "$O(\\mathfrak V_\\bullet/B)$ prediction of Theorems 4.16 and 4.17. "
        "(c) With the schedule $\\Delta t_n=8\\kappa_\\star/(n+n_0)$ of Theorem 4.21 the "
        "expected gap follows $O(1/N)$ and the additive floor disappears.",
        floors,
    )


def main_figure_4() -> None:
    """Applications: log-cosh grid and Bayesian logistic regression."""

    logcosh = load_experiment("D")
    logistic = load_experiment("L")
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)

    terminal = _terminal(logcosh)
    best = terminal.loc[terminal.groupby(["job_id", "method"])["objective_gap"].idxmin()]
    sizes = {2: 12.0, 10: 22.0, 50: 38.0}
    for method in ordered_methods(best):
        subset = best[best["method"] == method]
        axes[0].scatter(
            subset["kappa_star"], positive(subset["objective_gap"]),
            s=[sizes.get(int(d), 20.0) for d in subset["grid_dimension"]],
            facecolors="none", edgecolors=COLORS.get(method), linewidths=0.9,
            marker=MARKERS.get(method), label=method,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\kappa_\star$")
    axes[0].set_ylabel("best gap at fixed budget")
    panel_letter(axes[0], "a", "Log-cosh, 27 cells")

    if not logistic.empty:
        job = "L_logistic_d50_lam1_fc1e2"
        cell = logistic[logistic["job_id"] == job]
        if cell.empty:
            job = sorted(logistic["job_id"].unique())[0]
            cell = logistic[logistic["job_id"] == job]
        iterative = cell[cell["method"] != "Laplace"]
        terminal_cell = _terminal(iterative)
        for method in ordered_methods(iterative):
            subset = terminal_cell[terminal_cell["method"] == method]
            if subset.empty:
                continue
            best_step = float(
                subset.groupby("normalized_step_size")["objective_gap"].median().idxmin()
            )
            trajectory = iterative[
                (iterative["method"] == method)
                & np.isclose(iterative["normalized_step_size"], best_step)
            ]
            plot_median_band(axes[1], trajectory, "oracle_pairs", "objective_gap",
                             label=method, style=method_style(method))
            plot_median_band(axes[2], trajectory, "oracle_pairs", "predictive_nll",
                             label=method, style=method_style(method), log_y=False)
        laplace = cell[cell["method"] == "Laplace"]
        if not laplace.empty:
            axes[1].axhline(float(laplace["objective_gap"].iloc[-1]), color=COLORS["Laplace"],
                            linestyle=":", linewidth=1.1, label="Laplace")
            axes[2].axhline(float(laplace["predictive_nll"].iloc[-1]), color=COLORS["Laplace"],
                            linestyle=":", linewidth=1.1, label="Laplace")
        axes[1].set_xlabel("oracle pairs")
        axes[1].set_ylabel("objective gap")
        panel_letter(axes[1], "b", job.replace("_", " "))
        axes[2].set_xlabel("oracle pairs")
        axes[2].set_ylabel("held-out predictive NLL")
        panel_letter(axes[2], "c", "Predictive quality")

    tidy_log_axes(*axes)
    _legend(figure, axes)
    save_figure(
        figure,
        "main_figure_4",
        "Applications. (a) Best objective gap at a fixed oracle budget over the full "
        "shifted log-cosh grid ($d\\in\\{2,10,50\\}$, $\\kappa_{\\rm base}\\in\\{1,10,10^2\\}$, "
        "$\\rho\\in\\{0.1,1,5\\}$), each method at its own best swept step, plotted against "
        "the affine-invariant condition number $\\kappa_\\star$; marker size encodes the "
        "dimension. (b, c) Bayesian logistic regression with a proper Gaussian prior: "
        "objective gap and held-out predictive negative log-likelihood against the number "
        "of gradient--Hessian oracle pairs, with the non-iterative Laplace baseline shown "
        "for reference.",
        best,
    )


def main_figure_5() -> None:
    """Section 5: modal rates of the general affine-invariant metric family."""

    frame = load_experiment("M")
    summary = (
        frame.groupby(["job_id", "omega", "tau", "dimension", "step_size"])[
            ["predicted_traceless_rate", "fitted_traceless_rate",
             "predicted_trace_rate", "fitted_trace_rate"]
        ]
        .first()
        .reset_index()
    )
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.4), constrained_layout=True)
    finest = summary[summary["step_size"] == summary["step_size"].min()]

    for dimension, subset in finest.groupby("dimension"):
        axes[0].plot(subset["predicted_traceless_rate"], subset["fitted_traceless_rate"],
                     linestyle="none", marker="o", markersize=3.5, fillstyle="none",
                     label=f"$N={int(dimension)}$")
        axes[1].plot(subset["predicted_trace_rate"], subset["fitted_trace_rate"],
                     linestyle="none", marker="s", markersize=3.5, fillstyle="none",
                     label=f"$N={int(dimension)}$")
    for axis, column, xlabel in (
        (axes[0], "predicted_traceless_rate", r"$1/(2\omega)$"),
        (axes[1], "predicted_trace_rate", r"$1/(2(\omega+\tau N))$"),
    ):
        values = finest[column]
        limits = [float(values.min()) * 0.8, float(values.max()) * 1.25]
        axis.plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(f"predicted {xlabel}")
        axis.set_ylabel("fitted rate")
    panel_letter(axes[0], "a", "Traceless mode")
    panel_letter(axes[1], "b", "Trace mode")

    refinement = summary.copy()
    refinement["traceless"] = (
        refinement["fitted_traceless_rate"] / refinement["predicted_traceless_rate"] - 1.0
    ).abs()
    refinement["trace"] = (
        refinement["fitted_trace_rate"] / refinement["predicted_trace_rate"] - 1.0
    ).abs()
    grouped = refinement.groupby("step_size")[["traceless", "trace"]].median()
    axes[2].plot(grouped.index, positive(grouped["traceless"]), marker="o", label="traceless")
    axes[2].plot(grouped.index, positive(grouped["trace"]), marker="s", linestyle="--",
                 label="trace")
    steps = np.asarray(grouped.index, dtype=float)
    axes[2].plot(steps, steps * float(grouped["traceless"].iloc[-1]) / steps[-1],
                 color=REFERENCE_GREY, linestyle=":", linewidth=1.0, label=r"$O(\Delta t)$")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"step $\Delta t$")
    axes[2].set_ylabel("relative rate error")
    panel_letter(axes[2], "c", "Step refinement")

    tidy_log_axes(*axes)
    _legend(figure, axes)
    save_figure(
        figure,
        "main_figure_5",
        "Verification of the affine-invariant metric classification of Section 5. Running "
        "members $(\\omega,\\tau)$ of the classified family through the Riemannian retraction "
        "discretization on Gaussian targets with $N\\in\\{2,5,10\\}$, the traceless and trace "
        "covariance modes decay at the predicted rates $1/(2\\omega)$ and "
        "$1/(2(\\omega+\\tau N))$. Panel (c) shows the residual discrepancy is the "
        "$O(\\Delta t)$ bias of the retraction and vanishes under step refinement. The "
        "Fisher--Rao metric is the member $(\\omega,\\tau)=(1/2,0)$, where the two rates "
        "coincide at $1$.",
        summary,
    )


BUILDERS = {
    1: main_figure_1,
    2: main_figure_2,
    3: main_figure_3,
    4: main_figure_4,
    5: main_figure_5,
}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="*", type=int, default=None)
    args = parser.parse_args(arguments)
    configure_style()
    for index in args.figures or sorted(BUILDERS):
        BUILDERS[index]()
        print(f"main figure {index} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
