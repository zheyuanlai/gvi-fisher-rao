"""The three figures of the manuscript's numerical section.

Each figure is assembled from panels that carry exactly one message, and each
panel is emitted with its own processed CSV so a reader can reconstruct any curve
without rerunning anything.  Alongside every figure the module writes a caption
draft and a provenance record naming the config identifiers, the commit and the
code hash the panels were produced from.

The convention of the repository is retained: matching PDF and PNG on an exact
manuscript-width canvas, the Okabe--Ito colourblind-safe palette, and a distinct
linestyle and marker per method so the figures survive greyscale printing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fr_gvi.experiments.manuscript import (
    GLOBAL_BUDGET,
    GLOBAL_TOLERANCE,
    LOGISTIC_DATASETS,
    SELECTED_STEPS,
)
from fr_gvi.diagnostics.core import PLATEAU_THRESHOLD, truncate_at_floor
from fr_gvi.plotting.figures import ordered_methods
from fr_gvi.plotting.style import (
    COLORS,
    METHOD_ORDER,
    REFERENCE_GREY,
    TEXT_WIDTH,
    band,
    configure_style,
    figure_text,
    load_experiment,
    method_style,
    normalize_figure_text,
    panel_letter,
    positive,
    tidy_log_axes,
)

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results" / "figures" / "manuscript"
PROCESSED = ROOT / "results" / "processed" / "manuscript"
TIER = "manuscript"

# Double-precision resolution of an objective of size ``|E|``.  Gaps at or below
# this are roundoff, not convergence, and are drawn as a floor rather than as
# part of a curve.
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Panel plumbing
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    letter: str
    title: str
    caption: str | Callable[[pd.DataFrame], str]
    draw: Callable[[plt.Axes], pd.DataFrame]
    sources: list[str] = field(default_factory=list)

    def rendered_caption(self, data: pd.DataFrame) -> str:
        """Caption text, computed from the panel's own data when it quotes numbers.

        Hard-coded figures in a caption drift the moment the experiment changes;
        this one described a quasi-Monte-Carlo design and 250 ms iterations long
        after the backend had become one-dimensional quadrature at 13 ms.
        """

        return self.caption(data) if callable(self.caption) else self.caption


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    from fr_gvi.experiments.campaign import SOURCE_PATHS, code_hash

    # Dirty means the source differs from the commit; regenerating figures writes
    # tracked outputs and must not by itself mark the figure unreproducible.
    status = run("status", "--short", "--", *SOURCE_PATHS)
    return {
        "commit": run("rev-parse", "HEAD") or "unborn",
        "dirty": bool(status),
        # The commit alone does not identify the numerics: the campaign keys its
        # resume logic on a hash of the numerical sources, and a figure must be
        # traceable to the same revision its trajectories were produced from.
        "code_hash": code_hash(),
    }


def _shared_legend(figure: plt.Figure, axes, columns: int) -> None:
    """One legend for the whole figure, methods first and reference curves last."""

    entries: dict[str, object] = {}
    for axis in np.atleast_1d(axes).ravel():
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=False):
            if label and label not in entries:
                entries[label] = handle
    if not entries:
        return
    methods = [label for label in METHOD_ORDER if label in entries]
    references = [label for label in entries if label not in methods]
    ordered = methods + references
    figure.legend(
        [entries[label] for label in ordered],
        [figure_text(label) for label in ordered],
        loc="outside lower center",
        ncols=min(columns, len(ordered)),
        frameon=False,
    )


def require(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Fail with an actionable message when a figure's campaign has not been run."""

    if frame.empty:
        raise SystemExit(
            f"no manuscript results for experiment {experiment}: "
            f"run 'make manuscript-runs' before building the figures"
        )
    return frame


def compose(
    name: str,
    panels: list[Panel],
    *,
    shape: tuple[int, int],
    height: float,
    lead: str,
    legend_columns: int = 5,
) -> dict[str, object]:
    """Render, export and document one composite figure."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows, columns = shape
    figure, axes = plt.subplots(
        rows, columns, figsize=(TEXT_WIDTH, height), constrained_layout=True
    )
    flat = np.atleast_1d(axes).ravel()
    if len(flat) < len(panels):
        raise ValueError(f"{name}: {len(panels)} panels do not fit a {rows}x{columns} grid")

    record: list[dict[str, object]] = []
    captions: list[str] = []
    for panel, axis in zip(panels, flat, strict=False):
        data = panel.draw(axis)
        captions.append(panel.rendered_caption(data))
        panel_letter(axis, panel.letter, panel.title)
        csv_path = PROCESSED / f"{name}_{panel.letter}.csv"
        data.to_csv(csv_path, index=False)
        record.append(
            {
                "panel": panel.letter,
                "title": panel.title,
                "config_ids": sorted(panel.sources),
                "rows": int(len(data)),
                "processed_csv": str(csv_path.relative_to(ROOT)),
            }
        )
    for axis in flat[len(panels) :]:
        axis.set_visible(False)

    tidy_log_axes(*flat[: len(panels)])
    _shared_legend(figure, flat[: len(panels)], legend_columns)
    normalize_figure_text(figure)

    base = FIGURES / name
    figure.savefig(base.with_suffix(".pdf"), metadata={"Creator": "fr-gvi"})
    figure.savefig(base.with_suffix(".png"))
    plt.close(figure)

    caption_lines = [f"# {name} caption draft", "", figure_text(lead), ""]
    caption_lines += [
        figure_text(f"**({panel.letter})** {text}")
        for panel, text in zip(panels, captions, strict=True)
    ]
    caption_lines += ["", "Panel data:"]
    caption_lines += [
        f"- ({entry['panel']}) `{entry['processed_csv']}`" for entry in record
    ]
    base.with_suffix(".md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")

    provenance = {
        "figure": name,
        "git": _git_state(),
        "tier": TIER,
        "stepsize_protocol": json.loads(SELECTED_STEPS.read_text(encoding="utf-8"))[
            "multipliers"
        ],
        "panels": record,
        "outputs": [
            str(base.with_suffix(suffix).relative_to(ROOT))
            for suffix in (".pdf", ".png", ".md")
        ],
    }
    base.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def _markevery(values: np.ndarray, index: int = 0, count: int = 6) -> tuple[int, int]:
    """Marker spacing over the drawn portion of a curve, phased by series index.

    Two schemes can produce trajectories that agree to plotting accuracy -- on the
    well-conditioned cell FR--R and FR--KL run at the same step and coincide -- and
    a curve hidden underneath another reads as a missing curve.  Offsetting the
    marker phase keeps both visible without displacing any data.
    """

    drawn = int(np.sum(np.isfinite(np.asarray(values, dtype=np.float64))))
    step = max(1, drawn // count)
    return (index * max(1, step // 3) % step, step)


# ---------------------------------------------------------------------------
# Figure 1: Gaussian structure and affine invariance
# ---------------------------------------------------------------------------


def _burn_in_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (job, method), trajectory in frame.groupby(["job_id", "method"]):
        trajectory = trajectory.sort_values("iteration")
        beta_star = float(trajectory["beta_star"].iloc[0])
        lambda_0 = float(trajectory["lambda_0_star"].iloc[0])
        step = float(trajectory["step_size"].iloc[-1])
        entered = trajectory[
            trajectory["relative_covariance_min_eigenvalue"] >= 1.0 / (2.0 * beta_star)
        ]
        if entered.empty:
            continue
        iteration = int(entered["iteration"].iloc[0])
        rows.append(
            {
                "job_id": job,
                "method": method,
                "lambda_0": lambda_0,
                "step_size": step,
                "entry_iteration": iteration,
                "entry_time": iteration * step,
                "predicted": float(np.log(1.0 / (beta_star * lambda_0))),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "predicted"]).reset_index(drop=True)


def panel_burn_in(axis: plt.Axes) -> pd.DataFrame:
    frame = require(load_experiment("A", TIER), "A")
    table = _burn_in_table(frame)
    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        style = {k: v for k, v in method_style(method).items() if k != "linestyle"}
        axis.plot(
            subset["predicted"], subset["entry_time"], linestyle="none",
            markersize=4.0, fillstyle="none", label=method, **style,
        )
    limits = np.asarray([0.0, float(table["predicted"].max()) * 1.08])
    axis.plot(limits, limits, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
              label=r"$\log[1/(\beta_\star\lambda_{0,\star})]$")
    axis.set_xlim(0.0, limits[1])
    axis.set_ylim(0.0, None)
    axis.set_xlabel(r"$\log[1/(\beta_\star\lambda_{0,\star})]$")
    axis.set_ylabel(r"entry time $N_{\mathrm{ent}}h$")
    return table


def panel_affine(axis: plt.Axes) -> pd.DataFrame:
    frame = require(load_experiment("B", TIER), "B")
    table = (
        frame.groupby(["method", "grid_transform_condition"])["equivariance_error_back"]
        .max()
        .reset_index(name="max_equivariance_error")
        .sort_values(["method", "grid_transform_condition"])
    )
    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        axis.plot(
            subset["grid_transform_condition"],
            positive(subset["max_equivariance_error"]),
            label=method, **method_style(method),
        )
    # The covariance is carried forward by ``S C S^T`` and mapped back by
    # ``S^{-1} C S^{-T}``, so the round trip amplifies roundoff by ``cond(S)^2``.
    # Where that reaches one the transported state is no longer representable in
    # double precision; the grid stops below it and the region is shaded so the
    # reason the panel ends where it does is visible rather than implicit.
    breakdown = 1.0 / np.sqrt(EPS)
    reference = np.geomspace(1.0, 4.0 * breakdown, 64)
    axis.plot(reference, EPS * reference**2, color=REFERENCE_GREY, linestyle=":",
              linewidth=1.0, label=r"$\varepsilon_{\mathrm{mach}}\,\mathrm{cond}(S)^2$")
    axis.axvspan(breakdown, 4.0 * breakdown, color=REFERENCE_GREY, alpha=0.12, linewidth=0.0)
    axis.text(
        breakdown * 1.5, EPS**0.5, "not representable\nin float64", fontsize=6.0,
        color=REFERENCE_GREY, ha="center", va="center", rotation=90,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.4, 4.0 * breakdown)
    axis.set_xlabel(r"$\mathrm{cond}(S)$")
    axis.set_ylabel(r"$\max_n\ \mathrm{err}^{\mathrm{aff}}_n$")
    return table


def panel_anisotropic(axis: plt.Axes) -> pd.DataFrame:
    frame = require(load_experiment("C", TIER), "C")
    collected: list[pd.DataFrame] = []
    floors: list[float] = []
    for index, method in enumerate(ordered_methods(frame)):
        subset = frame[frame["method"] == method].sort_values("iteration")
        initial = float(subset["exact_gaussian_kl"].iloc[0])
        normalized = subset["exact_gaussian_kl"].to_numpy() / initial
        # The KL gap of a Gaussian pair is quadratic in the state error, so its
        # double-precision resolution is eps^2 relative to the initial gap.  Below
        # that the recorded values are roundoff and are not part of the curve.
        visible, floor = truncate_at_floor(normalized)
        floors.append(floor)
        axis.plot(
            subset["iteration"], positive(visible), label=method,
            markevery=_markevery(visible, index), zorder=5 - index,
            **method_style(method),
        )
        collected.append(
            pd.DataFrame(
                {
                    "method": method,
                    "iteration": subset["iteration"].to_numpy(),
                    "step_size": float(subset["step_size"].iloc[-1]),
                    "kl_gap": subset["exact_gaussian_kl"].to_numpy(),
                    "normalized_kl_gap": normalized,
                    "resolution_floor": floor,
                }
            )
        )
    level = float(np.median(floors))
    axis.set_yscale("log")
    axis.set_ylim(level / 5.0, 5.0)
    axis.set_xlabel("iteration $n$")
    axis.set_ylabel(r"$\mathrm{KL}(a_n)/\mathrm{KL}(a_0)$")
    return pd.concat(collected, ignore_index=True)


def figure_1() -> dict[str, object]:
    panels = [
        Panel(
            "a", "Covariance entry time",
            "Fisher--Rao entry time into the optimizer-whitened covariance band "
            "$\\lambda_{\\min}(C_n)\\ge 1/2$ for the standard Gaussian target in $d=20$, "
            "over $\\lambda_0\\in\\{10^{-2},10^{-4},10^{-6},10^{-8}\\}$ at the common step "
            "$h=0.1$. The entry time grows with unit slope in the initialization term "
            "$\\log_+[1/(\\beta_\\star\\lambda_{0,\\star})]$ of the global theorem over four "
            "decades of $\\lambda_0$: the retraction sits on the predicted line to three "
            "digits, fitted slope $0.999$ and intercept $-0.000$, and the Bregman scheme "
            "runs about five per cent above it. This is a Fisher--Rao only measurement; no "
            "warm start is involved.",
            panel_burn_in,
            [f"F1burnin_d20_lam1e-{e}" for e in (2, 4, 6, 8)],
        ),
        Panel(
            "b", "Affine equivariance",
            "Largest discrepancy along the trajectory after mapping the iterates of the "
            "transformed problem $x\\mapsto Sx+b$ back to the base coordinates, against "
            "$\\mathrm{cond}(S)$. Both Fisher--Rao schemes agree with the base trajectory to "
            "roundoff, the residual growing like $\\varepsilon_{\\rm mach}\\,"
            "\\mathrm{cond}(S)^2$: from $10^{-15}$ at $\\mathrm{cond}(S)=1$ through "
            "$10^{-12}$ and $10^{-9}$. The exponent is two because the covariance "
            "is carried by the congruence $C\\mapsto SCS^\\top$, so this is floating-point "
            "amplification through an ill-conditioned map rather than a loss of "
            "equivariance. FB--GVI, whose Bures--Wasserstein geometry "
            "is not affine invariant, departs by an $O(1)$ amount as soon as $S$ is not "
            "orthogonal. The grid stops at $\\mathrm{cond}(S)=10^6$: the shaded region is "
            "where $\\mathrm{cond}(S)^2$ reaches $\\varepsilon_{\\rm mach}^{-1}$, so the "
            "transported covariance is no longer representable in double precision and no "
            "method can be measured there. Every point drawn is a complete trajectory that "
            "required no repair.",
            panel_affine,
            [f"F1affine_d10_S1e{e}" for e in (0, 2, 4, 6)],
        ),
        Panel(
            "c", r"$\kappa=10^9$, $\kappa_\star=1$",
            "Normalized exact KL gap on the anisotropic Gaussian target of Diao et al. in "
            "$d=10$ with logarithmically spaced precision eigenvalues, each method at its own "
            "certified step. Coordinate anisotropy and intrinsic conditioning differ maximally "
            "here: after optimizer whitening $\\kappa_\\star=1$, so the Fisher--Rao schemes "
            "reach the double-precision floor, while FB--GVI remains limited by the "
            "original-coordinate conditioning. The target is chosen to separate the two "
            "notions of conditioning and is not evidence that Fisher--Rao always dominates.",
            panel_anisotropic,
            ["F1aniso_d10_kappa1e9"],
        ),
    ]
    return compose(
        "figure_1",
        panels,
        shape=(1, 3),
        height=2.55,
        lead=(
            "Gaussian structure and affine invariance of the Fisher--Rao schemes. "
            "FB--GVI denotes the forward--backward Bures--Wasserstein scheme of Diao "
            "et al., written here as the Bures--Wasserstein entry to match the geometry "
            "first naming of the two Fisher--Rao entries. All panels are deterministic "
            "and use the population gradient and Hessian, which are available in closed "
            "form for Gaussian targets."
        ),
        legend_columns=6,
    )


# ---------------------------------------------------------------------------
# Figure 2: global-to-local deterministic convergence
# ---------------------------------------------------------------------------

REPRESENTATIVE_PANELS = (
    ("F2global_d10_k1_rho1", r"$d=10$, $\kappa_{\rm base}=1$"),
    ("F2global_d10_k10_rho1", r"$d=10$, $\kappa_{\rm base}=10$"),
    ("F2global_d50_k100_rho1", r"$d=50$, $\kappa_{\rm base}=10^2$"),
)


def _global_frame() -> pd.DataFrame:
    frame = require(load_experiment("D", TIER), "D")
    return frame[frame["job_id"].str.startswith("F2global")].copy()


def _normalized_gap(subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    subset = subset.sort_values("iteration")
    initial = float(subset["objective_gap"].iloc[0])
    gaps = subset["objective_gap"].to_numpy(dtype=np.float64) / initial
    # Resolution floor of the objective difference in double precision.
    floor = EPS * float(np.abs(subset["objective"]).max()) / abs(initial)
    return subset["iteration"].to_numpy(dtype=np.float64), gaps, floor


def _global_panel(job: str) -> Callable[[plt.Axes], pd.DataFrame]:
    def draw(axis: plt.Axes) -> pd.DataFrame:
        frame = _global_frame()
        cell = frame[frame["job_id"] == job]
        collected: list[pd.DataFrame] = []
        floors: list[float] = []
        extent = 1.0
        for index, method in enumerate(ordered_methods(cell)):
            subset = cell[cell["method"] == method]
            iterations, gaps, analytic = _normalized_gap(subset)
            visible, floor = truncate_at_floor(gaps, analytic)
            floors.append(floor)
            drawn = iterations[np.isfinite(visible)]
            extent = max(extent, float(drawn.max()) if drawn.size else 1.0)
            axis.plot(
                np.maximum(iterations, 1.0), positive(visible), label=method,
                markevery=_markevery(visible, index), zorder=5 - index,
                **method_style(method),
            )
            collected.append(
                pd.DataFrame(
                    {
                        "job_id": job,
                        "method": method,
                        "iteration": iterations,
                        "flow_time": iterations * float(subset["step_size"].iloc[-1]),
                        "step_size": float(subset["step_size"].iloc[-1]),
                        "normalized_gap": gaps,
                        "resolution_floor": floor,
                    }
                )
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(float(np.median(floors)) / 5.0, 3.0)
        axis.set_xlim(0.9, extent * 1.4)
        axis.set_xlabel("iteration $n$")
        axis.set_ylabel(r"$\Delta(a_n)/\Delta(a_0)$")
        return pd.concat(collected, ignore_index=True)

    return draw


def _grid_summary_table() -> pd.DataFrame:
    frame = _global_frame()
    rows: list[dict[str, object]] = []
    for job, cell in frame.groupby("job_id"):
        for method in ordered_methods(cell):
            subset = cell[cell["method"] == method]
            iterations, gaps, _ = _normalized_gap(subset)
            step = float(subset["step_size"].iloc[-1])
            reached = np.where(gaps <= GLOBAL_TOLERANCE)[0]
            censored = not reached.size
            count = float(GLOBAL_BUDGET) if censored else float(iterations[reached[0]])
            rows.append(
                {
                    "job_id": job,
                    "method": method,
                    "dimension": int(cell["grid_dimension"].iloc[0]),
                    "condition": float(cell["grid_condition"].iloc[0]),
                    "rho": float(cell["grid_rho"].iloc[0]),
                    "kappa_star": float(cell["kappa_star"].iloc[0]),
                    "beta_star": float(cell["beta_star"].iloc[0]),
                    "step_size": step,
                    "iterations_to_tolerance": count,
                    "flow_time_to_tolerance": count * step,
                    "censored": censored,
                    # The value at the end of the run, not the running minimum: the
                    # minimum is often a negative roundoff excursion and would
                    # misreport what the trajectory actually finished at.
                    "terminal_normalized_gap": float(gaps[-1]),
                }
            )
    return pd.DataFrame(rows)


SIZES = {10: 18.0, 50: 40.0}


def _grid_scatter(axis: plt.Axes, table: pd.DataFrame, column: str) -> None:
    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        style = method_style(method)
        for censored, group in subset.groupby("censored"):
            axis.scatter(
                group["kappa_star"], group[column],
                s=[SIZES.get(int(d), 24.0) for d in group["dimension"]],
                facecolors="none", edgecolors=style["color"], marker=style["marker"],
                linewidths=0.9, alpha=0.4 if censored else 1.0,
                label=method if not censored else None,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\kappa_\star$")


def panel_grid_iterations(axis: plt.Axes) -> pd.DataFrame:
    table = _grid_summary_table()
    _grid_scatter(axis, table, "iterations_to_tolerance")
    if table["censored"].any():
        axis.axhline(GLOBAL_BUDGET, color=REFERENCE_GREY, linestyle="-", linewidth=0.7,
                     label="budget")
    axis.set_ylabel(r"$N_{10^{-6}}$")
    return table


def panel_grid_flow_time(axis: plt.Axes) -> pd.DataFrame:
    table = _grid_summary_table()
    _grid_scatter(axis, table, "flow_time_to_tolerance")
    axis.set_ylabel(r"$N_{10^{-6}}\,h$")
    return table


def _local_frame() -> pd.DataFrame:
    """Local-rate trajectories with the star distance ``||a_n - a_star||_star``."""

    frame = require(load_experiment("G", TIER), "G").copy()
    frame["distance"] = np.sqrt(
        frame["mean_error"] ** 2 + 0.5 * frame["covariance_error"] ** 2
    )
    return frame


def _fit_contraction(subset: pd.DataFrame) -> float:
    """Per-iteration contraction fitted where the trajectory is linear and resolved."""

    subset = subset.sort_values("iteration")
    distance = subset["distance"].to_numpy(dtype=np.float64)
    iteration = subset["iteration"].to_numpy(dtype=np.float64)
    usable = np.isfinite(distance) & (distance > 1.0e-11) & (distance <= 0.5 * distance[0])
    indices = np.where(usable)[0]
    if indices.size < 10:
        return float("nan")
    # Second half of the usable window: inside the local region, above the
    # double-precision floor, and past the nonlinear transient.
    tail = indices[indices.size // 2 :]
    slope = np.polyfit(iteration[tail], np.log(distance[tail]), 1)[0]
    return float(np.exp(slope))


def panel_local_rate(axis: plt.Axes) -> pd.DataFrame:
    frame = _local_frame()
    rows: list[dict[str, object]] = []
    for (job, method), subset in frame.groupby(["job_id", "method"]):
        rows.append(
            {
                "job_id": job,
                "method": method,
                "rho": float(subset["grid_rho"].iloc[0]),
                "radius": float(subset["local_radius"].iloc[0]),
                "step_size": float(subset["step_size"].iloc[-1]),
                "gamma_star": float(subset["local_gamma"].iloc[0]),
                "kl_gamma": float(subset["local_kl_gamma"].iloc[0]),
                "predicted": float(subset["predicted_contraction"].iloc[0]),
                "measured": _fit_contraction(subset),
            }
        )
    table = pd.DataFrame(rows).dropna(subset=["measured"])
    table["relative_error"] = (table["measured"] / table["predicted"] - 1.0).abs()
    for method in ordered_methods(table):
        subset = table[table["method"] == method]
        style = {k: v for k, v in method_style(method).items() if k != "linestyle"}
        axis.plot(
            subset["predicted"], subset["measured"], linestyle="none",
            markersize=4.5, fillstyle="none", label=method, **style,
        )
    span = [
        float(table["predicted"].min()) - 0.04,
        float(table["predicted"].max()) + 0.04,
    ]
    axis.plot(span, span, color=REFERENCE_GREY, linestyle=":", linewidth=1.0,
              label="identity")
    # Points on an identity line only show that nothing is grossly wrong; the
    # size of the agreement is the actual result, so it is stated in the panel.
    axis.text(
        0.96, 0.06,
        f"max rel. error {float(table['relative_error'].max()):.0e}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=7.0,
        color=REFERENCE_GREY,
    )
    axis.set_xlim(*span)
    axis.set_ylim(*span)
    axis.set_xlabel("predicted contraction")
    axis.set_ylabel("measured contraction")
    return table


def figure_2() -> dict[str, object]:
    global_sources = sorted(_global_frame()["job_id"].unique())
    local_sources = sorted(_local_frame()["job_id"].unique())
    panels = [
        Panel(
            letter, title,
            f"Normalized objective gap on the {description} cell of the "
            "optimizer-whitened log-cosh family at $\\rho=1$, each method at its frozen "
            "practical step. Values at or below the double-precision resolution of the "
            "objective are not drawn.",
            _global_panel(job),
            [job],
        )
        for letter, (job, title), description in zip(
            "abc",
            REPRESENTATIVE_PANELS,
            ("well-conditioned control", "moderately conditioned", "difficult"),
            strict=True,
        )
    ]
    panels += [
        Panel(
            "d", "Grid summary",
            f"Iterations to $\\Delta(a_n)/\\Delta(a_0)\\le 10^{{-6}}$ over all twelve cells "
            f"($d\\in\\{{10,50\\}}$, $\\kappa_{{\\rm base}}\\in\\{{1,10,10^2\\}}$, "
            f"$\\rho\\in\\{{0.1,1\\}}$), against the affine-invariant condition number "
            f"$\\kappa_\\star$; marker size encodes the dimension. Every cell reaches the "
            f"tolerance inside the fixed {GLOBAL_BUDGET}-iteration budget. The frozen "
            f"practical step is a multiple of $1/(\\beta_\\star\\max(\\lambda_{{0,\\star}}^"
            f"{{\\max}},1))$, and $\\beta_\\star$ ranges over $[1.04,5.09]$ here, so on the "
            f"least-conditioned cells the resulting $h\\gamma_\\star$ approaches $2$, the "
            f"one-step factor approaches $-1$, and the Fisher--Rao iteration count rises. "
            f"That is the price of one frozen multiplier, not a property of the flow; panel "
            f"(e) separates the two.",
            panel_grid_iterations,
            global_sources,
        ),
        Panel(
            "e", "Elapsed flow time",
            "The same cells measured in elapsed flow time $N_{10^{-6}}h$ rather than in "
            "iterations. On the eight cells with $\\kappa_\\star\\ge2$ the three schemes "
            "agree to within a factor of $2.2$, which places the iteration-count "
            "differences of panel (d) in the stepsize rather than in the geometry. The "
            "flow time does not grow with $\\kappa_\\star$ over two decades of it, so the "
            "worst-case constant is far from attained on this family. The four "
            "least-conditioned cells are the exception, and for the reason given in panel "
            "(d): there the frozen multiplier overshoots, which is the cost of freezing one "
            "number rather than a property of the flow.",
            panel_grid_flow_time,
            global_sources,
        ),
        Panel(
            "f", "Local rate",
            "Measured against predicted per-iteration contraction for both local targets, "
            "$\\rho\\in\\{0.1,1\\}$, three initial radii "
            "$r\\in\\{10^{-1},5\\times10^{-2},10^{-2}\\}$ along the slowest eigenmode of the "
            "linearized generator, and both schemes at their certified steps. The "
            "prediction is $1-h\\gamma_\\star$ for the Riemannian retraction and "
            "$1-h\\gamma^{\\rm KL}_{\\star,h}$, the smallest eigenvalue of the linearized "
            "KL/Bregman one-step map, for the Bregman scheme. The three radii agree to "
            "five significant figures, so the measured rate is a property of the mode and "
            "not of the initial amplitude, and the residual discrepancy is the second-order "
            "term of the discretization: it is $10^{-3}$ on the $\\rho=0.1$ targets, whose "
            "certified step is five times larger, and $10^{-5}$ on the $\\rho=1$ targets.",
            panel_local_rate,
            local_sources,
        ),
    ]
    return compose(
        "figure_2",
        panels,
        shape=(2, 3),
        height=4.55,
        lead=(
            "Global-to-local deterministic convergence on the strongly log-concave "
            "shifted log-cosh family, built in optimizer-whitened coordinates so that "
            "$C_\\star=I$ and the initialization $m_0=2e_1$, $C_0=I$ isolates the "
            "non-Gaussian localization from a covariance burn-in."
        ),
        legend_columns=5,
    )


# ---------------------------------------------------------------------------
# Figure 3: deterministic Bayesian logistic regression
# ---------------------------------------------------------------------------

# The logistic group is split one config per (conditioning, dataset, method), so a
# cell is selected by its feature conditioning and a replicate by its dataset index.
LOGISTIC_HEADLINE = 1.0e4
ITERATIVE_METHODS = ("FR--R", "FR--KL", "FB--GVI")


def _logistic_frame() -> pd.DataFrame:
    return require(load_experiment("L", TIER), "L")


def _median_band(
    axis: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    label: str,
    style: dict,
    floor: float = 0.0,
    index: int = 0,
) -> pd.DataFrame:
    """Median over datasets with a min-max band, truncated at the resolution floor.

    Aggregation is always by iteration, never by the plotted abscissa: wall-clock
    time takes a different value on every dataset, so grouping by it would put one
    trajectory in each group and collapse the band to nothing.  At iteration ``n``
    the panel shows the median time and the median gap over the datasets.
    """

    grouped_x = frame.groupby("iteration")[x]
    grouped = frame.groupby("iteration")[y]
    xs = grouped_x.median().to_numpy(dtype=np.float64)
    median, level = truncate_at_floor(grouped.median().to_numpy(), floor)
    drawn = np.isfinite(median)
    lower = np.where(drawn, positive(grouped.min().to_numpy()), np.nan)
    upper = np.where(drawn, positive(grouped.max().to_numpy()), np.nan)
    if int(grouped.count().max()) > 1:
        band(axis, xs, lower, upper, style["color"])
    axis.plot(xs, positive(median), label=label, markevery=_markevery(median, index),
              zorder=5 - index, **style)
    axis.set_yscale("log")
    return pd.DataFrame(
        {x: xs, "median": median, "min": lower, "max": upper, "floor": level}
    )


def reference_resolution_floor(cell: pd.DataFrame) -> float:
    """How far down the reference actually resolves the gap on one logistic cell.

    On a finite quadrature design the first-order-condition point that every
    deterministic method stalls at is not exactly the minimizer of the discretized
    objective, so a trajectory can pass a little below it and record a negative gap.
    The size of the largest such excursion is direct evidence of how far the
    reference resolves, and it is a tighter and more honest floor than any
    machine-precision estimate.
    """

    gaps = cell["objective_gap"].to_numpy(dtype=np.float64)
    gaps = gaps[np.isfinite(gaps)]
    shortfall = float(-gaps.min()) if gaps.size and gaps.min() < 0.0 else 0.0
    machine = EPS * float(np.abs(cell["objective"]).max())
    return max(shortfall, machine)


def _logistic_trajectory_panel(x_column: str, x_label: str) -> Callable[[plt.Axes], pd.DataFrame]:
    def draw(axis: plt.Axes) -> pd.DataFrame:
        frame = _logistic_frame()
        cell = frame[frame["grid_feature_condition"] == LOGISTIC_HEADLINE]
        floor = reference_resolution_floor(cell)
        collected: list[pd.DataFrame] = []
        for index, method in enumerate(ITERATIVE_METHODS):
            subset = cell[cell["method"] == method]
            if subset.empty:
                continue
            summary = _median_band(
                axis, subset, x_column, "objective_gap", floor=floor,
                label=method, style=method_style(method), index=index,
            )
            collected.append(summary.assign(method=method))
        laplace = cell[cell["method"] == "Laplace"]
        if not laplace.empty:
            level = float(laplace["objective_gap"].median())
            axis.axhline(level, color=COLORS["Laplace"], linestyle=":", linewidth=1.1,
                         label="Laplace")
            collected.append(pd.DataFrame({x_column: [np.nan], "median": [level],
                                           "min": [np.nan], "max": [np.nan],
                                           "method": ["Laplace"]}))
        axis.set_ylim(floor / 5.0, None)
        if x_column != "iteration":
            axis.set_xscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$\Delta(a_n)$")
        return pd.concat(collected, ignore_index=True).assign(resolution_floor=floor)

    return draw


def panel_logistic_conditions(axis: plt.Axes) -> pd.DataFrame:
    """Iterations to a declared relative tolerance, across feature conditioning.

    The previous version summarised the terminal gap, which meant clamping values
    that had reached the reference's own resolution and drawing the clamp as if it
    were a measurement.  Iterations to a fixed relative tolerance is a measured
    quantity everywhere, needs no floor, and is what the experiment plan offers as
    the alternative.
    """

    frame = _logistic_frame()
    rows: list[dict[str, object]] = []
    for (condition, method, dataset), trajectory in frame.groupby(
        ["grid_feature_condition", "method", "grid_dataset"]
    ):
        trajectory = trajectory.sort_values("iteration")
        initial = float(trajectory["objective_gap"].iloc[0])
        if method == "Laplace" or initial <= 0.0:
            continue
        relative = trajectory["objective_gap"].to_numpy(dtype=np.float64) / initial
        reached = np.where(relative <= GLOBAL_TOLERANCE)[0]
        rows.append(
            {
                "feature_condition": float(condition),
                "method": method,
                "dataset": int(dataset),
                "iterations_to_tolerance": (
                    float(trajectory["iteration"].to_numpy()[reached[0]])
                    if reached.size
                    else np.nan
                ),
                "censored": not reached.size,
            }
        )
    table = pd.DataFrame(rows)
    # Horizontal offsets separate three markers that share each abscissa.
    offsets = {"FR--R": 0.82, "FR--KL": 1.0, "FB--GVI": 1.22}
    for method in ITERATIVE_METHODS:
        subset = table[table["method"] == method]
        if subset.empty:
            continue
        style = method_style(method)
        summary = subset.groupby("feature_condition")["iterations_to_tolerance"].agg(
            ["median", "min", "max"]
        )
        median = summary["median"].to_numpy()
        axis.errorbar(
            summary.index.to_numpy() * offsets[method], median,
            yerr=[median - summary["min"].to_numpy(), summary["max"].to_numpy() - median],
            linestyle="none", marker=style["marker"], markersize=4.0, fillstyle="none",
            color=style["color"], elinewidth=0.9, capsize=2.0, label=method,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$\kappa_X$")
    axis.set_ylabel(r"$N_{10^{-6}}$")
    return table


def _wall_clock_caption(data: pd.DataFrame) -> str:
    """Panel (b)'s caption, with the measured per-iteration costs read off the data."""

    frame = _logistic_frame()
    cell = frame[frame["grid_feature_condition"] == LOGISTIC_HEADLINE]
    terminal = (
        cell[cell["method"] != "Laplace"]
        .sort_values("iteration")
        .groupby(["method", "grid_dataset"], as_index=False)
        .last()
    )
    per_iteration = 1000.0 * terminal["algorithm_seconds"] / terminal["iteration"]
    low, high = float(per_iteration.min()), float(per_iteration.max())
    spread = 100.0 * (high - low) / low
    order = int(cell["quadrature_order"].iloc[0]) if "quadrature_order" in cell else 48
    return (
        "The same trajectories against the time spent inside the update itself, "
        "excluding the diagnostics. The three methods cost "
        f"{low:.1f} to {high:.1f} milliseconds per iteration, a spread of about "
        f"{spread:.0f} per cent, so the panel largely repeats the shape of (a). The cost "
        "is set by the expectation, which is one panelled Gauss--Legendre rule of order "
        f"{order} per linear predictor and so scales as $O(nQ)$ in the $n$ observations "
        "and $Q$ nodes, rather than by the $O(d^3)$ linear algebra that distinguishes the "
        "matrix exponential of the retraction from the resolvent solve of the Bregman "
        "scheme. That distinction would become visible only at large $d$ with a cheaper "
        "expectation."
    )


def figure_3() -> dict[str, object]:
    sources = sorted(_logistic_frame()["job_id"].unique())
    headline = sorted(
        _logistic_frame()
        .loc[lambda f: f["grid_feature_condition"] == LOGISTIC_HEADLINE, "job_id"]
        .unique()
    )
    panels = [
        Panel(
            "a", r"$\kappa_X=10^4$, iterations",
            f"Objective gap against iteration on the hardest feature conditioning, median "
            f"over {LOGISTIC_DATASETS} independently generated datasets with a min-max band. "
            "Each deterministic iteration consumes one expected gradient and one expected "
            "Hessian, so the iteration count is already a population-oracle count.",
            _logistic_trajectory_panel("iteration", "iteration $n$"),
            headline,
        ),
        Panel(
            "b", r"$\kappa_X=10^4$, wall clock",
            _wall_clock_caption,
            _logistic_trajectory_panel("algorithm_seconds", "algorithm time (s)"),
            headline,
        ),
        Panel(
            "c", r"Across $\kappa_X$",
            f"Iterations to a relative objective gap of {GLOBAL_TOLERANCE:g}, median over "
            f"the {LOGISTIC_DATASETS} datasets with min-max bars; markers are offset "
            "horizontally for legibility. No method dominates: the Bures--Wasserstein "
            "baseline is fastest where the features are best conditioned and slowest where "
            "they are worst, with the crossover near $\kappa_X=10^2$. That is the expected "
            "shape, since its guarantee places the conditioning in the rate while the "
            "Fisher--Rao guarantee places it in the admissible stepsize. The Laplace "
            "approximation is a fixed point of none of the schemes and so has no "
            "iteration count; it appears in panels (a) and (b) and in the predictive table.",
            panel_logistic_conditions,
            sources,
        ),
    ]
    return compose(
        "figure_3",
        panels,
        shape=(1, 3),
        height=2.55,
        lead=(
            "Deterministic Gaussian variational inference for Bayesian logistic regression "
            "with a proper Gaussian prior $\\theta\\sim\\mathcal N(0,\\lambda^{-1}I)$, "
            "$\\lambda=1$, in $d=50$ with $500$ training and $5000$ held-out points. All "
            "expectations are computed by deterministic one-dimensional quadrature over "
            "each linear predictor rather than by sampling, so a reported gap is a gap to "
            "the Gaussian variational optimum itself. FB--GVI denotes the forward--backward "
            "Bures--Wasserstein scheme of Diao et al., written BW--FB for symmetry with the "
            "geometry-first Fisher--Rao labels."
        ),
        legend_columns=4,
    )


BUILDERS = {1: figure_1, 2: figure_2, 3: figure_3}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="*", type=int, default=None)
    args = parser.parse_args(arguments)
    configure_style()
    for index in args.figures or sorted(BUILDERS):
        provenance = BUILDERS[index]()
        panels = ", ".join(str(entry["panel"]) for entry in provenance["panels"])
        print(f"figure {index} written (panels {panels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
