"""Tables that accompany the three manuscript figures.

Three things belong in the text but not in a panel: the stepsize protocol (what
each method was actually run at, next to what its theorem certifies), the
certification of every reference solution, and the held-out predictive quality of
the logistic-regression fits.  Each is emitted as a CSV and as a booktabs body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fr_gvi.experiments.manuscript import (
    GLOBAL_TOLERANCE,
    ITERATIVE,
    LOGISTIC_EVALUATION_ORDER,
    LOGISTIC_UPDATE_ORDER,
    SELECTED_STEPS,
)
from fr_gvi.diagnostics.core import truncate_at_floor
from fr_gvi.experiments.tables import _write
from fr_gvi.plotting.style import load_experiment

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "results" / "manifests" / "manuscript"
TIER = "manuscript"


def _manuscript_manifests() -> list[dict]:
    manifests = []
    for path in sorted(MANIFESTS.glob("*.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return manifests


def stepsize_protocol_table() -> pd.DataFrame:
    """What each method ran at, beside what its own theorem certifies."""

    payload = json.loads(SELECTED_STEPS.read_text(encoding="utf-8"))
    multipliers = payload["multipliers"]
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for manifest in _manuscript_manifests():
        config = manifest.get("config", {})
        specification = manifest.get("method_specification", {})
        name = str(specification.get("name", ""))
        job = str(config.get("id", ""))
        if name not in ITERATIVE or (job, name) in seen or job.startswith("pilot"):
            continue
        seen.add((job, name))
        certified = float(specification.get("certified_step_size", np.nan))
        step = float(specification.get("step_size", np.nan))
        scale = specification.get("step_scale")
        rows.append(
            {
                "job_id": job,
                "experiment": str(config.get("experiment", "")),
                "method": name,
                "step_size": step,
                "certified_step_size": certified,
                "step_over_certified": step / certified if certified else np.nan,
                "step_scale": float(scale) if scale is not None else np.nan,
                "protocol": (
                    f"frozen practical, eta={multipliers.get(name, np.nan):g}"
                    if scale is not None
                    else "theorem compatible"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["experiment", "job_id", "method"])


def reference_certification_table() -> pd.DataFrame:
    """Residual of every reference solution, against the smallest gap it supports.

    Two numbers matter and they measure different things.  The on-design Fisher--Rao
    residual says how exactly the reference solves the discretized problem the
    algorithms solve, and it is what bounds the accuracy of every reported gap.  The
    transfer residual, evaluated on an independent design four times larger, says how
    well that discretized problem represents the continuum one; it is a property of
    the quadrature, not of the reference.
    """

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for manifest in _manuscript_manifests():
        config = manifest.get("config", {})
        job = str(config.get("id", ""))
        reference = manifest.get("reference", {})
        if not reference or job in seen or job.startswith("pilot"):
            continue
        seen.add(job)
        path = Path(str(reference.get("path", "")))
        metadata: dict = {}
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
        on_design = float(reference.get("fisher_rao_residual_squared", np.nan))
        bures = float(reference.get("bures_wasserstein_residual_squared", np.nan))
        transfer = metadata.get("design_transfer_fisher_rao_residual_squared")
        curvature = manifest.get("curvature", {})
        alpha_star = float(curvature.get("alpha_star", np.nan))
        # The proved conversion from residual to energy gap, with the manuscript's
        # own constant.  The reference is a stationary point so its covariance
        # equals the inverse expected Hessian; the Fisher--Rao branch of the bound
        # needs a covariance floor we do not carry here, so the Bures--Wasserstein
        # branch, which needs only alpha_star, is the one applied.
        certified = (
            float(bures / (2.0 * alpha_star))
            if np.isfinite(bures) and np.isfinite(alpha_star) and alpha_star > 0.0
            else np.nan
        )
        rows.append(
            {
                "job_id": job,
                "experiment": str(config.get("experiment", "")),
                "backend": str(metadata.get("backend", "exact")),
                "alpha_star": alpha_star,
                "residual_on_design": float(np.sqrt(max(on_design, 0.0))),
                "certified_gap_bound": certified,
                "residual_transfer": (
                    float(np.sqrt(max(float(transfer), 0.0))) if transfer is not None else np.nan
                ),
                "transfer_objective_difference": float(
                    metadata.get("design_transfer_objective_difference", np.nan)
                ),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["experiment", "job_id"]).reset_index(drop=True)


def smallest_reported_gaps() -> pd.DataFrame:
    """Smallest gap each figure actually draws, per experiment.

    The raw trajectories continue past the point where the objective difference
    reaches its own resolution, and those trailing values are roundoff rather than
    measurements: the figures truncate them and so must the certification, or the
    reference would be held to a tolerance set by numerical noise.  The truncation
    is the one the panels use, applied per trajectory.
    """

    rows: list[dict[str, object]] = []
    for experiment in ("C", "D", "G", "L"):
        frame = load_experiment(experiment, TIER)
        if frame.empty:
            continue
        frame = frame[~frame["job_id"].astype(str).str.startswith("pilot")]
        column = "exact_gaussian_kl" if experiment == "C" else "objective_gap"
        drawn: list[float] = []
        for _, trajectory in frame.groupby(["job_id", "method", "seed"]):
            values = trajectory.sort_values("iteration")[column].to_numpy(dtype=np.float64)
            visible, _ = truncate_at_floor(values)
            visible = visible[np.isfinite(visible) & (visible > 0.0)]
            if visible.size:
                drawn.append(float(visible.min()))
        rows.append(
            {
                "experiment": experiment,
                "trajectories": int(frame.groupby(["job_id", "method", "seed"]).ngroups),
                "smallest_plotted_gap": float(min(drawn)) if drawn else np.nan,
            }
        )
    return pd.DataFrame(rows)


def predictive_table() -> pd.DataFrame:
    """Held-out predictive quality at termination on the logistic problem."""

    frame = load_experiment("L", TIER)
    if frame.empty:
        return pd.DataFrame()
    terminal = (
        frame.sort_values("iteration")
        .groupby(["job_id", "method", "seed"], as_index=False)
        .last()
    )
    grouped = terminal.groupby(["grid_feature_condition", "method"])
    rows: list[dict[str, object]] = []
    for (condition, method), subset in grouped:
        rows.append(
            {
                "feature_condition": float(condition),
                "method": method,
                "datasets": int(len(subset)),
                "objective_gap_median": float(subset["objective_gap"].median()),
                "gradient_norm_median": float(
                    np.sqrt(subset["fisher_rao_residual_squared"]).median()
                ),
                "predictive_nll_median": float(subset["predictive_nll"].median()),
                "predictive_nll_min": float(subset["predictive_nll"].min()),
                "predictive_nll_max": float(subset["predictive_nll"].max()),
                "classification_error_median": float(subset["classification_error"].median()),
            }
        )
    order = {name: index for index, name in enumerate([*ITERATIVE, "Laplace"])}
    table = pd.DataFrame(rows)
    table["order"] = table["method"].map(order).fillna(99)
    return table.sort_values(["feature_condition", "order"]).drop(columns="order")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(arguments)
    print("manuscript tables:")
    _write(
        stepsize_protocol_table(),
        "manuscript_stepsizes",
        "Stepsize actually used by each method on each manuscript cell, beside the "
        "step its own convergence theorem certifies.",
        {"step_size": ".4g", "certified_step_size": ".4g", "step_over_certified": ".3g"},
    )
    _write(
        reference_certification_table(),
        "manuscript_reference_certification",
        "Certification of every reference solution used by the manuscript figures. "
        f"The logistic cells use exact one-dimensional quadrature, order "
        f"{LOGISTIC_UPDATE_ORDER} for the updates and {LOGISTIC_EVALUATION_ORDER} for the "
        "objective evaluation and the reference solve, so there is no sampling design and "
        "no transfer error. The certified bound is the manuscript's Gaussian variational "
        "PL inequality applied to the reference's own residual.",
        {
            "residual_on_design": ".3e",
            "residual_transfer": ".3e",
            "transfer_objective_difference": ".3e",
        },
    )
    _write(
        smallest_reported_gaps(),
        "manuscript_reported_gaps",
        "Smallest objective gap each experiment actually draws, after the same "
        "resolution truncation the panels use, for comparison with the certified "
        "suboptimality of each reference.",
        {"smallest_plotted_gap": ".3e"},
    )
    predictive = predictive_table()
    if not predictive.empty:
        _write(
            predictive,
            "manuscript_predictive",
            "Held-out predictive negative log-likelihood at termination on the "
            "Bayesian logistic-regression problem, median over five datasets, together "
            f"with the terminal objective gap and Fisher--Rao gradient norm. The "
            f"iteration budget is fixed and the tolerance reference is {GLOBAL_TOLERANCE:g}.",
            {
                "objective_gap_median": ".3e",
                "gradient_norm_median": ".3e",
                "predictive_nll_median": ".4f",
                "predictive_nll_min": ".4f",
                "predictive_nll_max": ".4f",
                "classification_error_median": ".4f",
            },
        )
    else:
        print("  manuscript_predictive: skipped, no logistic results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
