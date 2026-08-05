"""Manuscript tables built from the full-tier grids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fr_gvi.plotting.style import load_experiment

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "results" / "tables"


def _terminal(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("job_id", "method", "variant", "seed") if k in frame.columns]
    return frame.sort_values("iteration").groupby(keys, as_index=False).last()


def _write(frame: pd.DataFrame, name: str, caption: str, formats: dict[str, str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    formatted = frame.copy()
    for column, spec in formats.items():
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value, spec=spec: format(float(value), spec)
                if isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value))
                else "--"
            )
    # A plain booktabs body, written directly so the package does not depend on
    # jinja2 through pandas' styler path.
    columns = list(formatted.columns)
    lines = [
        f"% {caption}",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(column.replace("_", r"\_") for column in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in formatted.iterrows():
        lines.append(
            # Method names deliberately keep their en-dash form, matching the
            # manuscript's FR--R / FB--GVI nomenclature.
            " & ".join(str(value).replace("_", r"\_") for value in row) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {name}: {len(frame)} rows")


def stepsize_table() -> None:
    """Largest stable step and best gap at a fixed budget, per method and cell."""

    records = []
    for experiment in ("C", "D", "L"):
        frame = load_experiment(experiment)
        if frame.empty:
            continue
        frame = frame[frame["method"] != "Laplace"]
        if experiment == "L":
            # Superseded first-pass stochastic arms; see the Lstoch_* cells.
            frame = frame[
                ~(
                    frame["job_id"].str.startswith("L_logistic")
                    & ~frame["method"].isin(["FR--R", "FR--KL", "FB--GVI"])
                )
            ]
        if experiment == "C":
            frame = frame[~frame["variant"].str.contains("rescue", na=False)]
        initial = frame[frame["iteration"] == 0].groupby(["job_id", "method"])[
            "objective_gap"
        ].median()
        # Diao et al., Corollary D.2, certifies eta <= 1/beta only when the
        # initialization also satisfies beta^{-1} I <= Sigma_0.  All methods share
        # one initialization here, so that hypothesis is recorded rather than
        # engineered away.
        start_covariance = frame[frame["iteration"] == 0].groupby("job_id")[
            "covariance_min_eigenvalue"
        ].min()
        beta_by_job = frame.groupby("job_id")["beta"].first()
        terminal = _terminal(frame)
        for (job, method), subset in terminal.groupby(["job_id", "method"]):
            start = float(initial.get((job, method), np.inf))
            progressed = subset[subset["objective_gap"] < start]
            best = subset.loc[subset["objective_gap"].idxmin()]
            multiple = float(best["normalized_step_size"])
            records.append(
                {
                    "experiment": experiment,
                    "job_id": job,
                    "method": method,
                    "kappa_star": float(subset["kappa_star"].iloc[0]),
                    "certified_step": float(best["step_size"]) / multiple
                    if multiple > 0
                    else np.nan,
                    "largest_stable_multiple": float(progressed["normalized_step_size"].max())
                    if not progressed.empty
                    else np.nan,
                    "best_multiple": multiple,
                    "best_terminal_gap": float(best["objective_gap"]),
                    "oracle_pairs": float(best["oracle_pairs"]),
                    "wall_time_seconds": float(best["wall_time_seconds"]),
                    "diao_initialization_satisfied": bool(
                        float(start_covariance.get(job, 0.0))
                        >= 1.0 / float(beta_by_job.get(job, np.inf))
                    )
                    if method in {"FB--GVI", "S--FB--GVI"}
                    else "",
                }
            )
    if not records:
        return
    _write(
        pd.DataFrame(records).sort_values(["experiment", "job_id", "method"]),
        "stepsize_summary",
        "Per-cell stepsize summary: each method's certified step, the largest multiple of it "
        "that still makes progress, and the best terminal gap at a fixed oracle budget.",
        {
            "kappa_star": ".3g",
            "certified_step": ".3e",
            "largest_stable_multiple": ".3g",
            "best_multiple": ".3g",
            "best_terminal_gap": ".3e",
            "oracle_pairs": ".0f",
            "wall_time_seconds": ".3f",
        },
    )


def headline_table() -> None:
    """The headline verification numbers quoted in the text."""

    rows = []

    burn = load_experiment("A")
    if not burn.empty:
        ratios = []
        for (_, _, _), trajectory in burn.groupby(["job_id", "method", "seed"]):
            trajectory = trajectory.sort_values("iteration")
            beta_star = float(trajectory["beta_star"].iloc[0])
            entered = trajectory[
                trajectory["relative_covariance_min_eigenvalue"] >= 1.0 / (2.0 * beta_star)
            ]
            if entered.empty:
                continue
            step = float(trajectory["step_size"].iloc[-1])
            predicted = float(
                np.log(1.0 / (beta_star * float(trajectory["lambda_0_star"].iloc[0])))
            )
            ratios.append(int(entered["iteration"].iloc[0]) * step / predicted)
        if ratios:
            rows.append(
                {
                    "quantity": "burn-in time / predicted log envelope (median)",
                    "value": float(np.median(ratios)),
                    "worst_case": float(np.max(np.abs(np.asarray(ratios) - 1.0))),
                    "runs": len(ratios),
                }
            )

    affine = load_experiment("B")
    if not affine.empty:
        for method in ("FR--R", "FR--KL", "FB--GVI"):
            subset = affine[affine["method"] == method]
            if subset.empty:
                continue
            rows.append(
                {
                    "quantity": f"max affine equivariance error, {method}",
                    "value": float(subset["equivariance_error_covariance"].max()),
                    "worst_case": np.nan,
                    "runs": int(subset["job_id"].nunique()),
                }
            )

    cancellation = load_experiment("H")
    if not cancellation.empty:
        spread = (
            cancellation.groupby(["job_id", "method", "iteration"])["objective"]
            .agg(lambda values: float(values.max() - values.min()))
            .reset_index(name="spread")
        )
        for method, subset in spread.groupby("method"):
            rows.append(
                {
                    "quantity": f"max across-seed objective spread, {method}",
                    "value": float(subset["spread"].max()),
                    "worst_case": np.nan,
                    "runs": int(subset["job_id"].nunique()),
                }
            )

    metrics = load_experiment("M")
    if not metrics.empty:
        finest = metrics[metrics["step_size"] == metrics["step_size"].min()]
        summary = finest.groupby(["omega", "tau", "dimension"]).first().reset_index()
        for label, fitted, predicted in (
            ("traceless", "fitted_traceless_rate", "predicted_traceless_rate"),
            ("trace", "fitted_trace_rate", "predicted_trace_rate"),
        ):
            error = (summary[fitted] / summary[predicted] - 1.0).abs()
            rows.append(
                {
                    "quantity": f"max relative rate error, {label} mode (finest step)",
                    "value": float(error.max()),
                    "worst_case": np.nan,
                    "runs": int(len(summary)),
                }
            )

    if not rows:
        return
    _write(
        pd.DataFrame(rows),
        "headline_summary",
        "Headline verification numbers quoted in the text.",
        {"value": ".3e", "worst_case": ".3e", "runs": ".0f"},
    )


def reference_quality_table() -> None:
    """Certification of every reference solution used to compute objective gaps."""

    records = []
    for path in sorted((ROOT / "results" / "manifests").glob("reference_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Superseded smoke- and core-tier references linger on disk; the table
        # certifies the full and appendix tiers that the figures are drawn from.
        job = str(payload.get("job_id", ""))
        if job.endswith("_core") or job.endswith("_smoke") or job.endswith("_blocked"):
            continue
        metadata = payload.get("metadata", {})
        records.append(
            {
                "job_id": payload.get("job_id", ""),
                "backend": metadata.get("backend", ""),
                "fisher_rao_residual_squared": payload.get(
                    "fisher_rao_residual_squared", np.nan
                ),
                "bures_wasserstein_residual_squared": payload.get(
                    "bures_wasserstein_residual_squared", np.nan
                ),
                "quadrature_resolution_floor": metadata.get(
                    "quadrature_resolution_floor", np.nan
                ),
                "independent_design_residual_squared": metadata.get(
                    "design_transfer_fisher_rao_residual_squared", np.nan
                ),
            }
        )
    if not records:
        return
    _write(
        pd.DataFrame(records),
        "reference_quality",
        "Reference certification. Every reference is a stationary point of the problem the "
        "algorithms actually solve, certified by both the Fisher--Rao and the "
        "Bures--Wasserstein residual; for quadrature-based cells the resolution floor and "
        "the residual against an independent four-times-larger design are also reported.",
        {
            "fisher_rao_residual_squared": ".3e",
            "bures_wasserstein_residual_squared": ".3e",
            "quadrature_resolution_floor": ".3e",
            "independent_design_residual_squared": ".3e",
        },
    )


def main() -> int:
    print("writing tables:")
    stepsize_table()
    headline_table()
    reference_quality_table()
    print(f"tables written to {TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
