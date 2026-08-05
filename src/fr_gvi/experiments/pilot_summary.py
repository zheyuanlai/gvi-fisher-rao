from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fr_gvi.plotting.aggregation import aggregate_series
from fr_gvi.plotting.figures import ROOT, group_trajectories
from fr_gvi.plotting.main_figures import _observed_rate, _terminal_floors

RAW_CORE = ROOT / "results" / "raw" / "core"
MANIFESTS = ROOT / "results" / "manifests"
OUTPUT = ROOT / "reports" / "PILOT_RESULTS.json"


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan


def _rows(experiment: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for path in sorted((RAW_CORE / experiment).rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            output.extend(csv.DictReader(handle))
    return output


def _terminal(rows: list[dict[str, str]], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_job, method, _seed, _variant), trajectory in group_trajectories(rows).items():
        ordered = sorted(trajectory, key=lambda row: _number(row, "iteration"))
        for metric in metrics:
            by_method[method][metric].append(_number(ordered[-1], metric))
    return {
        method: {
            metric: float(np.nanmedian(values))
            for metric, values in method_metrics.items()
        }
        for method, method_metrics in by_method.items()
    }


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize() -> dict[str, Any]:
    a_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in sorted((MANIFESTS / "core").glob("A_burnin_*.json")):
        record = _manifest(path)
        config = record["config"]
        method = record["method_specification"]["name"]
        scale = float(config["target"]["initial_covariance_scale"])
        beta = float(config["target"]["condition"])
        step = float(record["method_specification"]["step_size"])
        a_records[method].append(
            {
                "lambda_0": scale,
                "log_inverse_beta_lambda": float(np.log(1.0 / (beta * scale))),
                "burn_in_iteration": int(record["burn_in_iteration"]),
                "burn_in_time": float(record["burn_in_iteration"]) * step,
            }
        )

    b_rows = _rows("B")
    affine: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in b_rows}):
        selected = [row for row in b_rows if row["method"] == method]
        affine[method] = {
            "max_mean_discrepancy": float(np.nanmax([_number(row, "equivariance_error_mean") for row in selected])),
            "max_covariance_discrepancy": float(
                np.nanmax([_number(row, "equivariance_error_covariance") for row in selected])
            ),
        }

    c_rows = _rows("C")
    c_competitive = [row for row in c_rows if row.get("quadratic_rescue") != "True"]
    c_rescue = [row for row in c_rows if row.get("quadratic_rescue") == "True"]

    g_rates: dict[str, dict[str, float]] = {}
    for (_job, method, _seed, _variant), trajectory in group_trajectories(_rows("G")).items():
        rate = _observed_rate(trajectory)
        if rate is not None:
            g_rates[method] = {"predicted_2gamma": rate[0], "observed_tail_rate": rate[1]}

    h_spread: dict[str, float] = {}
    h_collections: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in _rows("H"):
        h_collections[row["method"]][_number(row, "iteration")].append(_number(row, "objective_gap"))
    for method, iterations in h_collections.items():
        h_spread[method] = float(max(np.ptp(values) for values in iterations.values()))

    i_rows = _rows("I")
    i_ratio = np.asarray([_number(row, "stl_raw_variance_ratio") for row in i_rows])

    j_floors = {
        method: [
            {"batch_size": batch, "median": median, "q10": lower, "q90": upper}
            for batch, median, lower, upper in sorted(values)
        ]
        for method, values in _terminal_floors(_rows("J")).items()
    }

    k_series = aggregate_series(group_trajectories(_rows("K")), "objective_gap")
    k_summary: dict[str, dict[str, float]] = {}
    for item in k_series:
        finite = (item.x > 0) & np.isfinite(item.median) & (item.median > 0)
        x, gap = item.x[finite], item.median[finite]
        tail = x >= np.quantile(x, 0.5)
        slope = float(np.polyfit(np.log(x[tail]), np.log(gap[tail]), deg=1)[0])
        k_summary[item.method] = {
            "tail_loglog_slope": slope,
            "terminal_N_gap": float(x[-1] * gap[-1]),
            "terminal_gap": float(gap[-1]),
        }

    core_manifests = [_manifest(path) for path in sorted((MANIFESTS / "core").glob("*.json"))]
    status_counts: dict[str, int] = defaultdict(int)
    for record in core_manifests:
        status_counts[str(record.get("status", "unknown"))] += 1
    active_counts = {"completed": 0, "skipped": 0}
    for tier in ("smoke", "core"):
        for path in sorted((ROOT / "configs" / tier).glob("*.json")):
            config = _manifest(path)
            if config.get("blocked_reason"):
                active_counts["skipped"] += 1
            elif config["experiment"] == "I":
                active_counts["completed"] += int(config.get("seeds", 1))
            else:
                active_counts["completed"] += sum(
                    int(method.get("seeds", config.get("seeds", 1)))
                    for method in config.get("methods", [])
                )

    logistic_reference = _manifest(MANIFESTS / "reference_L_logistic_d10_n100_core.json")
    return {
        "scope": "completed core pilot; full and appendix grids are not included",
        "active_smoke_plus_core_campaign": active_counts,
        "core_manifest_status": dict(status_counts),
        "A_covariance_burn_in": dict(a_records),
        "B_affine_equivariance": affine,
        "C_diao_gaussian_terminal": _terminal(c_competitive, ("objective_gap", "wall_time_seconds")),
        "C_quadratic_rescue_terminal": _terminal(c_rescue, ("objective_gap", "oracle_pairs")),
        "D_logcosh_terminal": _terminal(_rows("D"), ("objective_gap", "mean_error", "covariance_error", "wall_time_seconds")),
        "F_gaussian_local_terminal": _terminal(
            [row for row in _rows("F") if row["method"] != "FB--GVI"],
            ("objective_gap", "w2_squared"),
        ),
        "G_local_spectral_rates": g_rates,
        "H_max_across_seed_objective_spread": h_spread,
        "I_STL_raw_variance_ratio": {
            "minimum": float(np.nanmin(i_ratio)),
            "maximum": float(np.nanmax(i_ratio)),
        },
        "J_tail_floor_by_batch": j_floors,
        "K_decreasing_step": k_summary,
        "L_logistic_terminal": _terminal(
            _rows("L"),
            ("objective_gap", "predictive_nll", "classification_error", "brier", "wall_time_seconds"),
        ),
        "L_reference": {
            "fisher_rao_residual_squared": logistic_reference["fisher_rao_residual_squared"],
            "bures_wasserstein_residual_squared": logistic_reference["bures_wasserstein_residual_squared"],
            "optimizer_success": logistic_reference["metadata"].get("optimizer_success"),
        },
    }


def main() -> int:
    report = summarize()
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
