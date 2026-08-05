from __future__ import annotations

import csv
import math
import json
from pathlib import Path

from fr_gvi.plotting.figures import PROCESSED, ROOT

FIGURE_EXPERIMENTS = {
    1: ("A", "B", "C"),
    2: ("D",),
    3: ("F", "G"),
    4: ("H", "I", "J", "K"),
    5: ("L",),
}


def _burn_in_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    manifests = ROOT / "results" / "manifests" / "core"
    for path in sorted(manifests.glob("A_burnin_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("burn_in_iteration") is None:
            continue
        config = record["config"]
        method = record["method_specification"]
        scale = float(config["target"]["initial_covariance_scale"])
        beta = float(config["target"]["condition"])
        step = float(method["step_size"])
        records.append(
            {
                "record_type": "derived_burn_in",
                "source_experiment": "A",
                "tier": "core",
                "experiment": "A",
                "job_id": str(config["id"]),
                "method": str(method["name"]),
                "initial_covariance_scale": repr(scale),
                "target_beta": repr(beta),
                "log_inverse_beta_lambda": repr(float(math.log(1.0 / (beta * scale)))),
                "burn_in_iteration": str(record["burn_in_iteration"]),
                "burn_in_time": repr(float(record["burn_in_iteration"]) * step),
                "source_manifest": str(path.relative_to(ROOT)),
            }
        )
    return records


def write_main_processed(by_experiment: dict[str, list[dict[str, str]]]) -> list[Path]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for number, experiments in FIGURE_EXPERIMENTS.items():
        rows: list[dict[str, str]] = []
        for experiment in experiments:
            for source in by_experiment.get(experiment, []):
                if source.get("tier") != "core":
                    continue
                row = dict(source)
                row["record_type"] = "trajectory_input"
                row["source_experiment"] = experiment
                rows.append(row)
        if number == 1:
            rows.extend(_burn_in_records())
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        output = PROCESSED / f"main_figure_{number}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        outputs.append(output)
    return outputs

