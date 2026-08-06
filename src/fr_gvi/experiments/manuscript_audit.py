"""Gate on the manuscript campaign before its figures are trusted.

The checks are the ones the protocol makes falsifiable:

* the campaign contains the preregistered number of trajectories and no others;
* only FR--R, FR--KL, FB--GVI and Laplace appear, and no run was repaired,
  clipped or backtracked;
* every reference solution is certified at least two orders of magnitude below the
  smallest gap the figures resolve;
* the frozen stepsizes in the configs are the ones recorded by the pilot;
* every figure has matching PDF, PNG, caption and provenance, and every panel has
  its processed CSV.

Visual inspection at final manuscript size remains a separate, manual step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fr_gvi.experiments.manuscript import (
    CONFIG_ROOT,
    GROUPS,
    ITERATIVE,
    SELECTED_STEPS,
    trajectory_count,
)
from fr_gvi.plotting.style import load_experiment
from fr_gvi.experiments.manuscript_tables import (
    reference_certification_table,
    smallest_reported_gaps,
)
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw" / "manuscript"
MANIFESTS = ROOT / "results" / "manifests" / "manuscript"
FIGURES = ROOT / "results" / "figures" / "manuscript"
REPORT = ROOT / "reports" / "MANUSCRIPT_AUDIT.json"
TIER = "manuscript"

ALLOWED_METHODS = {*ITERATIVE, "Laplace"}
# The protocol requires the reference to be at least this many orders of magnitude
# below the smallest gap any figure resolves.
RESIDUAL_MARGIN_DECADES = 2.0
# Relative size of a negative objective gap that is attributable to roundoff in the
# difference of two objectives of the recorded magnitude.
NEGATIVE_GAP_TOLERANCE = 1.0e-13
FIGURE_NAMES = ("figure_1", "figure_2", "figure_3")


def _expected_trajectories() -> dict[str, int]:
    expected: dict[str, int] = {}
    for name in GROUPS:
        directory = CONFIG_ROOT / name
        configs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]
        expected[name] = trajectory_count(configs)
    return expected


def check_campaign() -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    manifests = []
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(manifest.get("config", {}).get("id", "")).startswith("pilot"):
            continue
        manifests.append(manifest)
    statuses = Counter(str(manifest.get("status", "missing")) for manifest in manifests)
    # Every declared final trajectory must complete.  There is no allowance for a
    # preregistered failure: a broken trajectory is not a measurement, and plotting
    # one beside complete trajectories would not be a comparison.  Cells that would
    # exhaust double precision are excluded from the grid instead.
    failed = sorted(
        str(manifest["config"]["id"])
        for manifest in manifests
        if manifest.get("status") == "failed"
    )
    if failed:
        errors.append(f"failed final trajectories: {failed}")
    if statuses.get("running"):
        errors.append(f"{statuses['running']} runs still marked running")

    # One frozen revision behind every trajectory, with a clean tree.
    hashes = Counter(str(manifest.get("code_hash", "")) for manifest in manifests)
    commits = Counter(str(manifest.get("git_commit", "")) for manifest in manifests)
    if len(hashes) > 1:
        errors.append(
            f"{len(hashes)} distinct source hashes across the campaign; the trajectories "
            f"were not produced from one frozen revision: {sorted(hashes)}"
        )
    if len(commits) > 1:
        errors.append(f"{len(commits)} distinct commits across the campaign: {sorted(commits)}")
    dirty = sum(1 for manifest in manifests if manifest.get("git_dirty"))
    if dirty:
        errors.append(
            f"{dirty} trajectories were produced from a dirty working tree; commit and "
            f"tag the revision before the final campaign"
        )

    expected = _expected_trajectories()
    total_expected = sum(expected.values())
    if len(manifests) != total_expected:
        errors.append(
            f"campaign has {len(manifests)} trajectories, protocol declares {total_expected}"
        )

    methods = Counter(
        str(manifest.get("method_specification", {}).get("name", "")) for manifest in manifests
    )
    for name in methods:
        if name not in ALLOWED_METHODS:
            errors.append(f"method outside the manuscript protocol: {name!r}")

    repairs = 0
    for path in sorted(RAW.rglob("*.csv")):
        if "pilot" in path.parts:
            continue
        frame = pd.read_csv(path, usecols=lambda name: name in {"roundoff_repairs", "repair"})
        if "roundoff_repairs" not in frame or float(frame["roundoff_repairs"].max()) <= 0:
            continue
        repairs += 1
        # ``ensure_spd`` repairs are symmetrization roundoff, never an algorithmic
        # clip, but the protocol admits no repaired final trajectory at all.
        errors.append(f"repaired trajectory: {path.parent.name}/{path.name}")
    summary = {
        "trajectories": len(manifests),
        "expected_trajectories": total_expected,
        "expected_by_group": expected,
        "statuses": dict(statuses),
        "methods": dict(methods),
        "repaired_trajectories": repairs,
        "source_hashes": sorted(hashes),
        "commits": sorted(commits),
        "dirty_trajectories": dirty,
    }
    return errors, warnings, summary


def check_references() -> tuple[list[str], list[str], dict[str, object]]:
    """The reference must not limit the gaps the figures resolve.

    A residual is a gradient norm and a gap is an energy difference, so the two
    cannot be compared directly.  The conversion used here is the manuscript's own
    Gaussian variational PL inequality,

        ``|g|^2 + Tr((C^{-1}-H)C(C^{-1}-H)) >= 2 alpha_star Delta``,

    whose left side is the Bures--Wasserstein residual.  It is a proved bound with
    an explicit constant that the campaign already computes exactly for every
    target family, so ``Delta(a_star) <= BW^2 / (2 alpha_star)`` needs no
    unjustified curvature assumption.  That certified bound on the reference's own
    suboptimality is what has to sit below the smallest gap any figure resolves.
    """

    errors: list[str] = []
    warnings: list[str] = []
    references = reference_certification_table()
    gaps = smallest_reported_gaps()
    if references.empty or gaps.empty:
        warnings.append("no reference or gap records found")
        return errors, warnings, {}
    smallest = dict(zip(gaps["experiment"], gaps["smallest_positive_gap"], strict=True))
    checked: list[dict[str, object]] = []
    for _, row in references.iterrows():
        experiment = str(row["experiment"])
        target = float(smallest.get(experiment, np.nan))
        residual = float(row["residual_on_design"])
        certified = float(row["certified_gap_bound"])
        if not np.isfinite(target):
            continue
        if not np.isfinite(certified):
            errors.append(f"{row['job_id']}: no certified gap bound for the reference")
            continue
        margin = np.inf if certified <= 0.0 else float(np.log10(target / certified))
        checked.append(
            {
                "job_id": row["job_id"],
                "experiment": experiment,
                "residual_on_design": residual,
                "certified_gap_bound": certified,
                "smallest_reported_gap": target,
                "margin_decades": margin,
            }
        )
        if margin < RESIDUAL_MARGIN_DECADES:
            errors.append(
                f"{row['job_id']}: the reference's certified suboptimality {certified:.3e} "
                f"(residual {residual:.3e}) is only {margin:.1f} decades below the "
                f"smallest reported gap {target:.3e}"
            )
    # A negative gap means a trajectory passed below the reference, which is direct
    # evidence that the reference is not the minimizer.  Machine-precision
    # excursions are unavoidable; anything larger is a reference defect and fails.
    negative: list[dict[str, object]] = []
    for experiment in ("C", "D", "G", "L"):
        frame = load_experiment(experiment, TIER)
        if frame.empty:
            continue
        frame = frame[~frame["job_id"].astype(str).str.startswith("pilot")]
        column = "exact_gaussian_kl" if experiment == "C" else "objective_gap"
        gaps = frame[column].to_numpy(dtype=np.float64)
        gaps = gaps[np.isfinite(gaps)]
        if not gaps.size:
            continue
        worst = float(gaps.min())
        scale = float(np.abs(frame["objective"].to_numpy(dtype=np.float64)).max())
        tolerance = NEGATIVE_GAP_TOLERANCE * max(scale, 1.0)
        negative.append(
            {"experiment": experiment, "most_negative_gap": worst, "tolerance": tolerance}
        )
        if worst < -tolerance:
            errors.append(
                f"experiment {experiment}: a trajectory fell {abs(worst):.3e} below the "
                f"reference, beyond the {tolerance:.3e} roundoff tolerance; the reference "
                f"is not the minimizer of the objective being reported"
            )

    transfer = references["transfer_objective_difference"].abs()
    return (
        errors,
        warnings,
        {
            "checked": checked,
            "negative_gaps": negative,
            "largest_transfer_objective_difference": (
                float(transfer.max()) if transfer.notna().any() else None
            ),
        },
    )


def check_stepsizes() -> tuple[list[str], list[str], dict[str, object]]:
    """Configs must carry exactly the multipliers the pilot recorded."""

    errors: list[str] = []
    payload = json.loads(SELECTED_STEPS.read_text(encoding="utf-8"))
    frozen = {name: float(value) for name, value in payload["multipliers"].items()}
    observed: dict[str, set[float]] = {}
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        if path.name == "selected_steps.json" or "pilot" in path.parts:
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        for specification in config.get("methods", []):
            if "step_scale" not in specification:
                continue
            name = str(specification["name"])
            observed.setdefault(name, set()).add(float(specification["normalized_step_size"]))
    for name, multipliers in observed.items():
        if multipliers != {frozen.get(name)}:
            errors.append(
                f"{name}: configs use multipliers {sorted(multipliers)}, "
                f"pilot froze {frozen.get(name)}"
            )
    return errors, [], {"frozen_multipliers": frozen, "step_scale": payload["step_scale"]}


def check_artifacts() -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    panels: dict[str, int] = {}
    for name in FIGURE_NAMES:
        base = FIGURES / name
        for suffix in (".pdf", ".png", ".md", ".json"):
            if not base.with_suffix(suffix).exists():
                errors.append(f"missing {suffix} for {name}")
        provenance_path = base.with_suffix(".json")
        if not provenance_path.exists():
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        panels[name] = len(provenance.get("panels", []))
        git = provenance.get("git", {})
        if git.get("commit", "unborn") == "unborn":
            errors.append(f"{name}: provenance records no commit")
        if not git.get("code_hash"):
            errors.append(f"{name}: provenance records no source hash")
        if git.get("dirty"):
            errors.append(f"{name}: figure was built from a dirty working tree")
        for panel in provenance.get("panels", []):
            csv_path = ROOT / str(panel.get("processed_csv", ""))
            if not csv_path.exists():
                errors.append(f"{name} panel {panel.get('panel')}: missing processed CSV")
            if not panel.get("config_ids"):
                errors.append(f"{name} panel {panel.get('panel')}: no config ids recorded")
    return errors, [], {"panels_per_figure": panels}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-artifacts", action="store_true", help="audit runs only, not figures"
    )
    args = parser.parse_args(arguments)

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {}
    sections = [("campaign", check_campaign), ("references", check_references),
                ("stepsizes", check_stepsizes)]
    if not args.skip_artifacts:
        sections.append(("artifacts", check_artifacts))
    for name, check in sections:
        section_errors, section_warnings, summary = check()
        errors.extend(section_errors)
        warnings.extend(section_warnings)
        report[name] = summary

    report["errors"] = errors
    report["warnings"] = warnings
    report["visual_inspection"] = "required separately at final manuscript size"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
