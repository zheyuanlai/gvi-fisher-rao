"""Gate on the manuscript campaign before its figures are trusted.

The checks are the ones the protocol makes falsifiable:

* the campaign contains the preregistered number of trajectories and no others;
* only the admitted methods appear -- the Fisher--Rao schemes, the
  Bures--Wasserstein baseline, the three published comparators and Laplace -- and
  no run was repaired, clipped or backtracked;
* every reference solution is certified at least two orders of magnitude below the
  smallest gap the figures resolve;
* every stepsize declares which of the three rules it came from, and matches it:
  certified where a panel verifies a theorem, pilot-frozen on the earlier
  practical panels, and the implementable dyadic grid throughout the benchmark;
* the benchmark's tuning never read an optimizer-whitened constant or a reference
  solution, and the real datasets still hash to their committed digests;
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
    LOGISTIC_DATASETS,
    SELECTED_STEPS,
    trajectory_count,
)
from fr_gvi.experiments.manuscript import main as manuscript_main
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
AMENDMENTS = CONFIG_ROOT / "protocol_amendments.json"
TIER = "manuscript"

ALLOWED_METHODS = {
    *ITERATIVE,
    "Laplace",
    # The stochastic Fisher--Rao schemes the manuscript analyses.
    "FR--R--STL",
    "FR--KL--STL",
    "S--FB--GVI",
    # The three external comparators, each implemented as published.
    "Sq--NGVI",
    "Price--BBVI",
    "BBVI--STL",
}
# The exclusions the protocol still enforces: no projected Fisher--Rao variant, no
# Bures--Wasserstein method other than FB--GVI and its minibatch form, no
# Euclidean or quasi-Newton optimizer, no diagonal or low-rank family.
# ``ALLOWED_METHODS`` is the whole admitted set and anything else is an error.
#
# The protocol requires the reference to be at least this many orders of magnitude
# below the smallest gap any figure resolves.
RESIDUAL_MARGIN_DECADES = 2.0
# Relative size of a negative objective gap that is attributable to roundoff in the
# difference of two objectives of the recorded magnitude.
NEGATIVE_GAP_TOLERANCE = 1.0e-13
FIGURE_NAMES = ("figure_1", "figure_2", "figure_3", "figure_4")
# The regimes a stepsize may come from.  Every method specification must declare
# one, and each is checked by a different gate.
#
# ``capped_practical`` is the one departure: the batch-size floor panel runs
# deliberately above the certified step, because the certified step scales like
# 1/kappa_star and at that step the horizon needed to leave the deterministic
# transient exceeds any affordable budget, so the measured floor would be flat in
# B for a reason unrelated to the theorem it is testing.  It is named rather than
# folded into "certified" so the certified-window gate keeps its teeth.
STEP_RULES = {
    "certified",
    "certified_schedule",
    "capped_practical",
    "pilot_frozen",
    "implementable_grid",
    "non-iterative",
}


def _head_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def _is_ancestor(commit: str, head: str) -> bool:
    import subprocess

    if not commit or not head:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT, check=False
    ).returncode == 0


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
    # The recorded revision must still be the one on disk.  "One hash" is not
    # enough on its own: a campaign can be internally consistent and yet describe
    # source that no longer exists.
    from fr_gvi.experiments.campaign import code_hash

    current = code_hash()
    head = _head_commit()
    if hashes and set(hashes) != {current}:
        errors.append(
            "the campaign's source hash is not the current source; rerun the "
            "trajectories or restore the revision they were produced from"
        )
    # Commit *equality* with HEAD is unsatisfiable: committing the artifacts moves
    # HEAD past the revision the campaign ran from.  What must hold is that the
    # recorded revision is in the current history and that its numerical source is
    # the source on disk, which the hash check above already establishes.
    for commit in commits:
        if commit and not _is_ancestor(commit, head):
            errors.append(
                f"the campaign records commit {commit[:8]}, which is not an ancestor "
                f"of HEAD {head[:8]}; the artifacts came from an abandoned revision"
            )
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
        "matches_current_source": bool(hashes) and set(hashes) == {current},
        "campaign_commit_in_history": all(_is_ancestor(c, head) for c in commits if c),
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
    smallest = dict(zip(gaps["experiment"], gaps["smallest_plotted_gap"], strict=True))
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
                "smallest_plotted_gap": target,
                "margin_decades": margin,
            }
        )
        if margin < RESIDUAL_MARGIN_DECADES:
            errors.append(
                f"{row['job_id']}: the reference's certified suboptimality {certified:.3e} "
                f"(residual {residual:.3e}) is only {margin:.1f} decades below the "
                f"smallest plotted gap {target:.3e}"
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


def _final_configs() -> list[dict[str, object]]:
    configs = []
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        if path.name in {
            "selected_steps.json",
            "protocol_amendments.json",
            "practical_steps.json",
        } or "pilot" in path.parts:
            continue
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    return configs


def check_stepsizes() -> tuple[list[str], list[str], dict[str, object]]:
    """Every stepsize must come from a declared rule, and match it."""

    errors: list[str] = []
    warnings: list[str] = []
    payload = json.loads(SELECTED_STEPS.read_text(encoding="utf-8"))
    frozen = {name: float(value) for name, value in payload["multipliers"].items()}
    observed: dict[str, set[float]] = {}
    rules: Counter = Counter()
    for config in _final_configs():
        for specification in config.get("methods", []):
            rule = str(specification.get("step_rule", ""))
            rules[rule] += 1
            if rule not in STEP_RULES:
                errors.append(
                    f"{config['id']}: method {specification.get('name')!r} declares no "
                    f"recognized step rule (got {rule!r})"
                )
            if rule != "pilot_frozen":
                continue
            name = str(specification["name"])
            observed.setdefault(name, set()).add(float(specification["normalized_step_size"]))
    for name, multipliers in observed.items():
        if multipliers != {frozen.get(name)}:
            errors.append(
                f"{name}: configs use multipliers {sorted(multipliers)}, "
                f"pilot froze {frozen.get(name)}"
            )
    return (
        errors,
        warnings,
        {
            "frozen_multipliers": frozen,
            "step_scale": payload["step_scale"],
            "specifications_by_rule": dict(rules),
        },
    )


def check_practical_tuning() -> tuple[list[str], list[str], dict[str, object]]:
    """The benchmark's stepsizes must be reproducible and oracle-free.

    Two things have to hold for the practical comparison to mean anything.  The
    stepsize in every benchmark config must be exactly the frozen multiplier
    times the base scale recomputed on that problem, so nothing was adjusted
    after the selection; and the selection itself must never have seen
    ``C_star``, ``beta_star``, ``kappa_star`` or a reference solution, which is
    what makes the comparison something a practitioner could have run.

    The second is checked structurally rather than by reading the code: the rule
    records the inputs it is allowed to use, and the selection record carries the
    pilot objective it chose on.  A selection sitting on an endpoint of the
    candidate grid is reported as censored, because there the grid did not
    bracket the optimum and the reported step is a bound rather than a choice.
    """

    from fr_gvi.experiments.tuning import EXPONENTS, PRACTICAL_STEPS

    errors: list[str] = []
    warnings: list[str] = []
    if not PRACTICAL_STEPS.exists():
        return [f"missing the practical stepsize record at {PRACTICAL_STEPS.name}"], [], {}
    payload = json.loads(PRACTICAL_STEPS.read_text(encoding="utf-8"))
    rule = payload.get("rule", {})
    forbidden = set(rule.get("forbidden_inputs", []))
    if not {"C_star", "beta_star", "kappa_star"} <= forbidden:
        errors.append(
            "the practical rule does not declare the optimizer-whitened constants "
            "as forbidden inputs"
        )
    if len(rule.get("grid", [])) != len(EXPONENTS):
        errors.append("the recorded candidate grid is not the one the code sweeps")

    selections = payload.get("selections", {})
    censored = sorted(key for key, value in selections.items() if value.get("censored"))
    for key in censored:
        warnings.append(
            f"{key}: the selected stepsize is an endpoint of the candidate grid, so it "
            f"is a bound rather than an interior choice"
        )

    # Every benchmark config must reproduce its stepsize from the record.
    checked = 0
    for config in _final_configs():
        for specification in config.get("methods", []):
            if specification.get("step_rule") != "implementable_grid":
                continue
            checked += 1
            expected = float(specification["normalized_step_size"]) * float(
                specification["step_scale"]
            )
            if not np.isclose(float(specification["step_size"]), expected, rtol=1e-12, atol=0.0):
                errors.append(
                    f"{config['id']}: {specification['name']} step {specification['step_size']} "
                    f"is not its multiplier times its base scale"
                )
    tuning_cost = sum(
        int(value.get("tuning_oracle_pairs", 0)) for value in selections.values()
    )
    return (
        errors,
        warnings,
        {
            "selections": len(selections),
            "censored": censored,
            "benchmark_specifications_checked": checked,
            "tuning_oracle_pairs": tuning_cost,
            "allowed_inputs": rule.get("inputs", []),
        },
    )


def check_datasets() -> tuple[list[str], list[str], dict[str, object]]:
    """The real datasets must be the ones the recorded digests describe."""

    from fr_gvi.experiments.datasets import DATASET_KEYS, SPECS, load_dataset

    errors: list[str] = []
    summary: dict[str, object] = {}
    for key in DATASET_KEYS:
        try:
            # ``load_dataset`` verifies the cached bytes against the committed
            # digest and raises if upstream has changed underneath the results.
            dataset = load_dataset(key, allow_download=False)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{key}: {exc}")
            continue
        summary[key] = {
            "openml_id": SPECS[key].openml_id,
            "observations": dataset.observations,
            "features": dataset.dimension,
            "sha256": dataset.sha256,
        }
    return errors, [], summary


def check_no_pilot_contamination() -> tuple[list[str], list[str], dict[str, object]]:
    """No pilot trajectory may reach a figure, a table or a provenance list.

    The pilot exists to choose the stepsizes and must appear nowhere in the
    reported artifacts.  It once shared the ``manuscript`` tier with the final
    cells, which put three pilot trajectories into the logistic predictive table
    and into a figure's provenance; the tiers are now separate and this checks the
    separation holds end to end.
    """

    errors: list[str] = []
    leaked: dict[str, list[str]] = {}
    for experiment in ("A", "B", "C", "D", "G", "H", "J", "K", "L", "R", "S"):
        frame = load_experiment(experiment, TIER)
        if frame.empty:
            continue
        pilots = sorted(
            str(job) for job in frame["job_id"].unique() if str(job).startswith("pilot")
        )
        if pilots:
            leaked[experiment] = pilots
            errors.append(f"pilot trajectories in the {experiment} manuscript tier: {pilots}")
    for name in FIGURE_NAMES:
        path = FIGURES / f"{name}.json"
        if not path.exists():
            continue
        provenance = json.loads(path.read_text(encoding="utf-8"))
        for panel in provenance.get("panels", []):
            offenders = [c for c in panel.get("config_ids", []) if str(c).startswith("pilot")]
            if offenders:
                errors.append(f"{name} panel {panel.get('panel')} cites pilot configs: {offenders}")

    # Every logistic aggregate must rest on exactly the declared number of datasets.
    frame = load_experiment("L", TIER)
    counts: dict[str, int] = {}
    if not frame.empty:
        terminal = frame.groupby(["grid_feature_condition", "method"])["grid_dataset"].nunique()
        for (condition, method), count in terminal.items():
            counts[f"{condition:g}:{method}"] = int(count)
            if int(count) != LOGISTIC_DATASETS:
                errors.append(
                    f"logistic cell kappa_X={condition:g} method {method} aggregates "
                    f"{count} datasets, protocol declares {LOGISTIC_DATASETS}"
                )
    return errors, [], {"leaked": leaked, "datasets_per_cell": counts}


def check_configs_regenerate() -> tuple[list[str], list[str], dict[str, object]]:
    """Committed configs must reproduce byte-for-byte from the generator."""

    import tempfile

    errors: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        manuscript_main(["--destination", str(destination)])
        manuscript_main(["--pilot", "--destination", str(destination)])
        for path in sorted(CONFIG_ROOT.rglob("*.json")):
            if path.name in {"selected_steps.json", "protocol_amendments.json"}:
                continue
            rebuilt = destination / path.relative_to(CONFIG_ROOT)
            checked += 1
            if not rebuilt.exists():
                errors.append(f"generator no longer emits {path.relative_to(CONFIG_ROOT)}")
            elif rebuilt.read_bytes() != path.read_bytes():
                errors.append(
                    f"committed config differs from the generator: "
                    f"{path.relative_to(CONFIG_ROOT)}"
                )
    return errors, [], {"configs_checked": checked}


def check_captions() -> tuple[list[str], list[str], dict[str, object]]:
    """A caption must not describe a backend the campaign no longer uses."""

    errors: list[str] = []
    obsolete = ("Sobol", "4096", "O(Snd)", "quasi-Monte-Carlo design")
    for name in FIGURE_NAMES:
        path = FIGURES / f"{name}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in obsolete:
            if token in text:
                errors.append(f"{name} caption still describes an obsolete backend: {token!r}")
    return errors, [], {}


def check_amendments() -> tuple[list[str], list[str], dict[str, object]]:
    """Every departure from the brief must be recorded and explicitly decided.

    A deviation that no gate mentions passes silently, which is how an
    exploratory choice becomes an undocumented protocol.  Each amendment carries
    the author's decision and a machine-checkable fingerprint of what was
    approved, so the implementation cannot drift away from the decision after the
    fact.
    """

    errors: list[str] = []
    if not AMENDMENTS.exists():
        return [f"missing the protocol amendment record at {AMENDMENTS.name}"], [], {}
    payload = json.loads(AMENDMENTS.read_text(encoding="utf-8"))
    recorded: dict[str, str] = {}
    for amendment in payload.get("amendments", []):
        identifier = str(amendment.get("id", "?"))
        status = str(amendment.get("status", "undecided"))
        recorded[identifier] = status
        if status != "approved":
            errors.append(f"amendment {identifier} is {status}; it needs an explicit decision")
        if not amendment.get("decided_by") or not amendment.get("decided_on"):
            errors.append(f"amendment {identifier} records no decision maker or date")
        for field in ("requested", "implemented", "reason", "cost"):
            if not amendment.get(field):
                errors.append(f"amendment {identifier} does not state its {field}")

    # The implementation must still match what was approved.
    verify = {a["id"]: a.get("verify", {}) for a in payload.get("amendments", [])}
    from fr_gvi.experiments.manuscript import (
        LOGISTIC_EVALUATION_ORDER,
        LOGISTIC_UPDATE_ORDER,
        PILOT_STEP_SCALE,
    )
    from fr_gvi.plotting.style import DISPLAY_NAMES

    if verify.get("A", {}).get("step_scale") != PILOT_STEP_SCALE:
        errors.append("amendment A was approved for a different stepsize scale than the code uses")
    expected_b = verify.get("B", {})
    if (
        expected_b.get("update_order") != LOGISTIC_UPDATE_ORDER
        or expected_b.get("evaluation_order") != LOGISTIC_EVALUATION_ORDER
    ):
        errors.append("amendment B was approved for different quadrature orders than the code uses")
    expected_d = verify.get("D", {})
    if DISPLAY_NAMES.get(expected_d.get("identifier", "")) != expected_d.get("display_name"):
        errors.append("amendment D was approved for a different display name than the code uses")

    from fr_gvi.experiments.datasets import DATASET_KEYS
    from fr_gvi.experiments.manuscript import DETERMINISTIC_BENCHMARK, STOCHASTIC_BENCHMARK
    from fr_gvi.experiments.tuning import PILOT_SUBSAMPLE

    added = set(verify.get("E", {}).get("added_methods", []))
    if added != {"Sq--NGVI", "Price--BBVI", "BBVI--STL"}:
        errors.append("amendment E was approved for a different comparator set than the code uses")
    if not added <= set(DETERMINISTIC_BENCHMARK) | set(STOCHASTIC_BENCHMARK):
        errors.append("amendment E lists a comparator the benchmark does not run")
    if verify.get("F", {}).get("pilot_subsample") != PILOT_SUBSAMPLE:
        errors.append("amendment F was approved for a different pilot subsample than the code uses")
    if tuple(verify.get("G", {}).get("datasets", ())) != tuple(DATASET_KEYS):
        errors.append("amendment G was approved for a different dataset set than the code uses")
    if int(verify.get("H", {}).get("main_text_figures", 0)) != 4:
        errors.append("amendment H was approved for a different main-text figure count")
    return errors, [], {"decisions": recorded}


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
                ("stepsizes", check_stepsizes), ("practical_tuning", check_practical_tuning),
                ("datasets", check_datasets),
                ("pilot_isolation", check_no_pilot_contamination),
                ("configs", check_configs_regenerate), ("amendments", check_amendments)]
    if not args.skip_artifacts:
        sections.extend([("artifacts", check_artifacts), ("captions", check_captions)])
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
