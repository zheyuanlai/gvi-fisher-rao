"""Selection of the frozen practical stepsizes from the designated pilot cell.

The manuscript figures do not show a stepsize sweep.  One multiplier per method is
chosen once, on pilot targets that appear in no panel, and then reused for every
non-Gaussian experiment.  The four admissibility criteria are applied exactly as
stated in the protocol:

1. the covariance stays positive definite for the whole horizon;
2. every recorded objective is finite;
3. the objective decreases over the pilot horizon;
4. no clipping, repair or backtracking is logged.

The largest multiplier for which these hold, and for which they also hold at every
smaller multiplier, is the top of the contiguous stable range on the pilot cell.
That range ends at the *stability* boundary, which is not the same as the fastest
step: a Fisher--Rao step with ``h gamma_star`` near two is admissible by all four
criteria yet converges slowly, because the linearized one-step factor
``1 - h gamma_star`` is then close to ``-1``.  Among the admissible multipliers the
frozen one is therefore the fastest, that is the one reaching a relative gap of
``1e-6`` in the fewest iterations on the pilot.

The scale the multiplier applies to is
``1/(beta_star * max(lambda_{0,star}^max, 1))``; see ``practical_step_scale`` for why
neither the certified step nor ``1/beta_star`` alone transfers.  There are two pilot
cells, one per target family, because a single cell cannot calibrate both regimes,
and the frozen multiplier is the smallest of the per-pilot choices.

FB--GVI is additionally held at multiplier one, where Diao et al.'s requirement
``eta <= 1/beta`` ends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fr_gvi.experiments.manuscript import (
    ITERATIVE,
    PILOT_CELL,
    PILOT_LOGISTIC_CELL,
    PILOT_LOGISTIC_SEED,
    PILOT_SEED,
    SELECTED_STEPS,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw" / "manuscript"
MANIFESTS = ROOT / "results" / "manifests" / "manuscript"
# Relative gap used to rank the admissible multipliers by speed on each pilot.
PILOT_TOLERANCE = 1.0e-6


def _manifests() -> dict[tuple[str, str, float], dict[str, Any]]:
    found: dict[tuple[str, str, float], dict[str, Any]] = {}
    for path in sorted(MANIFESTS.glob("pilot_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        specification = manifest.get("method_specification", {})
        if "name" not in specification:
            continue
        family = str(manifest["config"]["grid"]["family"])
        key = (
            family,
            str(specification["name"]),
            float(specification["normalized_step_size"]),
        )
        found[key] = manifest
    return found


def evaluate() -> pd.DataFrame:
    """One row per (family, method, multiplier) with each criterion resolved."""

    manifests = _manifests()
    rows: list[dict[str, Any]] = []
    for path in sorted(RAW.glob("*/pilot_*/*.csv")):
        frame = pd.read_csv(path).sort_values("iteration")
        if frame.empty:
            continue
        method = str(frame["method"].iloc[0])
        multiplier = float(frame["normalized_step_size"].iloc[0])
        family = "logcosh" if "logcosh" in path.parent.name else "logistic"
        manifest = manifests.get((family, method, multiplier), {})
        objectives = frame["objective"].to_numpy(dtype=np.float64)
        finite = bool(np.all(np.isfinite(objectives)))
        positive = bool(np.all(frame["covariance_min_eigenvalue"].to_numpy() > 0.0))
        completed = str(manifest.get("status", "missing")) == "completed"
        # ``roundoff_repairs`` counts symmetrized eigenvalue repairs and the
        # ``repair`` column carries their details; either being present disqualifies
        # the step, since the protocol admits no repaired trajectory.
        repairs = int(np.nan_to_num(frame.get("roundoff_repairs", pd.Series([0])).max()))
        logged = bool(frame.get("repair", pd.Series([np.nan])).notna().any())
        repaired = repairs > 0 or logged
        decreased = finite and bool(objectives[-1] < objectives[0])
        normalized = (
            frame["objective_gap"].to_numpy(dtype=np.float64)
            / float(frame["objective_gap"].iloc[0])
        )
        reached = np.where(normalized <= PILOT_TOLERANCE)[0]
        iterations_to_tolerance = (
            float(frame["iteration"].to_numpy()[reached[0]]) if reached.size else np.inf
        )
        rows.append(
            {
                "family": family,
                "method": method,
                "multiplier": multiplier,
                "step_size": float(frame["step_size"].iloc[-1]),
                "step_scale": float(frame["step_size"].iloc[-1]) / multiplier,
                "certified_step_size": float(
                    manifest.get("method_specification", {}).get("certified_step_size", np.nan)
                ),
                "iterations_completed": int(frame["iteration"].iloc[-1]),
                "status": str(manifest.get("status", "missing")),
                "failure_reason": str(manifest.get("failure_reason", "")),
                "finite_objective": finite and completed,
                "positive_definite": positive and completed,
                "objective_decreased": decreased and completed,
                "no_repair": not repaired,
                "iterations_to_tolerance": iterations_to_tolerance,
                "final_objective_gap": float(frame["objective_gap"].iloc[-1]),
            }
        )
    table = pd.DataFrame(rows)
    table["admissible"] = (
        table["finite_objective"]
        & table["positive_definite"]
        & table["objective_decreased"]
        & table["no_repair"]
    )
    return table.sort_values(["family", "method", "multiplier"]).reset_index(drop=True)


# Diao et al. require ``eta <= 1/beta`` for FB--GVI, and ``beta = beta_star`` on the
# optimizer-whitened targets, so on the ``1/beta_star`` scale the external baseline's
# own admissible range ends at multiplier one.  Running it past that would compare
# against a method outside its published guarantee.  The Fisher--Rao schemes have no
# such ceiling on this scale: their certificate is stated on a different scale and is
# reported separately.
METHOD_CEILING = {"FB--GVI": 1.0}


def largest_admissible(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Top of the contiguous stable multiplier range, per family and method."""

    found: dict[str, dict[str, Any]] = {}
    for family in sorted(table["family"].unique()):
        for method in ITERATIVE:
            subset = table[
                (table["family"] == family) & (table["method"] == method)
            ].sort_values("multiplier")
            if subset.empty:
                raise RuntimeError(f"no pilot runs found for {method} on {family}")
            best = None
            for _, row in subset.iterrows():
                if not bool(row["admissible"]):
                    break
                best = float(row["multiplier"])
            if best is None:
                raise RuntimeError(f"no admissible stepsize for {method} on {family}")
            grid_maximum = float(subset["multiplier"].max())
            found[f"{family}:{method}"] = {
                "largest_admissible": best,
                "boundary_located": best < grid_maximum,
                "grid_maximum": grid_maximum,
            }
    return found


def _fastest(table: pd.DataFrame, family: str, method: str) -> float:
    """Fastest admissible multiplier for one method on one pilot family."""

    subset = table[(table["family"] == family) & (table["method"] == method)]
    ceiling = METHOD_CEILING.get(method, np.inf)
    candidates = subset[
        subset["admissible"]
        & (subset["multiplier"] <= ceiling)
        & np.isfinite(subset["iterations_to_tolerance"])
    ]
    if candidates.empty:
        # Nothing reaches the tolerance inside this pilot's horizon, so rank by the
        # gap actually achieved instead; the step is still required to be admissible.
        candidates = subset[subset["admissible"] & (subset["multiplier"] <= ceiling)]
        if candidates.empty:
            raise RuntimeError(f"no admissible stepsize for {method} on {family}")
        return float(
            candidates.sort_values(["final_objective_gap", "multiplier"]).iloc[0]["multiplier"]
        )
    return float(
        candidates.sort_values(["iterations_to_tolerance", "multiplier"]).iloc[0]["multiplier"]
    )


def select(table: pd.DataFrame) -> dict[str, float]:
    """Frozen multipliers: the fastest admissible multiplier on every pilot family.

    Ties break towards the smaller step, each method is held below the ceiling its
    own convergence theorem admits on this scale, and the minimum is taken across
    the pilot families so one frozen number is admissible on both.
    """

    families = sorted(table["family"].unique())
    return {
        method: min(_fastest(table, family, method) for family in families)
        for method in ITERATIVE
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="commit selected_steps.json")
    args = parser.parse_args(arguments)

    table = evaluate()
    columns = [
        "family", "method", "multiplier", "step_size", "status", "finite_objective",
        "positive_definite", "objective_decreased", "no_repair", "admissible",
        "iterations_to_tolerance", "final_objective_gap",
    ]
    print(table[columns].to_string(index=False))
    pilot_result = largest_admissible(table)
    chosen = select(table)
    print("\npilot sweep result:", json.dumps(pilot_result, sort_keys=True))
    print("frozen multipliers:", json.dumps(chosen, sort_keys=True))

    if args.write:
        payload = {
            "pilots": [
                {
                    "family": "optimizer-whitened shifted log-cosh",
                    "master_seed": PILOT_SEED,
                    "cell": PILOT_CELL,
                },
                {
                    "family": "Bayesian logistic regression",
                    "master_seed": PILOT_LOGISTIC_SEED,
                    "cell": PILOT_LOGISTIC_CELL,
                },
            ],
            "pilot_note": (
                "One pilot per target family, because a single cell cannot calibrate "
                "both regimes: the whitened log-cosh cell starts with its covariance "
                "already in the band, the logistic cell starts two orders of magnitude "
                "above it. Each uses the nominal parameters of a cell that appears in "
                "the figures but a different master seed, hence a different target "
                "instance; no trajectory shown in a figure was used to select a "
                "stepsize. The frozen multiplier is the smallest of the per-pilot "
                "choices, so it is admissible on both."
            ),
            "admissibility_criteria": [
                "positive definite covariance throughout",
                "finite objective values",
                "objective decrease over the pilot horizon",
                "no clipping, repairs or backtracking",
            ],
            "selection_rule": (
                "among the admissible multipliers, the one reaching a relative gap "
                f"of {PILOT_TOLERANCE:g} in the fewest iterations on the pilot, ties "
                "broken towards the smaller step"
            ),
            "step_scale": {
                "definition": "h = eta / (beta_star * max(lambda_0_star_max, 1))",
                "reason": (
                    "What must stay of order one is h beta_star times the whitened "
                    "covariance, which is largest at the initialization whenever C_0 "
                    "overshoots C_star. Two other scales each fail on one family. A "
                    "multiple of the certified step 1/(2 beta_star lambda_max_star) "
                    "carries the worst-case growth allowance lambda_max_star >= "
                    "1/alpha_star, never attained on the whitened log-cosh cells, so "
                    "its conservatism grows like kappa_star and a multiplier chosen at "
                    "kappa_star = 11 diverges at kappa_star = 2. A multiple of "
                    "1/beta_star alone ignores the initialization and diverges on the "
                    "logistic posteriors, where C_0 overshoots C_star by two orders of "
                    "magnitude. The scale used is the certified step with the "
                    "non-attained growth allowance removed and the initialization kept."
                ),
            },
            "pilot_sweep_result": pilot_result,
            "ceilings": {
                "values": METHOD_CEILING,
                "rule": "eta = min over pilots of min(fastest admissible, method ceiling)",
                "reason": (
                    "Diao et al. require eta <= 1/beta for FB-GVI, so the external "
                    "baseline is never run outside its own published admissible range."
                ),
            },
            "multipliers": chosen,
            "selected_step_sizes_on_pilots": [
                {
                    "family": family,
                    "method": method,
                    "step_scale": float(
                        table[(table["family"] == family) & (table["method"] == method)][
                            "step_scale"
                        ].median()
                    ),
                    "certified_step_size": float(
                        table[(table["family"] == family) & (table["method"] == method)][
                            "certified_step_size"
                        ].median()
                    ),
                    "step_size": chosen[method]
                    * float(
                        table[(table["family"] == family) & (table["method"] == method)][
                            "step_scale"
                        ].median()
                    ),
                }
                for family in sorted(table["family"].unique())
                for method in ITERATIVE
            ],
            "sweep": table.to_dict(orient="records"),
        }
        SELECTED_STEPS.parent.mkdir(parents=True, exist_ok=True)
        SELECTED_STEPS.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {SELECTED_STEPS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
