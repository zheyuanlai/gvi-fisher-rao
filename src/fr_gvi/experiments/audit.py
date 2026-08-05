from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _pathwise_covariance_bands() -> dict[str, object]:
    """Check the almost-sure covariance bands of Lemma 4.4 on every stochastic run.

    Under the quadratic rescue the Riemannian scheme must satisfy
    ``(2 beta_star)^{-1} I <= C_n <= alpha_star^{-1} I`` and the KL/Bregman scheme
    ``beta_star^{-1} I <= C_n <= alpha_star^{-1} I``, both in optimizer-whitened
    coordinates and pathwise rather than in expectation.
    """

    errors: list[str] = []
    warnings: list[str] = []
    lower_slack = {"FR--R--STL": [], "FR--KL--STL": []}
    upper_slack = {"FR--R--STL": [], "FR--KL--STL": []}
    checked = 0
    for path in sorted((ROOT / "results" / "raw" / "full").rglob("*.csv")):
        if "-qr" not in path.stem:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        method = rows[0].get("method", "")
        if method not in lower_slack:
            continue
        try:
            beta_star = float(rows[0]["beta_star"])
            alpha_star = float(rows[0]["alpha_star"])
        except (KeyError, ValueError):
            continue
        lower_bound = 1.0 / (2.0 * beta_star) if method == "FR--R--STL" else 1.0 / beta_star
        upper_bound = 1.0 / alpha_star
        checked += 1
        for line, row in enumerate(rows, start=2):
            try:
                minimum = float(row["relative_covariance_min_eigenvalue"])
                maximum = float(row["relative_covariance_max_eigenvalue"])
            except (KeyError, ValueError):
                continue
            lower_slack[method].append(minimum - lower_bound)
            upper_slack[method].append(upper_bound - maximum)
            if minimum < lower_bound * (1.0 - 1e-9):
                errors.append(
                    f"covariance band violated below at {path.name}:{line}: "
                    f"{minimum:.6e} < {lower_bound:.6e}"
                )
            if maximum > upper_bound * (1.0 + 1e-9):
                errors.append(
                    f"covariance band violated above at {path.name}:{line}: "
                    f"{maximum:.6e} > {upper_bound:.6e}"
                )
    if not checked:
        warnings.append("no rescued stochastic runs found for the covariance-band check")
    summary = {
        "runs_checked": checked,
        "minimum_lower_slack": {
            method: (min(values) if values else None) for method, values in lower_slack.items()
        },
        "minimum_upper_slack": {
            method: (min(values) if values else None) for method, values in upper_slack.items()
        },
    }
    return {"errors": errors, "warnings": warnings, "summary": summary}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args(arguments)
    errors: list[str] = []
    warnings: list[str] = []
    manifests = []
    for path in sorted((ROOT / "results" / "manifests").rglob("*.json")):
        if path.name.startswith("reference_") or path.name == "campaign_state.json":
            continue
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid manifest {path}: {exc}")
    statuses = Counter(str(manifest.get("status", "missing")) for manifest in manifests)
    if statuses["failed"] and not args.allow_failed:
        errors.append(f"{statuses['failed']} failed run manifests")
    if statuses["running"]:
        warnings.append(f"{statuses['running']} runs remain marked running")
    for path in sorted((ROOT / "results" / "raw").rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                value = row.get("covariance_min_eigenvalue", "")
                if value:
                    try:
                        if float(value) <= 0.0:
                            errors.append(f"nonpositive covariance at {path}:{line}")
                    except ValueError:
                        errors.append(f"invalid covariance eigenvalue at {path}:{line}")
    forbidden = ["BWGD", "BW--SGD", "covariance_clip", "projected_fisher", "hybrid_geometry"]
    for root in (ROOT / "src", ROOT / "configs"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".json", ".toml"}:
                continue
            if path.name == "audit.py":
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    errors.append(f"forbidden token {token!r} in {path.relative_to(ROOT)}")
    theory_checks = _pathwise_covariance_bands()
    errors.extend(theory_checks["errors"])
    warnings.extend(theory_checks["warnings"])
    for png in sorted((ROOT / "results" / "figures").glob("experiment_*.png")):
        pdf = png.with_suffix(".pdf")
        markdown = png.with_suffix(".md")
        if not pdf.exists():
            errors.append(f"missing PDF partner for {png.name}")
        if not markdown.exists():
            errors.append(f"missing caption draft for {png.name}")
    report = {
        "pathwise_covariance_bands": theory_checks["summary"],
        "manifest_statuses": dict(statuses),
        "manifest_count": len(manifests),
        "raw_csv_count": len(list((ROOT / "results" / "raw").rglob("*.csv"))),
        "figure_png_count": len(list((ROOT / "results" / "figures").glob("*.png"))),
        "errors": errors,
        "warnings": warnings,
    }
    output = ROOT / "reports" / "AUDIT_RESULTS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

