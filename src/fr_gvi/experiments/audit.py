from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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
    for png in sorted((ROOT / "results" / "figures").glob("experiment_*.png")):
        pdf = png.with_suffix(".pdf")
        markdown = png.with_suffix(".md")
        if not pdf.exists():
            errors.append(f"missing PDF partner for {png.name}")
        if not markdown.exists():
            errors.append(f"missing caption draft for {png.name}")
    report = {
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

