from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    rows: list[dict[str, str]] = []
    for path in sorted((ROOT / "results" / "raw").rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            values = list(csv.DictReader(handle))
        if not values or "objective_gap" not in values[-1]:
            continue
        final = values[-1]
        rows.append(
            {
                "experiment": final.get("experiment", ""),
                "job_id": final.get("job_id", ""),
                "method": final.get("method", ""),
                "seed": final.get("seed", ""),
                "iterations": final.get("iteration", ""),
                "objective_gap": final.get("objective_gap", ""),
                "w2_squared": final.get("w2_squared", ""),
                "wall_time_seconds": final.get("wall_time_seconds", ""),
                "oracle_pairs": final.get("oracle_pairs", ""),
            }
        )
    table_dir = ROOT / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "terminal_summary.csv"
    fields = list(rows[0]) if rows else ["experiment", "method"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tex_path = table_dir / "terminal_summary.tex"
    lines = [
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        r"Experiment & Job & Method & Iter. & Objective gap \\",
        r"\midrule",
    ]
    for row in rows:
        try:
            gap = f"{float(row['objective_gap']):.3e}"
        except ValueError:
            gap = "--"
        lines.append(
            f"{row['experiment']} & {row['job_id']} & {row['method']} & {row['iterations']} & {gap} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path} and {tex_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

