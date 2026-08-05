from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results" / "figures"


def _command(*arguments: str) -> str:
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


def audit_figure_pair(png: Path, expected_width_inches: float, tolerance: float = 0.002) -> dict[str, object]:
    """Check a PDF/PNG pair for physical width, font embedding and consistency.

    Figures retain their fixed manuscript-width canvas, so the PDF page width is
    checked against the nominal text width with only a small serialization tolerance.
    """

    pdf = png.with_suffix(".pdf")
    findings: dict[str, object] = {"png": str(png), "pdf": str(pdf), "errors": [], "warnings": []}
    errors = findings["errors"]
    warnings = findings["warnings"]
    if not pdf.exists():
        errors.append("missing matching PDF")
        return findings
    image = mpimg.imread(png)
    height_pixels, width_pixels = image.shape[:2]
    findings["png_pixels"] = [int(width_pixels), int(height_pixels)]
    if width_pixels < 1200:
        errors.append(f"PNG width {width_pixels}px is below the 1200px print threshold")
    if shutil.which("pdfinfo"):
        information = _command("pdfinfo", str(pdf))
        match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", information)
        if not match:
            errors.append("could not parse PDF page size")
        else:
            width_points, height_points = map(float, match.groups())
            findings["pdf_points"] = [width_points, height_points]
            expected_points = expected_width_inches * 72.0
            if abs(width_points - expected_points) > tolerance * expected_points:
                errors.append(
                    f"PDF width {width_points:.2f}pt is outside "
                    f"{100 * tolerance:.0f}% of the {expected_points:.2f}pt text width"
                )
            findings["width_inches"] = round(width_points / 72.0, 4)
    else:
        warnings.append("pdfinfo unavailable; physical PDF width not checked")
    if shutil.which("pdffonts"):
        font_output = _command("pdffonts", str(pdf))
        font_rows = [line.split() for line in font_output.splitlines()[2:] if line.strip()]
        findings["font_count"] = len(font_rows)
        if not font_rows:
            errors.append("PDF contains no detected fonts")
        for row in font_rows:
            if len(row) >= 6 and row[4].lower() != "yes":
                errors.append(f"font is not embedded: {' '.join(row[:2])}")
    else:
        warnings.append("pdffonts unavailable; font embedding not checked")
    if shutil.which("pdftotext"):
        extracted_text = _command("pdftotext", "-layout", str(pdf), "-")
        double_dash_tokens = sorted(set(re.findall(r"\S*--\S*", extracted_text)))
        if double_dash_tokens:
            errors.append(
                "PDF contains double-dash figure text: " + ", ".join(double_dash_tokens)
            )
    else:
        warnings.append("pdftotext unavailable; single-dash figure text not checked")
    caption = png.with_suffix(".md")
    if caption.exists() and "--" in caption.read_text(encoding="utf-8"):
        errors.append("caption draft contains double-dash text")
    return findings


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--width-inches", type=float, default=6.5)
    args = parser.parse_args(arguments)
    paths = sorted(
        {
            *args.figures.glob("experiment_*.png"),
            *args.figures.glob("main_figure_*.png"),
        }
    )
    pairs = [audit_figure_pair(path, args.width_inches) for path in paths]
    errors = [error for pair in pairs for error in pair["errors"]]
    warnings = [warning for pair in pairs for warning in pair["warnings"]]
    report = {
        "expected_width_inches": args.width_inches,
        "pairs": pairs,
        "errors": errors,
        "warnings": warnings,
        "visual_inspection": "required separately",
    }
    output = ROOT / "reports" / "PLOT_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

