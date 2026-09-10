#!/usr/bin/env python3
"""Plot nested cross-fitted recovery baselines without LLM inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = (
    SCRIPT_DIR
    / "results"
    / "final_benchmark"
    / "recovery_baseline_analysis"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "outputs"
    / "figures"
    / "tace"
    / "fig02b_oof_effectiveness.pdf"
)
SYSTEMS = [
    "Prevalence-only (nested cross-fitted)",
    "TACE-F1 (cross-fitted empirical-Bayes ensemble)",
    "Logistic stacker (nested cross-fitted)",
]
SYSTEM_LABELS = ["Prevalence only", "TACE", "Logistic stacker"]
DATASETS = ["SmartBugs-Curated", "ScrawlD", "BCCC"]
DATASET_LABELS = ["SmartBugs", "ScrawlD", "BCCC"]
COLORS = ["#6B7280", "#0072B2", "#D55E00"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_pdf(rows: list[dict[str, str]], output: Path) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    lookup = {(row["dataset"], row["system"]): float(row["primary_value"]) for row in rows}
    missing = [
        (dataset, system)
        for dataset in DATASETS
        for system in SYSTEMS
        if (dataset, system) not in lookup
    ]
    if missing:
        raise ValueError(f"Missing recovery rows: {missing}")

    width, height = 4.55 * 72.0, 3.45 * 72.0
    pdf = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    pdf.setTitle("Nested cross-fitted recovery baselines")
    left, right = 93.0, 8.0
    plot_width = width - left - right
    panel_height, panel_gap = 68.0, 11.0
    panel_tops = [
        height - 7.0,
        height - 7.0 - panel_height - panel_gap,
        height - 7.0 - 2 * (panel_height + panel_gap),
    ]
    limits = [(0.34, 0.98), (0.594, 0.602), (0.496, 0.523)]
    ticks = [
        [0.4, 0.6, 0.8],
        [0.594, 0.598, 0.602],
        [0.50, 0.51, 0.52],
    ]

    for panel_top, dataset, dataset_label, xlim, xticks in zip(
        panel_tops, DATASETS, DATASET_LABELS, limits, ticks
    ):
        values = [lookup[(dataset, system)] for system in SYSTEMS]
        x_min, x_max = xlim

        def x_at(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_width

        baseline_y = panel_top - 57.0
        pdf.setFillColor(HexColor("#1F2937"))
        pdf.setFont("Helvetica-Bold", 7.5)
        pdf.drawString(2.0, panel_top - 8.0, dataset_label)

        for tick in xticks:
            x = x_at(tick)
            pdf.setStrokeColor(HexColor("#D1D5DB"))
            pdf.setLineWidth(0.35)
            pdf.line(x, panel_top - 13.0, x, baseline_y)
            pdf.setFillColor(HexColor("#4B5563"))
            pdf.setFont("Helvetica", 5.8)
            tick_text = f"{tick:.3f}" if (x_max - x_min) < 0.1 else f"{tick:.1f}"
            pdf.drawCentredString(x, baseline_y - 8.0, tick_text)

        for index, (label, value, color) in enumerate(zip(SYSTEM_LABELS, values, COLORS)):
            y = panel_top - 19.0 - index * 13.0
            pdf.setFillColor(HexColor("#374151"))
            pdf.setFont("Helvetica", 6.2)
            pdf.drawRightString(left - 5.0, y + 1.5, label)
            value_x = x_at(value)
            pdf.setStrokeColor(HexColor("#CBD5E1"))
            pdf.setLineWidth(0.6)
            pdf.line(left, y + 2.0, value_x, y + 2.0)
            pdf.setFillColor(HexColor(color))
            pdf.circle(value_x, y + 2.0, 3.2, fill=1, stroke=0)
            value_text = f"{value:.4f}"
            pdf.setFont("Helvetica-Bold", 5.8)
            text_width = stringWidth(value_text, "Helvetica-Bold", 5.8)
            if value_x + text_width + 3.0 <= width - right:
                pdf.setFillColor(HexColor("#111827"))
                pdf.drawString(value_x + 2.0, y, value_text)
            else:
                pdf.setFillColor(HexColor("#111827"))
                pdf.drawRightString(value_x - 4.8, y, value_text)

        metric = "micro-F1" if dataset != "BCCC" else "balanced accuracy"
        pdf.setFillColor(HexColor("#4B5563"))
        pdf.setFont("Helvetica", 5.9)
        pdf.drawCentredString(left + plot_width / 2.0, baseline_y - 17.0, metric)

    pdf.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must use the .pdf extension")

    rows = read_rows(args.analysis_dir / "recovery_baseline_summary.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(rows, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
