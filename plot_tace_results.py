#!/usr/bin/env python3
"""Create publication subfigures for the TACE reliability analysis.

The figure intentionally keeps the method diagram separate from the empirical
trade-offs.  It never reads a model checkpoint or performs inference; all
numbers come from the cross-fitted TACE CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATION = SCRIPT_DIR / "results" / "final_benchmark" / "tace_analysis"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "figures" / "tace"
TACE_F1 = "TACE-F1 (cross-fitted empirical-Bayes ensemble)"
TACE_MACRO = "TACE-Macro (cross-fitted empirical-Bayes ensemble)"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required analysis result: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing {field!r} in row: {row}")
    return float(value)


def configure_style(matplotlib: Any) -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.3,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.70,
        }
    )


def rounded_box(ax: Any, x: float, y: float, width: float, height: float, text: str, color: str, text_color: str = "#1F1F1F") -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=0.9, edgecolor=color, facecolor="#FFFFFF",
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0, y + height / 2.0, text,
        transform=ax.transAxes, ha="center", va="center", fontsize=7.7,
        color=text_color, linespacing=1.18,
    )


def arrow(ax: Any, x0: float, y0: float, x1: float, y1: float, color: str = "#6B7280") -> None:
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": color, "shrinkA": 2, "shrinkB": 2},
    )


def method_panel(ax: Any) -> None:
    ax.axis("off")
    navy, orange, teal, gray = "#1F4E79", "#B45F06", "#0F766E", "#6B7280"
    rounded_box(ax, 0.02, 0.56, 0.19, 0.24, "Frozen outputs\nQwen / CodeLlama / Mistral", navy)
    rounded_box(ax, 0.28, 0.56, 0.18, 0.24, "Class-wise\nagreement pattern\n" r"$z \in \{0,1\}^3$", teal)
    rounded_box(ax, 0.53, 0.56, 0.19, 0.24, "Empirical-Bayes\nposterior\n" r"$\hat p(y_c=1\mid z,d,c)$", orange)
    rounded_box(ax, 0.79, 0.56, 0.18, 0.24, "Nested 5x3\nstratified folds\n(no outer-fold tuning)", navy)
    for x0, x1 in ((0.21, 0.28), (0.46, 0.53), (0.72, 0.79)):
        arrow(ax, x0, 0.68, x1, 0.68, gray)
    rounded_box(ax, 0.31, 0.16, 0.17, 0.22, "TACE-F1\nOverall utility", navy)
    rounded_box(ax, 0.52, 0.16, 0.17, 0.22, "TACE-Macro\nTaxonomy breadth", orange)
    rounded_box(ax, 0.73, 0.16, 0.17, 0.22, "Selective decision\naccept / defer", teal)
    # All three decision products inherit the same outer-fold isolation.
    arrow(ax, 0.88, 0.56, 0.40, 0.38, gray)
    arrow(ax, 0.88, 0.56, 0.61, 0.38, gray)
    arrow(ax, 0.88, 0.56, 0.82, 0.38, gray)
    ax.text(
        0.50, 0.015,
        "Fixed model outputs -> calibrated class posterior -> auditable operating point",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.1, color="#4B5563",
    )


def performance_panel(ax: Any, summary: list[dict[str, str]]) -> None:
    systems = [
        "Qwen2.5-Coder-7B-Instruct", "CodeLlama-7B-Instruct", "Mistral-7B-Instruct-v0.3",
        "OR-3 ensemble", "Majority-3 ensemble", "AND-3 ensemble", TACE_F1,
    ]
    labels = ["Qwen", "CodeLlama", "Mistral", "OR", "Majority", "AND", "TACE-F1"]
    lookup = {(row["dataset"], row["system"]): row for row in summary}
    datasets = [("SmartBugs-Curated", "SmartBugs", "micro-F1"), ("ScrawlD", "ScrawlD", "micro-F1"), ("BCCC", "BCCC", "balanced accuracy")]
    offsets = [-0.17, 0.0, 0.17]
    colors = {"SmartBugs": "#0072B2", "ScrawlD": "#D55E00", "BCCC": "#009E73"}
    markers = {"SmartBugs": "o", "ScrawlD": "s", "BCCC": "^"}
    # A neutral band highlights the learned TACE operating point without
    # competing with the dataset colors used by the legend.
    ax.axvspan(5.62, 6.38, color="#F1F5F9", zorder=0)
    for offset, (dataset, short, metric_name) in zip(offsets, datasets):
        values = [as_float(lookup[(dataset, system)], "primary_value") for system in systems]
        x = [value + offset for value in range(len(systems))]
        ax.scatter(
            x, values, s=34, marker=markers[short], color=colors[short],
            edgecolor="white", linewidth=0.55, zorder=3,
            label=f"{short} ({metric_name})",
        )
    ax.set_ylim(-0.04, 1.05)
    ax.set_ylabel("Dataset primary metric")
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=2)
    ax.grid(axis="y")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
        fontsize=6.1, handletextpad=0.30, columnspacing=0.75, labelspacing=0.25,
    )
    ax.annotate(
        "ScrawlD 0.599",
        xy=(6.0, as_float(lookup[("ScrawlD", TACE_F1)], "primary_value")),
        xytext=(4.72, 0.70), fontsize=6.7, color=colors["ScrawlD"],
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": colors["ScrawlD"]},
    )


def tradeoff_panels(ax: Any, ax_risk: Any, summary: list[dict[str, str]], selective: list[dict[str, str]]) -> None:
    lookup = {(row["dataset"], row["system"]): row for row in summary}
    scrawl_f1 = lookup[("ScrawlD", TACE_F1)]
    scrawl_macro = lookup[("ScrawlD", TACE_MACRO)]
    metric_names = ["Precision", "Recall", "micro-F1", "macro-F1\n(supported)"]
    f1_values = [as_float(scrawl_f1, "precision"), as_float(scrawl_f1, "recall"), as_float(scrawl_f1, "f1"), as_float(scrawl_f1, "macro_f1_supported")]
    macro_values = [as_float(scrawl_macro, "precision"), as_float(scrawl_macro, "recall"), as_float(scrawl_macro, "f1"), as_float(scrawl_macro, "macro_f1_supported")]
    x = list(range(len(metric_names)))
    ax.bar([value - 0.18 for value in x], f1_values, width=0.34, color="#1F4E79", label="TACE-F1")
    ax.bar([value + 0.18 for value in x], macro_values, width=0.34, color="#B45F06", label="TACE-Macro")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, metric_names)
    ax.set_ylabel("ScrawlD OOF value")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=6.7)

    subset = [row for row in selective if row["dataset"] in ("SmartBugs-Curated", "ScrawlD")]
    color_by_dataset = {"SmartBugs-Curated": "#0F766E", "ScrawlD": "#7C3AED"}
    short = {"SmartBugs-Curated": "SmartBugs", "ScrawlD": "ScrawlD"}
    for dataset in ("SmartBugs-Curated", "ScrawlD"):
        rows = sorted((row for row in subset if row["dataset"] == dataset), key=lambda row: as_float(row, "decision_coverage"))
        ax_risk.plot(
            [as_float(row, "decision_coverage") for row in rows],
            [as_float(row, "accepted_decision_risk") for row in rows],
            marker="o", markersize=3.0, lw=1.0, color=color_by_dataset[dataset], label=short[dataset],
        )
    ax_risk.set_xlabel("decision coverage", fontsize=6.7, labelpad=2)
    ax_risk.set_ylabel("accepted risk", fontsize=6.7, labelpad=2)
    ax_risk.tick_params(labelsize=6.1, pad=1)
    ax_risk.grid(alpha=0.5)
    ax_risk.legend(fontsize=6.0, loc="upper right", frameon=False, handlelength=1.2)


def save_figure(fig: Any, base: Path, formats: Sequence[str], dpi: int, plt: Any) -> list[Path]:
    outputs: list[Path] = []
    for suffix in formats:
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_support_files(output_dir: Path) -> None:
    (output_dir / "FIGURE_CAPTION.md").write_text(
        "**Figure 2. Taxonomy-aware calibrated ensemble (TACE).** (a) TACE fuses "
        "three frozen code-LLM outputs through a class-specific empirical-Bayes "
        "agreement posterior. Calibration and operating thresholds are nested within "
        "five outer stratified folds; the outer fold is never used to fit either. "
        "(b) Out-of-fold primary metric for the individual models, three parameter-free "
        "agreement rules, and TACE-F1. The metric is micro-F1 on SmartBugs-Curated and "
        "ScrawlD, and balanced accuracy on BCCC. (c) On ScrawlD, TACE-F1 prioritizes "
        "overall detection utility whereas TACE-Macro prioritizes supported-class breadth; "
        "the lower plot reports fixed-confidence selective risk versus coverage. Static-analyzer "
        "results are intentionally omitted because ScrawlD labels are not independent of "
        "those analyzers.",
        encoding="utf-8",
    )
    (output_dir / "FIGURE_INCLUDE.tex").write_text(
        "\\begin{figure}[H]\n"
        "  \\centering\n"
        "  \\begin{subfigure}[t]{\\textwidth}\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\textwidth]{figures/fig02a_tace_method.pdf}\n"
        "    \\caption{Cross-fitted calibration and decision path.}\n"
        "    \\label{fig:tace-method}\n"
        "  \\end{subfigure}\n"
        "  \\vspace{0.5em}\n"
        "  \\begin{subfigure}[t]{0.49\\textwidth}\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{figures/fig02b_oof_effectiveness.pdf}\n"
        "    \\caption{Out-of-fold effectiveness.}\n"
        "    \\label{fig:tace-effectiveness}\n"
        "  \\end{subfigure}\\hfill\n"
        "  \\begin{subfigure}[t]{0.49\\textwidth}\n"
        "    \\centering\n"
        "    \\includegraphics[width=\\linewidth]{figures/fig02c_tace_tradeoff.pdf}\n"
        "    \\caption{Utility and selective risk--coverage.}\n"
        "    \\label{fig:tace-tradeoff}\n"
        "  \\end{subfigure}\n"
        "  \\caption{Taxonomy-aware calibrated ensemble (TACE). See the companion "
        "\\texttt{FIGURE\\_CAPTION.md} for the final checked caption.}\n"
        "  \\label{fig:tace-method-tradeoff}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=("pdf", "svg", "png"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi < 150:
        parser.error("--dpi must be at least 150")
    args.formats = tuple(dict.fromkeys(args.formats))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required; install requirements-visualization.txt.") from error
    configure_style(matplotlib)
    evaluation = args.evaluation_dir.resolve()
    summary = read_csv(evaluation / "tace_oof_summary.csv")
    selective = read_csv(evaluation / "tace_selective_risk_coverage.csv")
    expected = {TACE_F1, TACE_MACRO}
    observed = {row["system"] for row in summary}
    if not expected <= observed:
        raise ValueError(f"Incomplete TACE summary rows: missing {sorted(expected - observed)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_method, ax_method = plt.subplots(figsize=(7.2, 2.05))
    method_panel(ax_method)
    fig_method.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.06)

    fig_performance, ax_performance = plt.subplots(figsize=(4.55, 3.45))
    performance_panel(ax_performance, summary)
    fig_performance.subplots_adjust(left=0.15, right=0.985, top=0.97, bottom=0.35)

    fig_tradeoff = plt.figure(figsize=(4.55, 3.95))
    tradeoff_grid = fig_tradeoff.add_gridspec(2, 1, height_ratios=[0.66, 0.34], hspace=0.58)
    ax_tradeoff = fig_tradeoff.add_subplot(tradeoff_grid[0])
    ax_risk = fig_tradeoff.add_subplot(tradeoff_grid[1])
    tradeoff_panels(ax_tradeoff, ax_risk, summary, selective)
    fig_tradeoff.subplots_adjust(left=0.15, right=0.985, top=0.98, bottom=0.14)

    generated: list[Path] = []
    generated.extend(save_figure(fig_method, output_dir / "fig02a_tace_method", args.formats, args.dpi, plt))
    generated.extend(save_figure(fig_performance, output_dir / "fig02b_oof_effectiveness", args.formats, args.dpi, plt))
    generated.extend(save_figure(fig_tradeoff, output_dir / "fig02c_tace_tradeoff", args.formats, args.dpi, plt))
    write_support_files(output_dir)
    manifest = {
        "schema_version": "tace-subfigures",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "evaluation_dir": str(evaluation),
        "inputs": [
            str(evaluation / "tace_oof_summary.csv"),
            str(evaluation / "tace_selective_risk_coverage.csv"),
        ],
        "settings": {"formats": list(args.formats), "png_dpi": args.dpi, "matplotlib": matplotlib.__version__},
        "generated": [str(path) for path in generated],
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(generated)} TACE subfigure files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
