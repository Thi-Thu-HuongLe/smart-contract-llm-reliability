#!/usr/bin/env python3
"""Render the overall three-contribution framework for the revised paper.

TACE is deliberately shown only as Contribution 2.  The figure separates:

  C1. structured, failure-aware code-LLM inference;
  C2. TACE taxonomy-aware calibrated decision fusion; and
  C3. an audit-aware evaluation and reporting protocol.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "outputs"
    / "figures"
    / "study_framework"
)

NAVY = "#1F4E79"
TEAL = "#0F766E"
ORANGE = "#B45F06"
SLATE = "#475569"
LIGHT = "#F8FAFC"
ARROW = "#64748B"


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
            "font.size": 8.3,
        }
    )


def rounded_box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    body: str,
    color: str,
    tag: str | None = None,
    facecolor: str = LIGHT,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    frame = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=1.15, edgecolor=color, facecolor=facecolor,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(frame)
    if tag:
        badge = FancyBboxPatch(
            (x + 0.017, y + height - 0.078), 0.075, 0.051,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            linewidth=0, facecolor=color, transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(badge)
        ax.text(
            x + 0.0545, y + height - 0.052, tag, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="white",
        )
    heading_y = y + height * (0.49 if tag else 0.62)
    body_y = y + height * (0.22 if tag else 0.30)
    ax.text(
        x + width / 2, heading_y, heading, transform=ax.transAxes,
        ha="center", va="center", fontsize=9.0, fontweight="bold",
        color=color, linespacing=1.10,
    )
    ax.text(
        x + width / 2, body_y, body, transform=ax.transAxes,
        ha="center", va="center", fontsize=7.25, color="#334155", linespacing=1.22,
    )


def arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    dashed: bool = False,
) -> None:
    ax.annotate(
        "", xy=end, xytext=start, xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.3 if not dashed else 1.0,
            "linestyle": "--" if dashed else "-",
            "color": ARROW,
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def routed_arrow(
    ax: Any,
    points: Sequence[tuple[float, float]],
    dashed: bool = False,
) -> None:
    """Draw an orthogonal route and put the only arrowhead at the final node."""

    if len(points) < 2:
        raise ValueError("A routed arrow requires at least two points")
    line_style = "--" if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            transform=ax.transAxes, color=ARROW,
            linewidth=1.0 if dashed else 1.3, linestyle=line_style,
            solid_capstyle="round", clip_on=False,
        )
    arrow(ax, points[-2], points[-1], dashed=dashed)


def draw_figure(plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(7.20, 4.15))
    ax.set_axis_off()

    # Explicit inputs. The final reference box is visually and semantically
    # separated because its labels are used only in calibration/evaluation.
    input_y, input_h, input_w = 0.805, 0.155, 0.222
    input_xs = [0.018, 0.266, 0.514, 0.762]
    rounded_box(
        ax, input_xs[0], input_y, input_w, input_h,
        "Three benchmarks",
        "SmartBugs-Curated: 143 (multilabel)\nScrawlD: 5,664 (multilabel)\nBCCC: 19,512 (balanced binary)",
        TEAL, facecolor="#F1F5F9",
    )
    rounded_box(
        ax, input_xs[1], input_y, input_w, input_h,
        "Three frozen code LLMs",
        "Qwen2.5-Coder-7B\nCodeLlama-7B\nMistral-7B-Instruct-v0.3",
        ORANGE, facecolor="#F1F5F9",
    )
    rounded_box(
        ax, input_xs[2], input_y, input_w, input_h,
        "Runtime controls",
        "Pinned revisions\nModel-native chat templates\nClosed taxonomy + JSON schema",
        SLATE, facecolor="#F1F5F9",
    )
    rounded_box(
        ax, input_xs[3], input_y, input_w - 0.004, input_h,
        "Evaluation reference",
        "Ground-truth labels\nRecord identities + manifests\nAnalysis only; never LLM input",
        NAVY, facecolor="#F1F5F9",
    )

    # C1 and C2 form the proposed inference/decision path. C3 is placed below
    # them because it audits both the raw C1 outputs and the calibrated C2
    # outputs; this avoids the false implication that only TACE is evaluated.
    upper_y, upper_h, upper_w = 0.480, 0.205, 0.360
    c1_x, c2_x = 0.080, 0.560
    rounded_box(
        ax, c1_x, upper_y, upper_w, upper_h,
        "Structured inference & repair",
        "Source-code chunking + model-native prompting\nClosed-taxonomy JSON validation\nTargeted selective repair for malformed\nor partial outputs; untouched valid records.",
        TEAL, "C1",
    )
    rounded_box(
        ax, c2_x, upper_y, upper_w, upper_h,
        "TACE decision layer",
        "Per-class 3-bit agreement patterns\nEmpirical-Bayes reliability posterior\nNested 5x3 cross-fitted thresholds\nTACE-F1 / TACE-Macro / accept-defer.",
        ORANGE, "C2",
    )

    c3_x, c3_y, c3_w, c3_h = 0.170, 0.245, 0.660, 0.155
    rounded_box(
        ax, c3_x, c3_y, c3_w, c3_h,
        "Audit-aware evaluation",
        "Compare raw LLMs, voting rules, and TACE  |  support-aware per-class reporting\nRepair sensitivity + paired bootstrap CIs  |  selective risk-coverage and benchmark limitations",
        NAVY, "C3",
    )

    # Solid runtime path: benchmark source, model checkpoints, and controls feed
    # C1. Evaluation labels follow dashed paths to C2/C3 only.
    arrow(ax, (input_xs[0] + input_w / 2, input_y), (c1_x + upper_w * 0.25, upper_y + upper_h))
    arrow(ax, (input_xs[1] + input_w / 2, input_y), (c1_x + upper_w * 0.50, upper_y + upper_h))
    arrow(ax, (input_xs[2] + input_w / 2, input_y), (c1_x + upper_w * 0.75, upper_y + upper_h))
    arrow(
        ax,
        (input_xs[3] + (input_w - 0.004) / 2, input_y),
        (c2_x + upper_w * 0.72, upper_y + upper_h),
        dashed=True,
    )
    routed_arrow(
        ax,
        (
            (input_xs[3] + input_w - 0.008, input_y + 0.015),
            (0.978, 0.710),
            (0.978, c3_y + c3_h / 2),
            (c3_x + c3_w, c3_y + c3_h / 2),
        ),
        dashed=True,
    )

    # C1 generates the frozen raw model predictions used by TACE. Both raw and
    # calibrated outputs are then audited in C3.
    arrow(ax, (c1_x + upper_w, upper_y + upper_h / 2), (c2_x, upper_y + upper_h / 2))
    ax.text(
        0.500, upper_y + upper_h / 2 + 0.028, "validated raw predictions",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.1, color=SLATE,
    )
    arrow(ax, (c1_x + upper_w * 0.68, upper_y), (c3_x + c3_w * 0.34, c3_y + c3_h))
    arrow(ax, (c2_x + upper_w * 0.32, upper_y), (c3_x + c3_w * 0.66, c3_y + c3_h))
    ax.text(
        0.315, 0.425, "raw outputs", transform=ax.transAxes,
        ha="center", va="center", fontsize=6.1, color=SLATE,
    )
    ax.text(
        0.685, 0.425, "TACE outputs", transform=ax.transAxes,
        ha="center", va="center", fontsize=6.1, color=SLATE,
    )

    rounded_box(
        ax, 0.105, 0.060, 0.790, 0.115,
        "Auditable results and evidence",
        "Validated JSONL + metadata  |  complete OOF TACE predictions/folds  |  summary and per-class metrics\nPaired CIs  |  repair sensitivity  |  selective risk-coverage  |  reproducibility manifest",
        SLATE, facecolor="#F1F5F9",
    )
    arrow(ax, (c3_x + c3_w / 2, c3_y), (0.50, 0.175))
    ax.text(
        0.5, 0.016,
        "Solid arrows: inference/analysis flow. Dashed arrows: ground truth used only for calibration and evaluation.",
        transform=ax.transAxes, ha="center", va="center", fontsize=7.1, color=SLATE, style="italic",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    return fig


def save_figure(fig: Any, output: Path, formats: Sequence[str], dpi: int, plt: Any) -> list[Path]:
    generated: list[Path] = []
    for suffix in formats:
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        generated.append(path)
    plt.close(fig)
    return generated


def write_support_files(output_dir: Path) -> None:
    (output_dir / "CAPTION.md").write_text(
        "**Proposed framework and methodological contributions.** The evaluation uses "
        "three benchmarks--SmartBugs-Curated (143 records), ScrawlD (5,664 records), and "
        "balanced BCCC (19,512 records)--and three frozen 7B code LLMs: "
        "Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct, and "
        "Mistral-7B-Instruct-v0.3. Contribution C1 is a structured, failure-aware "
        "inference protocol using model-native prompting, closed-taxonomy JSON validation, "
        "and targeted selective repair. Contribution C2 is TACE, a taxonomy-aware "
        "calibrated ensemble and selective decision layer. Contribution C3 is an "
        "audit-aware protocol with manifest-backed identities, support-aware reporting, "
        "repair sensitivity, paired bootstrap confidence intervals, and risk-coverage "
        "analysis. Solid arrows denote runtime/output flow; dashed arrows show that "
        "ground-truth labels are used only for calibration and evaluation, never as model input.",
        encoding="utf-8",
    )
    (output_dir / "INCLUDE.tex").write_text(
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/paper_contribution_framework.pdf}\n"
        "  \\caption{Overall methodological contributions of the revised framework.}\n"
        "  \\label{fig:paper-contribution-framework}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = save_figure(
        draw_figure(plt), output_dir / "paper_contribution_framework",
        args.formats, args.dpi, plt,
    )
    write_support_files(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper-contribution-framework",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": str(Path(__file__).resolve()),
                "settings": {"formats": list(args.formats), "png_dpi": args.dpi},
                "generated": [str(path) for path in generated],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} paper-contribution-framework files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
