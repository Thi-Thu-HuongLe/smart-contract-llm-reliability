#!/usr/bin/env python3
"""Render a standalone, vector-ready TACE contribution-framework figure.

This conceptual figure is intentionally independent of performance numbers.
Use it in the Introduction or Method section to state what is technically new:

  C1. class-specific empirical-Bayes reliability calibration;
  C2. nested, leakage-controlled operating-point selection; and
  C3. selective decisions with explicit coverage and risk.

It is a code-native vector figure (not an AI-generated illustration).
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
    / "tace_framework"
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


def box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    body: str,
    color: str,
    tag: str | None = None,
) -> None:
    from matplotlib.patches import FancyBboxPatch

    frame = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=1.15,
        edgecolor=color,
        facecolor=LIGHT,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(frame)
    if tag:
        tag_box = FancyBboxPatch(
            (x + 0.016, y + height - 0.079),
            0.075,
            0.052,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=0,
            edgecolor=color,
            facecolor=color,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(tag_box)
        ax.text(
            x + 0.0535, y + height - 0.053,
            tag, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="white", fontweight="bold",
        )
    heading_position = y + height * (0.58 if tag else 0.64)
    body_position = y + height * (0.20 if tag else 0.31)
    ax.text(
        x + width / 2,
        heading_position,
        heading,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=color,
        linespacing=1.1,
    )
    ax.text(
        x + width / 2,
        body_position,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#334155",
        linespacing=1.22,
    )


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], dashed: bool = False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.25 if not dashed else 0.9,
            "color": ARROW,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def draw_figure(plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(7.20, 3.15))
    ax.set_axis_off()

    # Frozen-output pipeline: all geometry is in figure coordinates so it
    # remains stable across PDF, SVG, and high-resolution PNG export.
    top_y, top_h, top_w = 0.70, 0.22, 0.176
    xs = [0.028, 0.229, 0.430, 0.631, 0.832]
    box(ax, xs[0], top_y, top_w, top_h, "Frozen model outputs", "Qwen\nCodeLlama\nMistral", SLATE)
    box(ax, xs[1], top_y, top_w, top_h, "Taxonomy encoding", "For each class $c$:\npattern $z_c\\in\\{0,1\\}^3$", TEAL)
    box(ax, xs[2], top_y, top_w, top_h, "Reliability posterior", "$\\hat p(y_c=1\\mid z_c,d,c)$\nEmpirical-Bayes shrinkage", ORANGE)
    box(ax, xs[3], top_y, top_w, top_h, "Nested cross-fitting", "5 outer x 3 inner folds\nNo outer-fold tuning", NAVY)
    box(ax, xs[4], top_y, top_w - 0.004, top_h, "Decision products", "TACE-F1 / TACE-Macro\nAccept or defer", TEAL)
    for index in range(4):
        arrow(ax, (xs[index] + top_w, top_y + top_h / 2), (xs[index + 1], top_y + top_h / 2))

    # Three explicitly labelled scientific contributions.
    lower_y, lower_h, lower_w = 0.17, 0.225, 0.267
    lower_xs = [0.063, 0.368, 0.673]
    box(
        ax, lower_xs[0], lower_y, lower_w, lower_h,
        "Class-aware fusion", "Learns whether each agreement\npattern is reliable for each\nvulnerability category.", TEAL, "C1",
    )
    box(
        ax, lower_xs[1], lower_y, lower_w, lower_h,
        "Evaluation without leakage", "Calibrator and threshold are\nchosen without the held-out\nrecord or outer fold.", NAVY, "C2",
    )
    box(
        ax, lower_xs[2], lower_y, lower_w, lower_h,
        "Risk-aware automation", "High-confidence decisions are\nautomated; uncertain class\ndecisions are deferred.", ORANGE, "C3",
    )
    arrow(ax, (xs[1] + top_w / 2, top_y), (lower_xs[0] + lower_w / 2, lower_y + lower_h), dashed=True)
    arrow(ax, (xs[3] + top_w / 2, top_y), (lower_xs[1] + lower_w / 2, lower_y + lower_h), dashed=True)
    arrow(ax, (xs[4] + top_w / 2, top_y), (lower_xs[2] + lower_w / 2, lower_y + lower_h), dashed=True)

    ax.text(
        0.5, 0.045,
        "Scope: post-inference decision layer over frozen outputs -- not a newly trained LLM.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        color="#475569",
        style="italic",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    return fig


def save_figure(fig: Any, output: Path, formats: Sequence[str], dpi: int, plt: Any) -> list[Path]:
    outputs: list[Path] = []
    for suffix in formats:
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_caption_files(output_dir: Path) -> None:
    (output_dir / "CAPTION.md").write_text(
        "**TACE methodological contribution framework.** TACE operates on three "
        "frozen code-LLM outputs. For every dataset and taxonomy class, it encodes "
        "the three model decisions as an agreement pattern and estimates a "
        "class-specific empirical-Bayes reliability posterior. Nested stratified "
        "cross-fitting prevents the held-out outer fold from affecting calibration "
        "or threshold selection. The resulting TACE-F1, TACE-Macro, and selective "
        "decision modes make both utility and uncertainty explicit.",
        encoding="utf-8",
    )
    (output_dir / "INCLUDE.tex").write_text(
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{figures/tace_contribution_framework.pdf}\n"
        "  \\caption{TACE methodological contribution framework.}\n"
        "  \\label{fig:tace-contribution-framework}\n"
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
        draw_figure(plt), output_dir / "tace_contribution_framework",
        args.formats, args.dpi, plt,
    )
    write_caption_files(output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "tace-contribution-framework",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": str(Path(__file__).resolve()),
                "settings": {"formats": list(args.formats), "png_dpi": args.dpi},
                "generated": [str(path) for path in generated],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(generated)} contribution-framework files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
