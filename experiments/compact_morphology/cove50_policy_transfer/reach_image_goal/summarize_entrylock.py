"""Summarize the frozen RGB entry-lock qualification and its retained miss."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-image-goal-20261010"


def main() -> None:
    matrix = json.loads((OUT / "heldout/matrix.json").read_text())
    cases = matrix["cases"]
    assert len(cases) == 8 and matrix["native_pass_count"] == sum(c["pass"] for c in cases)
    truth = json.loads((OUT / "diagnostics/seed-9973-truth-goal.json").read_text())
    summary = {
        "schema": "robot-learning-lab.cove50-virtual-rgb-entry-lock-summary.v1",
        "plan": str((OUT / "heldout-plan.json").relative_to(ROOT)),
        "matrix": str((OUT / "heldout/matrix.json").relative_to(ROOT)),
        "development_live_update_contact_s": json.loads((OUT / "development/seed-9921.json").read_text())["phases"]["reach"]["continuous_finger_contact_s"],
        "development_entry_lock_contact_s": json.loads((OUT / "development/seed-9921-entrylock.json").read_text())["phases"]["reach"]["continuous_finger_contact_s"],
        "heldout_pass_count": matrix["native_pass_count"],
        "heldout_sample_count": len(cases),
        "heldout_contact_range_s": [min(c["contact_s"] for c in cases), max(c["contact_s"] for c in cases)],
        "heldout_entry_error_range_mm": [min(c["entry_goal_error_mm_scoring_only"] for c in cases),
                                         max(c["entry_goal_error_mm_scoring_only"] for c in cases)],
        "failed_case": [c for c in cases if not c["pass"]],
        "same_start_truth_goal_contact_s_for_failed_case": truth["phases"]["reach"]["continuous_finger_contact_s"],
        "qualification": "Live native RGB goal drives the saved learned arm residual and authored entry-error route through a separate massless virtual camera. Exact simulated capture pose and known floor plane; no selected sensor, pinch, grasp or possession.",
    }
    (OUT / "entrylock-report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor("#f7f9fb")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dce4ea", linewidth=.7)
        ax.set_axisbelow(True)
    colors = ["#198f71" if c["pass"] else "#d3654f" for c in cases]
    labels = [str(c["seed"]) for c in cases]
    heights = [c["contact_s"] for c in cases]
    axes[0].bar(labels, heights, color=colors)
    axes[0].axhline(1, color="#284457", ls="--", lw=1.3, label="1 s contact gate")
    for i, height in enumerate(heights):
        axes[0].text(i, height + .09, f"{height:.2f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(heights) + .7)
    axes[0].set_ylabel("Longest continuous fingertip contact (s)")
    axes[0].set_xlabel("Fresh native seed")
    axes[0].legend(frameon=False, loc="upper left")
    for case, color in zip(cases, colors):
        axes[1].scatter(case["entry_goal_error_mm_scoring_only"], case["entry_orange_pixels"],
                        s=76, c=color, edgecolors="white", linewidths=.8, zorder=3)
        label_offset = {9972: (-28, 8), 9975: (8, -17),
                        9973: (5, 8), 9976: (5, -18)}.get(case["seed"], (5, 5))
        axes[1].annotate(str(case["seed"]), (case["entry_goal_error_mm_scoring_only"],
                          case["entry_orange_pixels"]), xytext=label_offset,
                         textcoords="offset points", fontsize=8)
    axes[1].axvline(10, color="#284457", ls="--", lw=1.3)
    axes[1].set_xlim(0, 11.5)
    axes[1].set_ylim(0, 135)
    axes[1].set_xlabel("Entry RGB goal XY error (mm), truth scoring only")
    axes[1].set_ylabel("Detected orange pixels")
    fig.suptitle("Cove-50 · entry-locked virtual RGB goal · 7/8 fresh native passes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "entrylock-evaluation.png", dpi=160)
    plt.close(fig)

    font_path = ("/System/Library/Fonts/Supplemental/Arial.ttf"
                 if Path("/System/Library/Fonts/Supplemental/Arial.ttf").exists()
                 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, 21) if Path(font_path).exists() else ImageFont.load_default()
    sheet = Image.new("RGB", (1280, 600), "#f7f9fb")
    draw = ImageDraw.Draw(sheet)
    for i, case in enumerate(cases):
        col, row = i % 4, i // 4
        x, y = col * 320, row * 300
        path = OUT / f"heldout/seed-{case['seed']}-entry.png"
        sheet.paste(Image.open(path).convert("RGB"), (x, y))
        draw.rectangle((x, y + 240, x + 319, y + 299), fill="#142338")
        status = "PASS" if case["pass"] else "FAIL"
        color = "#72e2b0" if case["pass"] else "#ff9d87"
        draw.text((x + 8, y + 245), f"{case['seed']}  {status}  {case['contact_s']:.2f}s",
                  font=font, fill=color)
        draw.text((x + 8, y + 270),
                  f"{case['entry_orange_pixels']} px  {case['entry_goal_error_mm_scoring_only']:.2f} mm",
                  font=font, fill="#dbe8f1")
    sheet.save(OUT / "entrylock-input-contact-sheet.png")
    print(json.dumps({k: summary[k] for k in ("heldout_pass_count", "heldout_sample_count",
                                             "heldout_contact_range_s", "failed_case")}, indent=2))


if __name__ == "__main__":
    main()
