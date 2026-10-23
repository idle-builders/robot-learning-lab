"""Build a compact native evaluation and paired opened-start explanation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-image-goal-20261010"


def read(relative: str) -> dict:
    return json.loads((OUT / relative).read_text())


def main() -> None:
    development = read("yaw15-development/matrix.json")
    reserved = read("yaw15-reserved/matrix.json")
    cases = reserved["cases"]
    assert len(development["cases"]) == 4 and len(cases) == 12
    reports = [read(f"yaw15-reserved/seed-{c['seed']}.json") for c in cases]
    summary = {
        "schema": "robot-learning-lab.cove50-virtual-rgb-yaw15-summary.v1",
        "development_plan": str((OUT / "yaw15-development-plan.json").relative_to(ROOT)),
        "reserved_plan": str((OUT / "yaw15-reserved-plan.json").relative_to(ROOT)),
        "development_pass_count": development["native_pass_count"],
        "development_sample_count": development["native_sample_count"],
        "reserved_pass_count": reserved["native_pass_count"],
        "reserved_sample_count": reserved["native_sample_count"],
        "reserved_contact_range_s": [min(c["contact_s"] for c in cases),
                                     max(c["contact_s"] for c in cases)],
        "reserved_entry_error_range_mm": [min(c["entry_goal_error_mm_scoring_only"] for c in cases),
                                          max(c["entry_goal_error_mm_scoring_only"] for c in cases)],
        "reserved_entry_orange_pixel_range": [min(c["entry_orange_pixels"] for c in cases),
                                              max(c["entry_orange_pixels"] for c in cases)],
        "reserved_reach_two_foot_range": [min(c["reach_both_feet_fraction"] for c in cases),
                                          max(c["reach_both_feet_fraction"] for c in cases)],
        "reserved_reach_rated_leg_range": [min(c["reach_rated_leg_fraction"] for c in cases),
                                           max(c["reach_rated_leg_fraction"] for c in cases)],
        "peak_head_yaw_effort_nm": round(max(abs(row["head_yaw_effort_nm"])
                                              for report in reports for row in report["timeseries"]), 5),
        "missing_goal_count": sum(r["image_goal_missing_at_s"] is not None for r in reports),
        "entry_goal_update_count_range": [min(r["image_goal_used_update_count"] for r in reports),
                                          max(r["image_goal_used_update_count"] for r in reports)],
        "same_start_9973": {
            "truth_goal_contact_s": read("diagnostics/seed-9973-truth-goal.json")["phases"]["reach"]["continuous_finger_contact_s"],
            "rgb_yaw13_contact_s": read("heldout/seed-9973.json")["phases"]["reach"]["continuous_finger_contact_s"],
            "rgb_yaw15_contact_s": read("yaw15-development/seed-9973.json")["phases"]["reach"]["continuous_finger_contact_s"],
            "rgb_yaw13_entry_error_mm": read("heldout/seed-9973.json")["image_goal_capture_events"][0]["xy_error_truth_mm_scoring_only"],
            "rgb_yaw15_entry_error_mm": read("yaw15-development/seed-9973.json")["image_goal_capture_events"][0]["xy_error_truth_mm_scoring_only"],
        },
        "qualification": "Fresh no-reset native selected-body reach with saved arm residual and goal derived from an unmounted massless virtual RGB camera at exact simulator capture pose and known floor plane. Fixed 7.5 s entry goal carries through later occlusion. No selected camera or physical pinch/possession.",
    }
    (OUT / "yaw15-report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(13, 4.7))
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce4ea", linewidth=.7)
    ax.set_axisbelow(True)
    x = np.arange(len(cases))
    colors = ["#198f71" if c["pass"] else "#d3654f" for c in cases]
    ax.bar(x, [c["contact_s"] for c in cases], color=colors)
    ax.axhline(1, color="#284457", ls="--", lw=1.3, label="1 s contact gate")
    for i, c in enumerate(cases):
        ax.text(i, c["contact_s"] + .08, f"{c['contact_s']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [str(c["seed"]) for c in cases])
    ax.set_ylim(0, max(c["contact_s"] for c in cases) + .7)
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_xlabel("Fresh native seed · 11–12 V, 0–10 ms, toy offset within ±6 mm")
    ax.legend(frameon=False)
    ax.set_title(f"Cove-50 · virtual RGB goal at motor-attained −1.5 rad · {reserved['native_pass_count']}/12 fresh gates",
                 fontsize=14, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "yaw15-evaluation.png", dpi=160)
    plt.close(fig)

    paired = summary["same_start_9973"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    fig.patch.set_facecolor("#f7f9fb")
    names = ["Truth goal", "RGB −1.3", "RGB −1.5"]
    contacts = [paired["truth_goal_contact_s"], paired["rgb_yaw13_contact_s"],
                paired["rgb_yaw15_contact_s"]]
    axes[0].bar(names, contacts, color=["#71859a", "#d3654f", "#198f71"])
    axes[0].axhline(1, color="#284457", ls="--", lw=1.2)
    axes[0].set_ylabel("Continuous contact (s)")
    axes[0].set_ylim(0, 3.7)
    for i, value in enumerate(contacts):
        axes[0].text(i, value + .08, f"{value:.2f}", ha="center")
    pixels = [read("heldout/seed-9973.json")["image_goal_capture_events"][0]["orange_pixels"],
              read("yaw15-development/seed-9973.json")["image_goal_capture_events"][0]["orange_pixels"]]
    errors = [paired["rgb_yaw13_entry_error_mm"], paired["rgb_yaw15_entry_error_mm"]]
    axes[1].bar(["RGB −1.3", "RGB −1.5"], errors, color=["#d3654f", "#198f71"])
    axes[1].set_ylim(0, 10.5)
    axes[1].set_ylabel("Entry XY error (mm), truth scoring only")
    for i, (value, px) in enumerate(zip(errors, pixels)):
        axes[1].text(i, value + .2, f"{value:.2f} mm · {px} px", ha="center")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dce4ea", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Opened seed 9973 · same start, changed goal source or legal gaze angle",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "yaw15-opened-start-comparison.png", dpi=160)
    plt.close(fig)

    font_path = ("/System/Library/Fonts/Supplemental/Arial.ttf"
                 if Path("/System/Library/Fonts/Supplemental/Arial.ttf").exists()
                 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, 21) if Path(font_path).exists() else ImageFont.load_default()
    sheet = Image.new("RGB", (1280, 900), "#f7f9fb")
    draw = ImageDraw.Draw(sheet)
    for i, case in enumerate(cases):
        col, row = i % 4, i // 4
        x, y = col * 320, row * 300
        sheet.paste(Image.open(OUT / f"yaw15-reserved/seed-{case['seed']}-entry.png").convert("RGB"), (x, y))
        draw.rectangle((x, y + 240, x + 319, y + 299), fill="#142338")
        status = "PASS" if case["pass"] else "FAIL"
        draw.text((x + 8, y + 245), f"{case['seed']}  {status}  {case['contact_s']:.2f}s",
                  font=font, fill="#72e2b0" if case["pass"] else "#ff9d87")
        draw.text((x + 8, y + 270),
                  f"{case['entry_orange_pixels']} px  {case['entry_goal_error_mm_scoring_only']:.2f} mm",
                  font=font, fill="#dbe8f1")
    sheet.save(OUT / "yaw15-input-contact-sheet.png")
    print(json.dumps({k: summary[k] for k in ("development_pass_count", "reserved_pass_count",
                                             "reserved_sample_count", "reserved_contact_range_s")}, indent=2))


if __name__ == "__main__":
    main()
