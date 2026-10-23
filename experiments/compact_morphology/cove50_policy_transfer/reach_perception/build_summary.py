"""Condense retained Cove camera screens for the local results inspector."""

from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageFont

from screen_retained_rgb import OUT


def save(name: str, rows: list[dict], schema: str) -> None:
    (OUT / name).write_text(json.dumps({"schema": schema, "cases": rows},
                                       indent=2, allow_nan=False) + "\n")


def main() -> None:
    direct = json.loads((OUT / "report.json").read_text())
    early = json.loads((OUT / "early-visibility.json").read_text())
    pitch = json.loads((OUT / "pitch-sensitivity.json").read_text())
    first = []
    for case in direct["cases"]:
        seed = case["case"]
        early_rows = [r for r in early["rows"] if r["trace"].endswith(f"{seed}.npz")]
        first.append({
            "case": seed, "featured": True,
            "saved_entry_visible": case["direct_entry_center_visible"],
            "saved_reach_visible_fraction": case["direct_reach_visible_fraction"],
            "best_pre_entry_legal_yaw_margin_px": max(
                r["best_legal_yaw_projection"]["margin_px"] for r in early_rows),
            "legal_yaw_entry_visible_upper_bound": case["legal_yaw_entry_visible_upper_bound"],
            "saved_rgb_detection_frames": sum(any(v["orange_bbox"] is not None
                for v in row["direct_rgb"].values()) for row in case["rows"]),
            "report": "artifacts/ape-cove50-reach-camera-20261010/report.json",
        })
    save("selected-camera-matrix.json", first,
         "robot-learning-lab.cove50-selected-reach-camera-matrix.v1")
    second = []
    for case in pitch["cases"]:
        render = next(r for r in pitch["rendered_cases"] if r["case"] == case["case"])
        entry = [r for r in render["rows"] if r["time_s"] == 7.5]
        errors = [r["xy_error_mm"] for r in render["rows"]
                  if r["xy_error_mm"] is not None]
        second.append({
            "case": case["case"],
            "featured": case["case"] in ("pitch-0-yaw--1.5", "pitch-10-yaw--1.5",
                                        "pitch-15-yaw--1.5"),
            "extra_pitch_deg": case["additional_downward_pitch_deg"],
            "fixed_head_yaw_rad": case["head_yaw_target_rad"],
            "min_center_margin_px": case["minimum_center_edge_margin_px"],
            "center_in_frame_samples": case["center_visible_samples"],
            "rgb_entry_detected": sum(r["orange_bbox"] is not None for r in entry),
            "rgb_detected_samples": render["rgb_detected_samples"],
            "rgb_under_10mm_samples": render["rgb_error_under_10mm_samples"],
            "largest_measured_error_mm": max(errors) if errors else None,
            "all_20_rgb_gate": render["all_20_rgb_gate"],
            "report": "artifacts/ape-cove50-reach-camera-20261010/pitch-sensitivity.json",
        })
    save("pitch-sensitivity-matrix.json", second,
         "robot-learning-lab.cove50-reach-camera-pitch-matrix.v1")

    labels = ["Selected lower RGB; saved yaw", "Virtual +10 deg; yaw -1.5",
              "Virtual +15 deg; yaw -1.5"]
    seeds = (9921, 9942)
    canvas = Image.new("RGB", (3 * 320, 2 * 280 + 36), "#e8edf2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(labels):
        draw.text((column * 320 + 8, 8), label, fill="#18283a", font=font)
    for row, seed in enumerate(seeds):
        for column in range(3):
            if column == 0:
                path = OUT / f"seed-{seed}-lower_rgb-7.5s-saved-head.png"
            else:
                angle = 10 if column == 1 else 15
                path = OUT / f"seed-{seed}-lower-pitch{angle}-yaw-1.5-7.5s.png"
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB"), (column * 320, 36 + row * 280))
            draw.text((column * 320 + 8, 36 + row * 280 + 246),
                      f"development seed {seed} / reach entry 7.5 s",
                      fill="#18283a", font=font)
    canvas.save(OUT / "reach-camera-observability.png")
    print("selected 0/4, legal yaw entry 0/4; virtual pitch cases",
          len(second), "best image 10/20 detected, 9/20 <=10 mm")


if __name__ == "__main__":
    main()
