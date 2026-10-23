"""Prepare the frozen virtual view for the session inspector."""

from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageFont

from screen_head_mount import OUT


def main() -> None:
    screen = json.loads((OUT / "head-mount-screen.json").read_text())
    frozen = json.loads((OUT / "heldout-freeze.json").read_text())
    result = json.loads((OUT / "frozen-view-report.json").read_text())
    rows = []
    for case in screen["cases"]:
        rows.append({
            "case": case["case"],
            "featured": case["case"] == frozen["chosen_case"] or
                        (case["entry_gate"] and case["head_local_camera_xyz_m"][0] == .048),
            "extra_pitch_deg": case["extra_pitch_deg"],
            "xyz_m": case["head_local_camera_xyz_m"],
            "module_center_shift_mm": case["module_center_shift_mm"],
            "head_center_radius_mm": case["head_sphere_center_distance_mm"],
            "entry_gate": case["entry_gate"],
            "rgb_detected": case["rgb_entry_detected"],
            "rgb_under_10mm": case["rgb_entry_under_10mm"],
            "min_rgb_pixels": case["min_rgb_pixels"],
            "max_xy_error_mm": (case["max_xy_error_mm"] if
                                case["max_xy_error_mm"] < 1e5 else None),
            "min_margin_px": case["min_projected_center_margin_px"],
            "report": "artifacts/ape-cove50-reach-viewpoint-20261010/head-mount-screen.json",
        })
    (OUT / "head-mount-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-head-viewpoint-matrix.v1",
        "cases": rows}, indent=2, allow_nan=False) + "\n")

    labels = [
        ("Held-out seed 9951 · entry", "heldout-seed-9951-7.5s-frozen-head-view.png"),
        ("Held-out seed 9952 · entry", "heldout-seed-9952-7.5s-frozen-head-view.png"),
        ("Held-out seed 9957 · entry", "heldout-seed-9957-7.5s-frozen-head-view.png"),
        ("Held-out seed 9958 · entry", "heldout-seed-9958-7.5s-frozen-head-view.png"),
        ("Development · entry 7.5 s", "development-seed-9921-7.5s-frozen-head-view.png"),
        ("Development · occluded 10 s", "development-seed-9921-10s-frozen-head-view.png"),
    ]
    canvas = Image.new("RGB", (640, 3 * 270), "#e8edf2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (label, name) in enumerate(labels):
        x, y = i % 2 * 320, i // 2 * 270
        draw.text((x + 8, y + 7), label, fill="#18283a", font=font)
        with Image.open(OUT / name) as image:
            canvas.paste(image.convert("RGB"), (x, y + 25))
    canvas.save(OUT / "frozen-view-contact-sheet.png")
    print("heldout", result["heldout_entry_pass_count"], "/",
          result["heldout_entry_sample_count"], "virtual view only")


if __name__ == "__main__":
    main()
