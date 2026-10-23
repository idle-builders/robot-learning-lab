"""Summarize frozen motor-attained gaze, native reach, and virtual RGB evidence."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from screen_head_mount import OUT, ROOT


def main() -> None:
    native_dir = OUT / "native-gaze"
    fast = json.loads((native_dir / "dev-9921.json").read_text())
    slow = json.loads((native_dir / "dev-9921-slow.json").read_text())
    dev_image = json.loads((native_dir / "dev-9921-slow-attained-view-report.json").read_text())
    original = json.loads((ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/development/seed-9921.json").read_text())
    development = []
    for label, report, view in (("saved no-gaze", original, None),
                                ("5–7 s ramp", fast, json.loads((native_dir / "dev-9921-attained-view-report.json").read_text())),
                                ("0–7 s ramp", slow, dev_image)):
        entry = next(r for r in report["timeseries"] if r["time_s"] == 7.5)
        image_entry = None if view is None else next(r for r in view["cases"] if r["time_s"] == 7.5)
        development.append({
            "case": label, "pass": label == "0–7 s ramp",
            "entry_head_yaw_rad": entry.get("head_yaw_rad", 0),
            "entry_image_pass": None if image_entry is None else image_entry["pass"],
            "entry_orange_pixels": None if image_entry is None or image_entry["orange_bbox"] is None else image_entry["orange_bbox"]["pixels"],
            "entry_xy_error_mm": None if image_entry is None else image_entry["xy_error_mm"],
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
            "first_lost_feet_s": next((r["time_s"] for r in report["timeseries"]
                                       if not all(r["feet_grounded"])), None),
            "max_head_yaw_effort_nm": max((abs(r["head_yaw_effort_nm"]) for r in report["timeseries"]
                                           if "head_yaw_effort_nm" in r), default=None),
            "result": ("artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/development/seed-9921.json"
                       if label == "saved no-gaze" else str((native_dir / ("dev-9921.json" if label == "5–7 s ramp"
                                                                             else "dev-9921-slow.json")).relative_to(ROOT))),
        })
    (native_dir / "development-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-gaze-development-matrix.v1",
        "cases": development}, indent=2, allow_nan=False) + "\n")

    rows = []
    images = []
    for split in ("heldout", "confirmation"):
        matrix = json.loads((native_dir / split / "matrix.json").read_text())
        assert matrix["native_pass_count"] == 4 and matrix["native_sample_count"] == 4
        for native in matrix["cases"]:
            seed = native["seed"]
            view = json.loads((native_dir / f"seed-{seed}-attained-view-report.json").read_text())
            entry = next(r for r in view["cases"] if r["time_s"] == 7.5)
            baseline = json.loads((ROOT / f"artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/reserved/seed-{seed}.json").read_text())
            baseline_contact = baseline["phases"]["reach"]["continuous_finger_contact_s"]
            rows.append({
                "case": f"seed-{seed}", "split": split, "rail_v": native["rail_v"],
                "delay_ms": native["delay_ms"], "toy_offset_b_m": [native["toy_offset_x_m"], native["toy_offset_y_m"]],
                "pass": bool(native["pass"] and entry["pass"]),
                "native_pass": native["pass"], "image_pass": entry["pass"],
                "entry_head_yaw_rad": native["entry_head_yaw_rad"],
                "entry_orange_pixels": entry["orange_bbox"]["pixels"] if entry["orange_bbox"] else 0,
                "entry_center_margin_px": entry["projection"]["pixel_edge_margin"],
                "entry_xy_error_mm": entry["xy_error_mm"],
                "contact_s": native["contact_s"],
                "baseline_no_gaze_contact_s": baseline_contact,
                "contact_change_s": round(native["contact_s"] - baseline_contact, 2),
                "reach_both_feet_fraction": native["reach_both_feet_fraction"],
                "reach_rated_leg_fraction": native["reach_rated_leg_fraction"],
                "native_result": native["result"],
                "image_result": str((native_dir / f"seed-{seed}-attained-view-report.json").relative_to(ROOT)),
                "entry_image": entry["image"],
            })
            images.append((seed, ROOT / entry["image"], entry["xy_error_mm"], native["contact_s"]))
    result = {
        "schema": "robot-learning-lab.cove50-native-attained-gaze-summary.v1",
        "classification": "Native motor-attained no-reset truth-goal reach with frozen massless virtual camera; not selected camera hardware or image-driven policy",
        "selected_pointer_sha256": json.loads((OUT / "gaze-native-plan.json").read_text())["selected_pointer_sha256"],
        "selected_body_sha256": json.loads((OUT / "gaze-native-plan.json").read_text())["selected_body_sha256"],
        "saved_reach_actor_sha256": json.loads((OUT / "gaze-native-plan.json").read_text())["saved_reach_actor_sha256"],
        "gaze_evaluator_sha256": json.loads((OUT / "gaze-native-plan.json").read_text())["gaze_evaluator_sha256"],
        "fast_ramp_failure": "5–7 s command attains yaw but falls by 6.94 s; zero reach contact",
        "slow_ramp_development": "0–7 s command attains yaw; 3.86 s reach contact and 100% two-foot support; virtual entry RGB pass",
        "native_and_entry_image_pass_count": sum(r["pass"] for r in rows),
        "heldout_sample_count": len(rows),
        "minimum_contact_s": min(r["contact_s"] for r in rows),
        "minimum_entry_orange_pixels": min(r["entry_orange_pixels"] for r in rows),
        "maximum_entry_xy_error_mm": max(r["entry_xy_error_mm"] for r in rows),
        "minimum_entry_center_margin_px": min(r["entry_center_margin_px"] for r in rows),
        "ten_second_occlusion": "All four original heldout 10 s images abstain; contact timing can outlast visibility",
        "mount_status": "Virtual massless moved camera only; modeled module has no bracket to head sphere; selected RGB remains nonobservable",
        "development": development,
        "cases": rows,
    }
    (native_dir / "report.json").write_text(json.dumps(result, indent=2,
                                                     allow_nan=False) + "\n")
    (native_dir / "evaluation-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-gaze-evaluation-matrix.v1",
        "cases": rows}, indent=2, allow_nan=False) + "\n")

    canvas = Image.new("RGB", (4 * 324, 2 * 274 + 54), "#f7f9fa")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), "Motor-attained -1.3 rad gaze at 7.5 s: frozen virtual camera, 8 distinct no-reset entries", fill="#213748", font=font)
    draw.text((12, 30), "All native reaches still use simulator-truth toy goals; this sheet scores only RGB observability.", fill="#54697a", font=font)
    for index, (seed, path, error, contact) in enumerate(images):
        x = (index % 4) * 324
        y = 54 + (index // 4) * 274
        canvas.paste(Image.open(path).convert("RGB"), (x, y))
        draw.text((x + 4, y + 244), f"{seed}  RGB error {error:.1f} mm  contact {contact:.2f} s",
                  fill="#213748", font=font)
    canvas.save(native_dir / "attained-entry-contact-sheet.png")
    print(result["native_and_entry_image_pass_count"], "/", len(rows),
          "min contact", result["minimum_contact_s"],
          "max image error", result["maximum_entry_xy_error_mm"])


if __name__ == "__main__":
    main()
