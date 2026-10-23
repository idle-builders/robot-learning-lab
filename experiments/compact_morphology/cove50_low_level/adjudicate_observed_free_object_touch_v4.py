"""Score pre-impulse validity and RGB calibration on saved V4 native traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261013/free-object-observed-touch-v4"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads((BASE / "report.json").read_text())
    assert report["episodes_complete"] and len(report["records"]) == 12
    coefficients = np.asarray(report["calibration"]["coefficients"])
    rows = []
    for record in report["records"]:
        trace_path = ROOT / record["trace"]
        assert digest(trace_path) == record["trace_sha256"]
        trace = np.load(trace_path)["step_alpha_rgb_du_dv_cube_contacts_target"]
        assert trace.shape == (200, 19)
        rgb = trace[:, 2:8]
        visible = (rgb[:, 3] == 1) & (rgb[:, 5] == 0)
        image_xy = np.column_stack((np.ones(len(trace)), rgb[:, :3])) @ coefficients
        image_error_mm = np.linalg.norm((image_xy - trace[:, 10:12]) * 1000, axis=1)
        row = {"seed": record["seed"], "mode": record["mode"],
               "challenge": record["challenge"],
               "nominal_teacher_pass": record["nominal_teacher_pass"],
               "supported_touch_s": record["longest_supported_touch_s"],
               "cube_peak_drift_mm": record["cube_peak_drift_mm"],
               "rgb_visible_frames": int(visible.sum()),
               "rgb_xy_error_median_visible_mm": float(np.median(image_error_mm[visible])) if visible.any() else None,
               "rgb_xy_error_step80_mm": float(image_error_mm[79]),
               "rgb_xy_error_at_end_mm": float(image_error_mm[-1]),
               "correction_count": len(record["correction_steps"])}
        if record["impulse"] is not None:
            pre = trace[108]  # control step 109, immediately before step-110 impulse
            pre_valid = bool(pre[13] and pre[15] and pre[16])
            raw = record["impulse"]
            actual_valid = bool(pre_valid and raw["max_post_impulse_xy_shift_mm"] >= 3 and
                                raw["first_lost_finger_step"] is not None and
                                raw["first_lost_finger_step"] >= 110)
            row["impulse_adjudication"] = {
                "pre_impulse_finger_contact": bool(pre[13]),
                "pre_impulse_table_contact": bool(pre[15]),
                "pre_impulse_physical_touch": bool(pre[16]),
                "pre_impulse_cube_center_z_m": float(pre[12]),
                "moved_at_least_3mm": raw["max_post_impulse_xy_shift_mm"] >= 3,
                "actual_valid_challenge": actual_valid,
                "recovery_pass": bool(actual_valid and raw["off_path_pass"]),
                "best_supported_recontact_within_1s_s": raw["best_supported_recontact_within_1s_s"]}
        rows.append(row)
    nominal = [r for r in rows if r["challenge"] == "nominal"]
    impulses = [r for r in rows if r["challenge"] == "impulse"]
    valid_impulses = [r for r in impulses if r["impulse_adjudication"]["actual_valid_challenge"]]
    adjudication = {
        "schema": "robot-learning-lab.cove50-free-object-touch-v4-adjudication.v1",
        "raw_report": str((BASE / "report.json").relative_to(ROOT)),
        "raw_report_sha256": digest(BASE / "report.json"),
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": digest(Path(__file__)),
        "correction_to_raw_field": "Raw impulse.valid_challenge checks displacement and post-impulse contact loss, but omits physical fingertip/table contact immediately before the impulse. This adjudication requires that precondition and supersedes only that validity field.",
        "nominal_episodes": len(nominal),
        "nominal_teacher_passes": sum(r["nominal_teacher_pass"] for r in nominal),
        "valid_impulse_episodes": len(valid_impulses),
        "recovery_passes_on_valid_impulses": sum(r["impulse_adjudication"]["recovery_pass"] for r in valid_impulses),
        "invalid_impulse_episodes": len(impulses) - len(valid_impulses),
        "rows": rows,
        "claim_limit": "Offline adjudication of source-matched local CPU scripted traces; no learned or two-pad skill."}
    (BASE / "adjudication.json").write_text(json.dumps(adjudication, indent=2) + "\n")
    print(json.dumps({k: adjudication[k] for k in (
        "nominal_episodes", "nominal_teacher_passes", "valid_impulse_episodes",
        "recovery_passes_on_valid_impulses", "invalid_impulse_episodes")}, indent=2))


if __name__ == "__main__":
    main()
