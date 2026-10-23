"""Summarize the frozen paired load/COM and stance-roll sign screens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-coupled-transition-20261010"
BASE = REPO / "artifacts/ape-cove50-contact-transition-20261010/trials"
CASES = (
    ("roll08_com0", BASE, "loadroll", 0.08, 0.0, 0.0),
    ("roll04_com0", ROOT / "trials", "roll04_com0", 0.04, 0.0, 0.0),
    ("roll04_com2", ROOT / "trials", "roll04_com2", 0.04, 2.0, 0.0),
    ("roll08_com2", ROOT / "trials", "roll08_com2", 0.08, 2.0, 0.0),
    ("stance_in04", ROOT / "stance-roll-sign", "stance_in04", 0.04, 0.0, -0.04),
    ("stance_out04", ROOT / "stance-roll-sign", "stance_out04", 0.04, 0.0, 0.04),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(path: Path) -> None:
    for line in path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert sha(REPO / relative) == digest, relative


def at(rows: list[dict], time_s: float) -> dict:
    row = min(rows, key=lambda item: abs(item["time_s"] - time_s))
    assert abs(row["time_s"] - time_s) <= 0.011, time_s
    return row


def main() -> None:
    plan = json.loads((ROOT / "plan.json").read_text())
    sign_plan = json.loads((ROOT / "stance-roll-sign-plan.json").read_text())
    assert sha(REPO / plan["selected_design_pointer"]["path"]) == plan["selected_design_pointer"]["sha256"]
    assert sha(REPO / plan["selected_body"]["path"]) == plan["selected_body"]["sha256"]
    assert sha(REPO / plan["actuator_catalog"]["path"]) == plan["actuator_catalog"]["sha256"]
    assert sha(REPO / plan["probe_source"]["path"]) == plan["probe_source"]["sha256"]
    assert sign_plan["selected_body_sha256"] == plan["selected_body"]["sha256"]
    assert sign_plan["probe_source_sha256"] == plan["probe_source"]["sha256"]
    for side, item in plan["baseline_reports"].items():
        assert sha(REPO / item["path"]) == item["sha256"], side
    for side, item in sign_plan["baseline_reports"].items():
        assert sha(REPO / item["path"]) == item["sha256"], side
    checked_manifest(ROOT / "trials/remote-sha256.txt")
    checked_manifest(ROOT / "stance-roll-sign/remote-sha256.txt")

    cases = []
    for name, directory, prefix, roll_cap, com_gain, stance_delta in CASES:
        for side in ("right", "left"):
            report_path = directory / f"{prefix}-{side}-nominal.json"
            report = json.loads(report_path.read_text())
            assert report["design_pointer_sha256"] == plan["selected_design_pointer"]["sha256"]
            assert report["body_sha256"] == plan["selected_body"]["sha256"]
            assert report["stance"] == side and report["seed"] == 7144
            assert report["rail_v"] == 12 and report["slide_friction"] == 1
            assert report["delay_ms"] == 0 and report["reset_jitter"] == "none"
            assert report["shift_roll_rad"] == 0.33
            assert report["swing_joint_targets_rad"]["hip_pitch"] == -0.6
            assert report["swing_load_feedback_gain_rad_per_fraction"] == 1
            assert report["swing_load_roll_limit_rad"] == roll_cap
            assert report["stance_com_feedback_gain_rad_per_m"] == com_gain
            assert report["swing_joint_targets_rad"]["stance_hip_roll_delta"] == stance_delta
            assert report["duration_s"] >= 4.6
            assert not report["lift_skill_screen_pass"]
            trace_path = REPO / report["state_trace"]
            video_path = REPO / report["video"]
            assert sha(trace_path) == report["state_trace_sha256"]
            assert sha(video_path) == report["video_sha256"]
            rows = report["timeseries"]
            first_rated = next((row for row in rows if row["rated_leg_joints"]), None)
            r42 = at(rows, 4.2)
            r46 = at(rows, 4.6)
            mirror = -1 if side == "right" else 1
            cases.append({
                "case": f"{name}/{side}",
                "stance": side,
                "report": str(report_path.relative_to(REPO)),
                "report_sha256": sha(report_path),
                "trace": report["state_trace"],
                "video": report["video"],
                "swing_roll_cap_rad": roll_cap,
                "com_gain_rad_per_m": com_gain,
                "stance_roll_delta_rad": stance_delta,
                "duration_s": report["duration_s"],
                "completed_schedule": report["completed_schedule"],
                "prelift_stance_load": report["dwell_mean_stance_foot_load_fraction"],
                "load_at_4p2": r42["stance_foot_load_fraction"],
                "load_at_4p6": r46["stance_foot_load_fraction"],
                "stance_side_com_at_4p2_mm": round(mirror * r42["com_minus_stance_world_xy_m"][1] * 1000, 2),
                "stance_side_com_at_4p6_mm": round(mirror * r46["com_minus_stance_world_xy_m"][1] * 1000, 2),
                "swing_sole_at_4p6_mm": round(r46["swing_sole_z_m"] * 1000, 2),
                "longest_exclusive_support_s": report["longest_single_support_s"],
                "first_rated_joint": first_rated["rated_leg_joints"][0] if first_rated else None,
                "first_rated_s": first_rated["time_s"] if first_rated else None,
                "rated_leg_fraction": report["rated_leg_fraction"],
                "max_stance_slip_mm": round(report["max_stance_slip_m"] * 1000, 2),
                "self_contact_frames": report["self_contact_frames"],
                "hand_contact_frames": report["hand_contact_frames"],
                "nonsole_floor_contact_frames": report["nonsole_floor_contact_frames"],
                "full_gate_pass": report["lift_skill_screen_pass"],
            })
    output = {
        "schema": "robot-learning-lab.cove50-coupled-transition-comparison.v1",
        "method": "Paired authored native MJLab selected-motor replays; simulator load and COM truth are diagnostic, not learned policy observations.",
        "selected_design_pointer_sha256": plan["selected_design_pointer"]["sha256"],
        "selected_body_sha256": plan["selected_body"]["sha256"],
        "actuator_catalog_sha256": plan["actuator_catalog"]["sha256"],
        "plans": [{"path": str(path.relative_to(REPO)), "sha256": sha(path)} for path in
                  (ROOT / "plan.json", ROOT / "stance-roll-sign-plan.json")],
        "gate": plan["full_gate"],
        "held_out_stress": "Skipped: no new candidate passed both nominal stances.",
        "cases": cases,
    }
    (ROOT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "passes": sum(row["full_gate_pass"] for row in cases),
                      "best_exclusive_support_s": max(row["longest_exclusive_support_s"] for row in cases),
                      "output": str(ROOT / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
