"""Condense retained native shift and lift probes for the session inspector."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-support-recovery-20261009"
EXPERIMENTS = (
    "native-shift", "native-load-screen", "native-arm-clearance",
    "native-lift-pilot", "native-lateral-lift", "native-com-feedback",
    "native-mixed-lift", "native-hip-roll-envelope",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = []
    for experiment in EXPERIMENTS:
        for path in sorted((ROOT / experiment).glob("*.json")):
            trial = json.loads(path.read_text())
            rows = trial["timeseries"]
            first_rated = next(({"time_s": row["time_s"], "joints": row["rated_leg_joints"]}
                                for row in rows if row["rated_leg_joints"]), None)
            rated_counts = Counter(joint for row in rows for joint in row["rated_leg_joints"])
            lift_rows = [row for row in rows if row["phase"] in ("lift", "hold", "lower")]
            matrix.append({
                "experiment": experiment, "case": path.stem,
                "evidence": str(path.relative_to(REPO)), "evidence_sha256": sha256(path),
                "video": trial.get("video"),
                "stance": trial["stance"], "seed": trial["seed"],
                "rail_v": trial["rail_v"], "slide_friction": trial["slide_friction"],
                "delay_ms": trial["delay_ms"], "reset_jitter": trial["reset_jitter"],
                "hip_roll_effort_scale_hypothetical": trial.get("hip_roll_effort_scale_hypothetical", 1.0),
                "shift_roll_rad": trial["shift_roll_rad"],
                "arm_clearance_roll_rad": trial.get("clear_swing_arm_roll_rad", 0.0),
                "swing_joint_targets_rad": trial.get("swing_joint_targets_rad"),
                "duration_s": trial["duration_s"],
                "completed_schedule": trial["completed_schedule"],
                "entry_screen_pass": trial["entry_screen_pass"],
                "high_load_entry_screen_pass": trial.get("high_load_entry_screen_pass"),
                "mean_stance_load_dwell": trial.get("dwell_mean_stance_foot_load_fraction"),
                "longest_single_support_s": trial.get("longest_single_support_s"),
                "final_two_foot_stand": trial.get("final_two_foot_stand"),
                "lift_skill_screen_pass": trial.get("lift_skill_screen_pass"),
                "max_stance_slip_m": trial["max_stance_slip_m"],
                "max_swing_sole_z_m": trial.get("max_swing_sole_z_m"),
                "rated_leg_fraction": trial["rated_leg_fraction"],
                "first_rated_leg": first_rated,
                "rated_joint_frames": dict(rated_counts),
                "max_abs_lift_com_stance_y_m": (max(abs(row["com_minus_stance_world_xy_m"][1])
                                                for row in lift_rows) if lift_rows else None),
                "self_contact_frames": trial["self_contact_frames"],
                "nonsole_floor_contact_frames": trial["nonsole_floor_contact_frames"],
                "hand_contact_frames": trial["hand_contact_frames"],
                "cpu_audited_self_contact_pairs": trial["cpu_audited_self_contact_pairs"],
            })
    body = REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    pointer = REPO / "robot/ape-current-design.json"
    report = {
        "schema": "robot-learning-lab.cove50-native-support-summary.v1",
        "selected_design_id": json.loads(pointer.read_text())["id"],
        "selected_design_pointer_sha256": sha256(pointer),
        "selected_body_sha256": sha256(body),
        "actuator_catalog_sha256": sha256(REPO / "robot/actuator-catalog.yaml"),
        "probe_source_sha256": sha256(Path(__file__).with_name("probe_native_shift.py")),
        "matrix": "artifacts/ape-cove50-support-recovery-20261009/native-support-matrix.json",
        "trial_count": len(matrix),
        "selected_motor_high_load_entry": [
            row["evidence"] for row in matrix
            if row["experiment"] == "native-arm-clearance" and
            row["high_load_entry_screen_pass"]],
        "selected_motor_lift_passing_cases": [
            row["evidence"] for row in matrix
            if row["hip_roll_effort_scale_hypothetical"] == 1.0 and
            row["lift_skill_screen_pass"]],
        "hypothetical_motor_lift_passing_cases": [
            row["evidence"] for row in matrix
            if row["hip_roll_effort_scale_hypothetical"] != 1.0 and
            row["lift_skill_screen_pass"]],
        "interpretation": "Authored native motor screens and failure traces only. A two-foot high-load entry is retained; full single-support and learned policy qualification require separate gates. Hypothetical motor scales are diagnostic and do not change the selected body or catalog.",
    }
    (ROOT / "native-support-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-support-matrix.v1",
        "cases": matrix,
    }, indent=2) + "\n")
    (ROOT / "native-support-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
