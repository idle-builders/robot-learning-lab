"""Summarize the selected-motor stance-feedforward and clearance screens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-support-recovery-20261009"
FAMILIES = ("native-stance-roll-feedforward", "native-hip-pitch-clearance")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(rows: list[dict], predicate) -> dict | None:
    return next((row for row in rows if predicate(row)), None)


def main() -> None:
    pointer_hash = sha256(REPO / "robot/ape-current-design.json")
    body_hash = sha256(ROOT.parent / "ape-cove50-transfer-20261008/cove50-direct-training.xml")
    cases = []
    for family in FAMILIES:
        plan_path = ROOT / f"{family}-plan.json"
        plan = json.loads(plan_path.read_text())
        assert plan["selected_design_pointer_sha256"] == pointer_hash
        assert plan["selected_body_sha256"] == body_hash
        paths = sorted(p for p in (ROOT / family).glob("*.json") if not p.stem.endswith("-video"))
        assert len(paths) == (10 if family == FAMILIES[0] else 2), (family, len(paths))
        for path in paths:
            trial = json.loads(path.read_text())
            assert trial["design_pointer_sha256"] == pointer_hash
            assert trial["body_sha256"] == body_hash
            assert trial["hip_roll_effort_scale_hypothetical"] == 1.0
            trace_path = REPO / trial["state_trace"]
            assert sha256(trace_path) == trial["state_trace_sha256"]
            if trial["video"]:
                assert sha256(REPO / trial["video"]) == trial["video_sha256"]
            rows = trial["timeseries"]
            first_rated = first(rows, lambda row: bool(row["rated_leg_joints"]))
            first_self = first(rows, lambda row: row["self_contact"])
            pair = [trial["stance"] == "left", trial["stance"] == "right"]
            first_support = first(rows, lambda row: row["feet_grounded"] == pair and
                                  row["swing_sole_z_m"] > .025)
            cases.append({
                "family": family,
                "case": path.stem,
                "report": str(path.relative_to(REPO)),
                "report_sha256": sha256(path),
                "trace": trial["state_trace"],
                "video": trial["video"],
                "stance": trial["stance"],
                "stance_hip_roll_delta_rad": trial["swing_joint_targets_rad"]["stance_hip_roll_delta"],
                "swing_hip_pitch_rad": trial["swing_joint_targets_rad"]["hip_pitch"],
                "high_load_entry_pass": trial["high_load_entry_screen_pass"],
                "lift_pass": trial["lift_skill_screen_pass"],
                "duration_s": trial["duration_s"],
                "longest_single_support_s": trial["longest_single_support_s"],
                "final_two_foot_stand": trial["final_two_foot_stand"],
                "rated_leg_fraction": trial["rated_leg_fraction"],
                "max_stance_slip_m": trial["max_stance_slip_m"],
                "self_contact_frames": trial["self_contact_frames"],
                "nonsole_floor_contact_frames": trial["nonsole_floor_contact_frames"],
                "hand_contact_frames": trial["hand_contact_frames"],
                "first_rated_leg": ({"time_s": first_rated["time_s"],
                                     "joints": first_rated["rated_leg_joints"]}
                                    if first_rated else None),
                "first_self_contact_s": first_self["time_s"] if first_self else None,
                "first_exclusive_support_s": first_support["time_s"] if first_support else None,
                "self_contact_pairs": trial["cpu_audited_self_contact_pairs"],
            })
    extra_video = ROOT / FAMILIES[0] / "right-delta0.20-video.json"
    repeat = json.loads(extra_video.read_text())
    original = json.loads((ROOT / FAMILIES[0] / "right-delta0.20.json").read_text())
    for key in ("duration_s", "longest_single_support_s", "rated_leg_fraction",
                "max_stance_slip_m", "self_contact_frames", "nonsole_floor_contact_frames",
                "hand_contact_frames"):
        assert repeat[key] == original[key], key
    assert sha256(REPO / repeat["video"]) == repeat["video_sha256"]
    matrix_path = ROOT / "native-followup-matrix.json"
    matrix_path.write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-native-support-followup-matrix.v1",
        "cases": cases,
    }, indent=2) + "\n")
    report = {
        "schema": "robot-learning-lab.cove50-native-support-followup-report.v1",
        "selected_design_pointer_sha256": pointer_hash,
        "selected_body_sha256": body_hash,
        "actuator_catalog_sha256": sha256(REPO / "robot/actuator-catalog.yaml"),
        "probe_source_sha256": sha256(Path(__file__).with_name("probe_native_shift.py")),
        "plans": {family: {"path": str((ROOT / f"{family}-plan.json").relative_to(REPO)),
                            "sha256": sha256(ROOT / f"{family}-plan.json")}
                  for family in FAMILIES},
        "case_count": len(cases),
        "feedforward_lift_passes": [row["case"] for row in cases
                                    if row["family"] == FAMILIES[0] and row["lift_pass"]],
        "clearance_lift_passes": [row["case"] for row in cases
                                  if row["family"] == FAMILIES[1] and row["lift_pass"]],
        "matrix": str(matrix_path.relative_to(REPO)),
        "representative_feedforward_failure_video": repeat["video"],
        "interpretation": "Authored native-motor diagnostic paths. A prelift high-load two-foot shift is retained; these screens do not qualify a learned policy or an upright-to-single-support skill.",
    }
    (ROOT / "native-followup-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
