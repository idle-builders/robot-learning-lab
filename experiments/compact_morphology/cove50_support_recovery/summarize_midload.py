"""Compare frozen selected-body mid-load lift paths with native evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-support-recovery-20261009"
FAMILIES = (
    ("native-midload-lift", "lateral +0.20 rad", "native-midload-lift-plan.json"),
    ("native-midload-pitch", "pitch -0.40 rad", "native-midload-pitch-plan.json"),
    ("native-midload-pitch06", "pitch -0.60 rad", "native-midload-pitch06-plan.json"),
    ("native-midload-ankle", "pitch -0.40 + ankle +0.20 rad", "native-midload-ankle-plan.json"),
    ("native-highload-pitch06", "high-load shift + pitch -0.60 rad", "native-highload-pitch06-plan.json"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(rows: list[dict], predicate) -> dict | None:
    return next((row for row in rows if predicate(row)), None)


def main() -> None:
    pointer_sha = sha(REPO / "robot/ape-current-design.json")
    body_sha = sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    cases = []
    plans = []
    for family, path_label, plan_name in FAMILIES:
        plan_path = ROOT / plan_name
        plan = json.loads(plan_path.read_text())
        assert plan["selected_design_pointer_sha256"] == pointer_sha
        assert plan["selected_body_sha256"] == body_sha
        plans.append({"path": str(plan_path.relative_to(REPO)), "sha256": sha(plan_path)})
        paths = sorted((ROOT / family).glob("*.json"))
        assert len(paths) in (2, 4), (family, len(paths))
        for path in paths:
            trial = json.loads(path.read_text())
            assert trial["design_pointer_sha256"] == pointer_sha
            assert trial["body_sha256"] == body_sha
            assert trial["hip_roll_effort_scale_hypothetical"] == 1.0
            assert sha(REPO / trial["state_trace"]) == trial["state_trace_sha256"]
            assert sha(REPO / trial["video"]) == trial["video_sha256"]
            rows = trial["timeseries"]
            first_rated = first(rows, lambda row: bool(row["rated_leg_joints"]))
            first_self = first(rows, lambda row: row["self_contact"])
            pair = [trial["stance"] == "left", trial["stance"] == "right"]
            first_support = first(rows, lambda row: row["feet_grounded"] == pair and
                                  row["swing_sole_z_m"] > .025)
            cases.append({
                "case": family + "/" + path.stem,
                "path": path_label,
                "stance": trial["stance"],
                "condition": path.stem.split("-")[-1],
                "report": str(path.relative_to(REPO)),
                "report_sha256": sha(path),
                "trace": trial["state_trace"],
                "video": trial["video"],
                "prelift_stance_load_share": trial["dwell_mean_stance_foot_load_fraction"],
                "completed_schedule": trial["completed_schedule"],
                "duration_s": trial["duration_s"],
                "longest_single_support_s": trial["longest_single_support_s"],
                "max_swing_sole_z_m": trial["max_swing_sole_z_m"],
                "max_swing_sole_z_mm": round(trial["max_swing_sole_z_m"] * 1000, 2),
                "final_two_foot_stand": trial["final_two_foot_stand"],
                "rated_leg_fraction": trial["rated_leg_fraction"],
                "max_stance_slip_m": trial["max_stance_slip_m"],
                "max_stance_slip_mm": round(trial["max_stance_slip_m"] * 1000, 2),
                "self_contact_frames": trial["self_contact_frames"],
                "nonsole_floor_contact_frames": trial["nonsole_floor_contact_frames"],
                "hand_contact_frames": trial["hand_contact_frames"],
                "first_rated_leg": ({"time_s": first_rated["time_s"],
                                     "joints": first_rated["rated_leg_joints"]}
                                    if first_rated else None),
                "first_rated_joint": (", ".join(first_rated["rated_leg_joints"])
                                      if first_rated else None),
                "first_rated_s": first_rated["time_s"] if first_rated else None,
                "first_self_contact_s": first_self["time_s"] if first_self else None,
                "first_exclusive_support_s": (first_support["time_s"]
                                              if first_support else None),
                "lift_pass": trial["lift_skill_screen_pass"],
            })
    out = {
        "schema": "robot-learning-lab.cove50-native-support-path-comparison.v1",
        "method": "Authored 50 Hz native MJLab motor/contact screens on the selected body; no learned policy pass is inferred.",
        "selected_design_pointer_sha256": pointer_sha,
        "selected_body_sha256": body_sha,
        "actuator_catalog_sha256": sha(REPO / "robot/actuator-catalog.yaml"),
        "probe_source_sha256": sha(Path(__file__).with_name("probe_native_shift.py")),
        "plans": plans,
        "static_corner_screens": [
            str((ROOT / name).relative_to(REPO)) for name in
            ("native-midload-pitch-corner-screen.json", "native-midload-ankle-corner-screen.json",
             "native-highload-pitch-corner-screen.json")
        ],
        "cases": cases,
    }
    midload = out.copy()
    midload["schema"] = "robot-learning-lab.cove50-native-midload-comparison.v1"
    midload["plans"] = plans[:-1]
    midload["static_corner_screens"] = out["static_corner_screens"][:-1]
    midload["cases"] = [row for row in cases if not row["case"].startswith("native-highload-pitch06/")]
    (ROOT / "native-midload-comparison.json").write_text(json.dumps(midload, indent=2) + "\n")
    path = ROOT / "native-support-path-comparison.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(path), "cases": len(cases),
                      "passes": [row["case"] for row in cases if row["lift_pass"]]}, indent=2))


if __name__ == "__main__":
    main()
