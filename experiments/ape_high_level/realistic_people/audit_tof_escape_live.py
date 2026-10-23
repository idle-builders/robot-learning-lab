"""Compare a live ToF-triggered eye escape with its held-pose control."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_viewpoint_route import box_clearance


def episode(run: Path) -> dict:
    evidence = run / "evidence"
    scene_path = evidence / "scene-info.json"
    tof_path = evidence / "low-tof-trace.json"
    scene = json.loads(scene_path.read_text())
    trace = json.loads(tof_path.read_text())
    size = scene["route_obstacle_evaluator_only"]["size_xy_m"]
    rows = []
    for sample in trace["samples"]:
        gap = box_clearance(sample["eye_xy_m_evaluator_only"],
                            sample["cube_center_xy_m_evaluator_only"], size)
        rows.append({"time_s": sample["sim_time_s"],
                     "eye_cube_box_gap_m_evaluator_only": gap,
                     "eye_xy_m_evaluator_only": sample["eye_xy_m_evaluator_only"],
                     "cube_xy_m_evaluator_only": sample[
                         "cube_center_xy_m_evaluator_only"]})
    assert rows and all(b["time_s"] > a["time_s"] for a, b in zip(rows, rows[1:]))
    response_path = evidence / "tof-response-trace.json"
    response = json.loads(response_path.read_text()) if response_path.exists() else None
    mount_audit = run / "low-tof-intrusion-audit.json"
    mount_rows = (json.loads(mount_audit.read_text())["per_mount"]
                  if mount_audit.exists() else None)
    active_flank = (max(("front_left", "front_right"), key=lambda name:
        mount_rows[name]["samples_with_cube_return"])
        if mount_rows is not None else None)
    mount_gap = (mount_rows[active_flank][
        "minimum_mount_cube_box_gap_m_evaluator_only"]
        if mount_rows is not None else None)
    alarm = response["summary"]["alarm"] if response is not None else None
    alarm_row = min(rows, key=lambda row: abs(row["time_s"] - alarm["time_s"])) if alarm else None
    response_samples = response["samples"] if response else []
    movement = [row for row in response_samples
                if (row["response"].get("translation_increment_m",
                                        row["response"].get("reverse_increment_m", 0.)) > 0.)]
    nearest = min(rows, key=lambda row: row["eye_cube_box_gap_m_evaluator_only"])
    moves = [row["time_s"] for row in movement]
    action_path = evidence / "online-action-trace.json"
    actions = json.loads(action_path.read_text())["actions"] if action_path.exists() else []
    source_invalidation = {"missed": [], "unnecessary": [],
                           "fresh_look_frames_after_escape": []}
    for action in actions:
        source_s = action["observation_time_s"]
        decision_s = action["decision_time_s"]
        motion_after_source = any(source_s + 1e-6 < time_s <= decision_s + 1e-6
                                  for time_s in moves)
        invalidated = (action["actor"].get("reason") ==
                       "tof_escape_invalidated_image")
        if motion_after_source and not invalidated:
            source_invalidation["missed"].append(action["frame"])
        if invalidated and not motion_after_source:
            source_invalidation["unnecessary"].append(action["frame"])
        if (moves and source_s > moves[-1] + 1e-6 and
                action["actor"].get("intent") == "look_to_owner"):
            source_invalidation["fresh_look_frames_after_escape"].append(
                action["frame"])
    post_escape_actions = [action for action in actions if moves and
                           action["observation_time_s"] > moves[-1] + 1e-6]
    max_gap_s = max(b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:]))
    # In this fixture the cube is scripted at no more than 0.15 m/s and the
    # surrogate response at no more than 0.25 m/s. This worst-case sampling
    # bound is useful only for this authored motion, not a real speed limit.
    unsampled_bound = nearest["eye_cube_box_gap_m_evaluator_only"] - .40 * max_gap_s
    return {"run": run.name,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("scene", scene_path), ("tof", tof_path))},
            "samples": len(rows),
            "maximum_sample_gap_s": max_gap_s,
            "initial_eye_cube_gap_m": rows[0]["eye_cube_box_gap_m_evaluator_only"],
            "minimum_eye_cube_gap_m": nearest["eye_cube_box_gap_m_evaluator_only"],
            "conservative_between_samples_gap_bound_m": unsampled_bound,
            "active_flank_mount": active_flank,
            "minimum_active_flank_mount_cube_gap_m": mount_gap,
            "conservative_mount_gap_bound_m": (mount_gap - .40 * max_gap_s
                if mount_gap is not None else None),
            "minimum_gap_time_s": nearest["time_s"],
            "entered_20cm_eye_proxy": nearest["eye_cube_box_gap_m_evaluator_only"] < .2,
            "final_eye_cube_gap_m": rows[-1]["eye_cube_box_gap_m_evaluator_only"],
            "alarm": alarm,
            "gap_at_alarm_m_evaluator_only": (alarm_row[
                "eye_cube_box_gap_m_evaluator_only"] if alarm_row else None),
            "first_motion_time_s": movement[0]["time_s"] if movement else None,
            "last_motion_time_s": movement[-1]["time_s"] if movement else None,
            "escape_total_m": (response["summary"].get("escape_total_m",
                response["summary"].get("retreat_total_m", 0.)) if response else 0.),
            "escape_heading_rel_deg": (response["summary"].get(
                "escape_heading_rel_deg") if response else None),
            "response_state": response["summary"]["state"] if response else None,
            "motion_steps": len(movement),
            "source_invalidation": source_invalidation,
            "post_escape_actor_intents": dict(Counter(
                action["actor"].get("intent") for action in post_escape_actions)),
            "post_escape_social_actor_frames": sum(
                action.get("social_actor") is not None
                for action in post_escape_actions),
            "corridor_sources": sorted({(row["response"].get("chosen_corridor") or
                                         row["response"].get("corridor") or {}).get("reason")
                for row in movement})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_run", type=Path)
    parser.add_argument("held_control_run", type=Path)
    parser.add_argument("static_control_run", type=Path)
    args = parser.parse_args()
    report = {"scope": "live Isaac eye-proxy response outcome, 15 Hz simulated-time "
               "sampling; evaluator geometry used only in audit",
              "response": episode(args.response_run),
              "held_control": episode(args.held_control_run),
              "static_control": episode(args.static_control_run),
              "limitations": [
                  "Three independent GPU runs can have different inference timing",
                  "20 cm eye proxy only, not full articulated body contact; a flank mount may be closer",
                  "Between-sample gap bound assumes scripted 15 cm/s cube and 25 cm/s surrogate escape",
                  "Escape authorization uses ideal 360-degree dual-height PhysX rays",
                  "No optical faults, foot slip, joint/IMU/contact dynamics or selected body"]}
    output = args.response_run / "tof-escape-live-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: row[key] for key in (
        "minimum_eye_cube_gap_m", "conservative_between_samples_gap_bound_m",
        "active_flank_mount", "minimum_active_flank_mount_cube_gap_m",
        "conservative_mount_gap_bound_m",
        "entered_20cm_eye_proxy",
        "gap_at_alarm_m_evaluator_only", "escape_total_m", "response_state")}
        for name, row in report.items() if name in
        ("response", "held_control", "static_control")}, indent=2))


if __name__ == "__main__":
    main()
