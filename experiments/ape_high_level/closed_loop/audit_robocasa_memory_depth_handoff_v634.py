"""Audit saved head-depth arithmetic, motion and owner outcomes in V633."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_reactive_hold_v626 as range_audit
import run_robocasa_memory_depth_handoff_v633 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_episode(ep: dict, source: str, label: str,
                  dropout: tuple[float, float] | None, depth_fallback: bool,
                  prior, camera_model) -> dict:
    summary = ep["memory_return"]
    assert summary["source"] == source
    assert summary["run_label"] == (
        f"{label}_{'drop' if dropout else 'healthy'}_"
        f"{'depth' if depth_fallback else 'range_only'}_held")
    assert summary["head_depth_fallback_requested"] == depth_fallback
    assert summary["hold_scan_dropout_window_s_requested"] == (
        None if dropout is None else list(dropout))
    original_motion, original_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    try:
        crossing.prop_motion = lambda condition: (
            original_motion("stationary") if condition == "stationary" else
            lambda time_s: original_motion("moving_across_return")(
                time_s + trial.SHIFTS_S[condition]))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        geometry = base_audit.audit_episode(ep, label)
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    monitor = range_audit.audit_monitor(ep, prior, .1)
    missing = [row for row in ep["trace"]
               if row["stage"] == "memory_post_stop_range_missing"]
    assert len(missing) == summary["hold_missing_scan_count"]
    if dropout is not None:
        assert all(dropout[0] <= row["time_s"] < dropout[1] and
                   row["source_health"] == "missing" and
                   "range_72_beams_m" not in row for row in missing)
    else:
        assert not missing
    aims = [row for row in ep["trace"] if row["stage"] == "memory_head_depth_aim"]
    captures = [row for row in ep["trace"]
                if row["stage"] == "memory_head_depth_fallback"]
    assert len(captures) == summary["head_depth_capture_count"]
    if depth_fallback and missing:
        assert len(aims) == 1 and aims[0]["time_s"] == missing[0]["time_s"]
        track = aims[0]["actor_last_range_track"]
        assert track["status"] == "confirmed_now" and track["radius_m"] >= .18
        assert track["last_seen_s"] < aims[0]["time_s"]
        assert all(row["time_s"] >= aims[0]["actor_ready_time_s"]
                   for row in captures)
        estimate = np.asarray(missing[0]["actor_estimated_xy_m"])
        direction = np.asarray(track["center_xy_m"]) - estimate
        yaw = -float(np.arctan2(direction[0], direction[1]))
        assert abs(yaw - aims[0]["actor_commanded_yaw_rad"]) < .00002
    else:
        assert not aims and not captures
    base_quat = camera_model.cam_quat[camera_model.camera("head").id].copy()
    head_id = camera_model.camera("head").id
    checked_captures = []
    for row in captures:
        rgb_path, depth_path = Path(row["rgb_path"]), Path(row["depth_path"])
        assert base_audit.sha(rgb_path) == row["rgb_sha256"]
        assert base_audit.sha(depth_path) == row["depth_sha256"]
        depth = np.load(depth_path)
        assert depth.shape == (480, 640)
        yaw = row["actor_commanded_yaw_rad"]
        yaw_quat = np.array([np.cos(yaw / 2.), 0., 0., np.sin(yaw / 2.)])
        camera_quat = np.empty(4)
        mujoco.mju_mulQuat(camera_quat, yaw_quat, base_quat)
        flat = np.empty(9)
        mujoco.mju_quat2Mat(flat, camera_quat)
        replay = memory.head_depth_compact_surface(
            depth, np.asarray(row["actor_estimated_xy_m"]),
            camera_model.cam_pos[head_id, 2], flat.reshape(3, 3),
            camera_model.cam_fovy[head_id], aims[0]["actor_last_range_track"]["radius_m"])
        assert replay is not None
        saved = row["actor_depth_observation"]
        assert saved is not None
        assert replay["blob_pixels"] == saved["blob_pixels"]
        assert np.linalg.norm(np.asarray(replay["center_xy_m"]) -
                              np.asarray(saved["center_xy_m"])) < .00004
        assert abs(replay["median_depth_m"] - saved["median_depth_m"]) < .00002
        checked_captures.append({"time_s": row["time_s"],
                                 "blob_pixels": saved["blob_pixels"],
                                 "center_xy_m": saved["center_xy_m"],
                                 "capture_wall_s": row["capture_wall_s"]})
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    assert len(decisions) == summary["held_reactive_escape_count"] <= 1
    for decision in decisions:
        if decision.get("actor_source") != "head_depth_after_range_loss":
            scan = next(row for row in ep["trace"]
                        if row["stage"] == "memory_post_stop_range_scan" and
                        row["time_s"] == decision["time_s"])
            assert decision["actor_track"] in scan["actor_multi_tracks"]
            assert abs(range_audit.projected_gap(scan, decision["actor_track"]) -
                       decision["actor_projected_gap_m"]) < .00002
            continue
        assert len(captures) >= 2 and decision["time_s"] == captures[-1]["time_s"]
        current, previous = captures[-1], captures[-2]
        center = np.asarray(current["actor_depth_observation"]["center_xy_m"])
        old_center = np.asarray(previous["actor_depth_observation"]["center_xy_m"])
        velocity = (center - old_center) / (current["time_s"] - previous["time_s"])
        assert velocity[0] < -.3
        assert np.linalg.norm(velocity - np.asarray(
            decision["actor_track"]["velocity_xy_m_s"])) < .00003
        radius = decision["actor_track"]["radius_m"]
        pose = np.asarray(current["actor_estimated_xy_m"])
        gap = min(float(np.linalg.norm(pose - (center + velocity * dt)) -
                        radius - .17 - .03 - .02 * dt)
                  for dt in np.arange(0., 1.201, .1))
        assert abs(gap - decision["actor_projected_gap_m"]) < .00003
        perpendicular = np.array([-velocity[1], velocity[0]])
        if np.dot(perpendicular, pose - center) < 0:
            perpendicular *= -1
        perpendicular /= np.linalg.norm(perpendicular)
        for option, distance in zip(decision["actor_candidate_goals"],
                                    (.35, .45, .25)):
            assert np.linalg.norm(np.asarray(option["goal_xy_m"]) -
                                  (pose + distance * perpendicular)) < .00005
        selected = next((option for option in decision["actor_candidate_goals"]
                         if option["plan_status"] == "known_goal" and
                         option["has_request"]), None)
        assert decision["actor_selected_goal_xy_m"] == (
            None if selected is None else selected["goal_xy_m"])
    if label == "stationary":
        assert summary["item_status"] == "fresh_metric_item_refound"
    else:
        assert summary["item_status"] == "not_rechecked"
        assert any(row["stage"] == "predeparture_owner_help" for row in ep["trace"])
    return {"source": source, "scenario": label,
            "depth_fallback": depth_fallback,
            "missing_range_packets": len(missing),
            "head_depth_captures_checked": checked_captures,
            "second_action_time_s": decisions[0]["time_s"] if decisions else None,
            **geometry, **monitor}


def compare_before_loss(no_depth: dict, depth: dict) -> dict:
    def packets(ep):
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                     "memory_route_decision",
                                     "memory_post_stop_range_scan") and
                row["time_s"] < trial.DROP_WINDOW_S[0]]
    first, second = packets(no_depth), packets(depth)
    assert len(first) == len(second)
    unequal = []
    for a, b in zip(first, second):
        assert a["stage"] == b["stage"] and a["time_s"] == b["time_s"]
        if a["range_72_beams_m"] != b["range_72_beams_m"]:
            unequal.append(a["time_s"])
    return {"compared": len(first), "identical": len(first) - len(unequal),
            "first_unequal_time_s": unequal[0] if unequal else None}


def main() -> None:
    declaration_path = OUT / "predeclared-depth-handoff.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    prior = np.load(joined.PRIOR)
    camera_model = joined.scene_with_owner_and_prop(crossing.START)
    rows = [audit_episode(ep, source, label, dropout, depth_fallback,
                          prior, camera_model)
            for ep, (source, label, dropout, depth_fallback) in zip(
                report["episodes"], trial.PROFILES)]
    pairs = {source: compare_before_loss(report["episodes"][i],
                                         report["episodes"][i + 1])
             for source, i in (("range", 2), ("rgbd", 4))}
    result = {"schema": "robot-learning-lab.robocasa-memory-depth-handoff-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "videos_decoded": len(rows),
              "pre_loss_pair_comparison": pairs,
              "depth_captures_replayed": sum(
                  len(row["head_depth_captures_checked"]) for row in rows),
              "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
