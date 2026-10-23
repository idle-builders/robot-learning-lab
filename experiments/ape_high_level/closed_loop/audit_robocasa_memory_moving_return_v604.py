"""Independent full-horizon geometry, sensor and media audit for V603."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
from audit_robocasa_layout2_routes_v409 import clearance
import audit_robocasa_person_joined_v498 as owner_audit
import robocasa_person_interaction as interaction
import run_robocasa_memory_moving_return_v603 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_moving_prop_v423 as moving
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def sha(path: Path) -> str:
    return memory.digest(path)


def audit_episode(ep: dict, label: str) -> dict:
    summary = ep["memory_return"]
    assert summary is not None
    assert ep["outcome"] == "owner_confirmed_metric_find"
    held_rows = [row for row in ep["trace"] if row["stage"] == "safe_side_hold"]
    pre_memory = {**ep,
                  "final_actual_xy_m": ep["route_episode"]["end_true_xy_m"],
                  "hold_until_s": held_rows[-1]["time_s"]}
    old_motion = owner_audit.motion
    owner_audit.motion = lambda _: trial.prop_motion(label)
    try:
        baseline = owner_audit.audit_episode(pre_memory)
    finally:
        owner_audit.motion = old_motion
    assert baseline["scene_fingerprint_pass"]
    assert baseline["ape_5cm_margin_pass"]
    assert baseline["person_room_5cm_pass"]
    assert baseline["can_room_nonoverlap_pass"]
    assert baseline["video_sha_pass"] and baseline["video_frames_pass"]

    model = joined.scene_with_owner_and_prop(trial.START)
    data = mujoco.MjData(model)
    interaction.show_pose(model, 1.)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    script = trial.prop_motion(label)
    fixture_gap = .3
    for time_s in np.arange(0., trial.POST_STOP_UNTIL_S + .0001, .02):
        xy = script(float(time_s))
        model.body_pos[prop_body, :2] = xy
        mujoco.mj_forward(model, data)
        gap, _ = prop_fixture_gap(model, data, prop_geom, ape_geom, xy)
        fixture_gap = min(fixture_gap, gap)
    assert fixture_gap >= 0.

    minimum = .1
    post_stop_minimum = .1
    first_breach = None
    movement_samples = 0
    stopped_checks = 0
    hold_samples = 0
    watch_samples = 0
    fresh_views = []

    def sample(time_s: float, xy: np.ndarray, recorded_gap: float | None,
               phase: str, guard_range: float | None = None,
               guard_xy: np.ndarray | None = None) -> None:
        nonlocal minimum, post_stop_minimum, first_breach
        model.body_pos[prop_body, :2] = script(time_s)
        if guard_range is not None:
            origin_xy = xy if guard_xy is None else guard_xy
            model.cam_pos[model.camera("head").id, :2] = origin_xy
            mujoco.mj_forward(model, data)
            observed = moving.raw_nearest_range(model, data, origin_xy)
            assert abs(observed - guard_range) <= .00003
        gap, _ = clearance(model, data, ape_body, ape_geom, xy)
        if recorded_gap is not None:
            assert abs(gap - recorded_gap) <= .00003
        if phase != "guard_stop":
            minimum = min(minimum, gap)
        if phase == "post_stop_hold":
            post_stop_minimum = min(post_stop_minimum, gap)
        if gap < .05 and first_breach is None:
            first_breach = {"time_s": time_s, "phase": phase,
                            "gap_m": round(gap, 5)}

    for row in ep["trace"]:
        stage = row["stage"]
        if stage == "memory_fresh_view":
            fresh_views.append(row)
            assert sha(Path(row["image_path"])) == row["image_sha256"]
            assert sha(Path(row["depth_path"])) == row["depth_sha256"]
        if stage == "predeparture_guard":
            xy = np.asarray(row["evaluator_actual_xy_m"])
            sample(float(row["time_s"]), xy,
                   row["evaluator_proxy_gap_m"],
                   "predeparture_watch", row["actor_guard_range_m"], xy)
            watch_samples += 1
        if stage == "predeparture_range_scan":
            assert len(row["range_72_beams_m"]) == 72
        if stage == "memory_route_decision":
            assert len(row["range_72_beams_m"]) == 72
            assert row["range_capture_time_s"] == row["time_s"]
            assert row["memory_capture_time_s"] <= row["time_s"]
            command = row.get("actor_bounded_command_xy_m")
            if command is not None:
                assert np.linalg.norm(command) <= .10001
            current = np.asarray(row["evaluator_actual_xy_m"])
            for sub in row["motion_substeps"]:
                if sub["stopped_before_move"]:
                    sample(float(sub["time_s"]), current, None,
                           "guard_stop", sub["actor_guard_range_m"], current)
                    assert sub["actor_guard_range_m"] < moving.GUARD_CENTER_RANGE_M
                    stopped_checks += 1
                else:
                    xy = np.asarray(sub["evaluator_actual_xy_m"])
                    sample(float(sub["time_s"]), xy,
                           sub["evaluator_proxy_gap_m"],
                           "executed_move", sub["actor_guard_range_m"], current)
                    current = xy
                    movement_samples += 1
        elif stage == "memory_post_stop_hold":
            sample(float(row["time_s"]),
                   np.asarray(row["evaluator_actual_xy_m"]),
                   row["evaluator_proxy_gap_m"],
                   "post_stop_hold", row["actor_guard_range_m"])
            hold_samples += 1
    assert abs(round(minimum, 5) - summary[
        "minimum_proxy_gap_m_evaluator_only"]) <= .00002, (
            label, summary["source"], minimum,
            summary["minimum_proxy_gap_m_evaluator_only"])
    assert abs(round(post_stop_minimum, 5) - summary[
        "post_stop_minimum_proxy_gap_m_evaluator_only"]) <= .00002
    assert (post_stop_minimum < .05) == summary[
        "post_stop_margin_breach_evaluator_only"]
    if summary["item_status"] == "fresh_metric_item_refound":
        assert summary["last_pose_health"] == "registered"
        assert summary["fresh_track_status"] == "visually_reacquired_unlocalized"
        assert summary["fresh_frame_age_s"] <= interaction.owner.MAX_FRESH_AGE_S
        assert fresh_views[-1]["item_box_iou_evaluator_only"] >= .5
        assert summary["fresh_item_error_m_evaluator_only"] < .1
    if summary["route_status"] != "running":
        assert not fresh_views
        assert summary["item_status"] == "not_rechecked"
    video = Path(ep["video"]["path"])
    assert sha(video) == ep["video"]["sha256"]
    with imageio.get_reader(video) as reader:
        frame_count = sum(1 for frame in reader if frame.ndim == 3 and
                          frame.shape[2] == 3)
    assert frame_count == ep["video"]["frames"]
    return {"source": summary["source"], "prop_motion": label,
            "owner_outcome": ep["outcome"],
            "route_status": summary["route_status"],
            "item_status": summary["item_status"],
            "moves": summary["moves"],
            "fixture_minimum_gap_m_capped_at_30cm": round(fixture_gap, 5),
            "proxy_minimum_gap_m_capped_at_10cm": round(minimum, 5),
            "post_stop_minimum_gap_m_capped_at_10cm": round(post_stop_minimum, 5),
            "first_5cm_breach": first_breach,
            "movement_samples_20ms": movement_samples,
            "guard_stop_checks": stopped_checks,
            "post_stop_samples_20ms": hold_samples,
            "predeparture_samples_20ms": watch_samples,
            "fresh_views": len(fresh_views),
            "video_frames": frame_count}


def main() -> None:
    declaration_path = OUT / "predeclared-moving-return.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == sha(declaration_path)
    assert declaration["source_sha256"] == sha(OUT / trial.SOURCE.name)
    assert declaration["memory_source_sha256"] == sha(OUT / memory.SOURCE.name)
    assert declaration["joined_source_sha256"] == sha(
        OUT / Path(joined.__file__).name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    rows = [audit_episode(ep, label) for ep, (_, label) in
            zip(report["episodes"], trial.PROFILES)]
    result = {"schema": "robot-learning-lab.robocasa-memory-moving-return-audit.v1",
              "declaration_sha256": sha(declaration_path),
              "report_sha256": sha(report_path),
              "videos_decoded": len(rows),
              "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "post_stop_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
