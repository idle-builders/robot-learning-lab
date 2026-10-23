"""Independent replay of V686 owner-station and memory-return episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as memory_audit
import audit_robocasa_person_standoff_v508 as meeting_audit
import run_robocasa_memory_joint_station_v686 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT
BASELINE = trial.ROOT / "artifacts/ape-robocasa-memory-meeting-standoff-v680/report.json"
PREFLIGHT = trial.ROOT / "artifacts/ape-robocasa-meeting-refuge-preflight-v685/report.json"


def route_core(ep: dict) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "rgb_sha256"}
            for row in ep["route_episode"]["trace"]]


def map_segment(snapshot: dict, origin: np.ndarray, goal: np.ndarray) -> bool:
    xs = np.asarray(snapshot["xs_m"])
    ys = np.asarray(snapshot["ys_m"])
    free = {tuple(cell) for cell in snapshot["free_indices_yx"]}
    occupied = {tuple(cell) for cell in snapshot["occupied_indices_yx"]}
    distance = np.linalg.norm(goal - origin)
    for fraction in np.linspace(0., 1., max(2, int(np.ceil(distance / .04)) + 1)):
        xy = origin + fraction * (goal - origin)
        cell = (int(np.argmin(np.abs(ys - xy[1]))),
                int(np.argmin(np.abs(xs - xy[0]))))
        if cell not in free or cell in occupied:
            return False
    return True


def audit_station(ep: dict, snapshot: dict) -> dict:
    first_watch = next(row for row in ep["trace"] if row["stage"] ==
                       "predeparture_guard")
    pre_memory = {**ep,
                  "final_actual_xy_m": first_watch["evaluator_actual_xy_m"],
                  "hold_until_s": round(first_watch["time_s"] - .02, 5)}
    physical = meeting_audit.audit_episode(pre_memory)
    assert physical["scene_fingerprint_pass"]
    assert physical["video_sha_pass"] and physical["video_frames_pass"]
    assert physical["ape_5cm_margin_pass"]
    assert physical["person_room_5cm_pass"]
    assert physical["can_room_nonoverlap_pass"]
    choice = next(row for row in ep["trace"] if row["stage"] ==
                  "choose_owner_standoff")["chosen"]
    assert choice is not None and choice["selected_refuge"] is not None
    start = np.asarray(snapshot["estimated_xy_m"])
    station = np.asarray(choice["target_xy_est_m"])
    refuge = np.asarray(choice["selected_refuge"]["target_xy_est_m"])
    assert np.linalg.norm(station - (start + [.3, -.9])) < .00001
    assert np.linalg.norm(refuge - (station + [-.2, -.2])) < .00001
    assert map_segment(snapshot, start, station)
    assert map_segment(snapshot, station, refuge)
    track = snapshot["tracks_actor"][0]
    envelope = track["radius_m"] + .17 + .05 + .03 + .60
    assert np.linalg.norm(station - track["center_xy_m"]) - envelope >= .25
    assert np.linalg.norm(refuge - track["center_xy_m"]) - envelope >= .25
    moves = [row for row in ep["trace"] if row["stage"] == "standoff_backtrack"]
    assert len(moves) == round(np.linalg.norm(station - start) / .25 / .02)
    assert np.linalg.norm(np.asarray(moves[-1]["ape_xy_m"]) -
                          np.asarray(first_watch["evaluator_actual_xy_m"])) < .00001
    assert all(row["skill_status"] == "kinematic_surrogate_unqualified"
               for row in moves)
    return {"station_xy_est_m": station.tolist(),
            "refuge_xy_est_m": refuge.tolist(),
            "owner_show_motion_samples_20ms": len(moves),
            "pre_memory_geometry_samples": physical["geometry_samples"],
            "pre_memory_minimum_gap_m_capped_at_10cm": physical[
                "nearest"]["ape_any"]["gap_m_capped_at_10cm"]}


def main() -> None:
    declaration_path = OUT / "predeclared.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    baseline = json.loads(BASELINE.read_text())
    snapshot = json.loads(PREFLIGHT.read_text())["actor_map"]
    assert report["declaration_sha256"] == memory.digest(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 6
    assert declaration["source_sha256"] == memory.digest(OUT / trial.SOURCE.name)
    assert declaration["selector_source_sha256"] == memory.digest(
        OUT / Path(trial.joint_meeting_refuge.__file__).name)
    assert declaration["standoff_source_sha256"] == memory.digest(
        OUT / Path(trial.standoff.__file__).name)
    assert declaration["memory_source_sha256"] == memory.digest(
        OUT / memory.SOURCE.name)
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    old_meeting_motion = meeting_audit.motion
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = 20.8
        for ep, (direction, source) in zip(report["episodes"], trial.PROFILES):
            crossing.prop_motion = old_motion
            script = (old_motion("stationary") if direction == "stationary"
                      else trial.bearing.challenge_motion(direction))
            crossing.prop_motion = lambda _, script=script: script
            meeting_audit.motion = lambda _, script=script: script
            assert ep["scene_fingerprint"] == baseline["episodes"][0][
                "scene_fingerprint"]
            assert route_core(ep) == route_core(baseline["episodes"][0])
            assert ep["outcome"] == "owner_confirmed_metric_find"
            assert ep["memory_return"]["source"] == source
            assert ep["memory_return"]["predeparture_confirmed_track_hold_select_requested"]
            assert not ep["memory_return"]["evaluator_abort_on_margin_breach_requested"]
            station = audit_station(ep, snapshot)
            memory_result = memory_audit.audit_episode(ep, direction)
            assert memory_result["proxy_minimum_gap_m_capped_at_10cm"] >= .05
            rows.append({"direction": direction, "source": source,
                         **station, **memory_result})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
        meeting_audit.motion = old_meeting_motion
    result = {
        "schema": "robot-learning-lab.robocasa-memory-joint-station-audit.v1",
        "declaration_sha256": memory.digest(declaration_path),
        "report_sha256": memory.digest(report_path),
        "preflight_sha256": memory.digest(PREFLIGHT),
        "baseline_sha256": memory.digest(BASELINE),
        "scope": "Actor-map station/refuge arithmetic, paired pre-owner route, independent 20ms owner/memory MuJoCo geometry and complete video decode; one room and one range seed only",
        "videos_decoded": len(rows),
        "owner_show_motion_samples_20ms": sum(row[
            "owner_show_motion_samples_20ms"] for row in rows),
        "pre_memory_geometry_samples": sum(row["pre_memory_geometry_samples"]
                                           for row in rows),
        "memory_movement_samples_20ms": sum(row["movement_samples_20ms"]
                                            for row in rows),
        "memory_held_samples_20ms": sum(row["post_stop_samples_20ms"]
                                        for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
