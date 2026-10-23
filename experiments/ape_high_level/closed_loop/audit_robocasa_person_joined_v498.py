"""Independent full-horizon geometry and video audit of joined owner episodes.

Reconstructs the seeded room and replays the executed ape route, scripted
hand/item positions, and moving prop at 20 ms intervals through the hold.
The evaluator uses MuJoCo signed distances; the actor did not receive them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from run_robocasa_dynamic_suite_v439 import motion
from screen_robocasa_owner_person_v489 import gap_to_room


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "artifacts/ape-robocasa-person-joined-v497/report.json"
OUT = ROOT / "artifacts/ape-robocasa-person-joined-v497/audit-v498.json"
SUBSTEP_S = .02


def nearest_can_room(model, data):
    can = model.geom("shown_can_geom").id
    ape = model.geom("audit_ape_geom").id
    people = {model.geom(name).id for name in person.PERSON_GEOMS}
    best = {"gap_m_capped_at_10cm": .1, "geom": None}
    for other in range(model.ngeom):
        if other in people | {can, ape} or model.geom(other).name.startswith("floor_"):
            continue
        if np.linalg.norm(data.geom_xpos[other] - data.geom_xpos[can]) > (
                model.geom_rbound[other] + model.geom_rbound[can] + .1):
            continue
        gap = float(mujoco.mj_geomDistance(model, data, can, other, .1, None))
        if gap < best["gap_m_capped_at_10cm"]:
            best = {"gap_m_capped_at_10cm": gap, "geom": model.geom(other).name}
    return best


def audit_episode(ep):
    case = ep["scenario"]
    model = joined.scene_with_owner_and_prop(case["from_xy_m"])
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    can_body = model.body("shown_can").id
    script = motion(case)
    events = []
    for row in ep["trace"]:
        if row["stage"] == "owner_reaches":
            events.append((float(row["time_s"]), row["pose_evaluator"]))
        elif row["stage"] == "owner_withdraws_during_recognition":
            events.append((float(row["time_s"]), None))
    events.sort(key=lambda item: item[0])
    min_ape = {"signed_gap_m_capped_at_10cm": .1, "geom": None,
               "time_s": None, "phase": None}
    min_prop = {"signed_gap_m_capped_at_10cm": .1,
                "time_s": None, "phase": None}
    min_person_ape = {"gap_m_capped_at_50cm": .5,
                      "time_s": None, "person_geom": None}
    min_person_room = {"gap_m_capped_at_30cm": .3,
                       "time_s": None, "pair": None}
    min_can_room = {"gap_m_capped_at_10cm": .1,
                    "time_s": None, "geom": None}
    samples = 0

    def sample(t, xy, phase):
        nonlocal min_ape, min_prop, min_person_ape, min_person_room, min_can_room, samples
        model.body_pos[prop_body, :2] = script(t)
        prior = [entry for time_s, entry in events if time_s <= t + 1e-9]
        state = prior[-1] if prior else "rest"
        if state == "rest":
            person.set_owner_pose(model, interaction.TORSO_XY,
                                  interaction.REST_HAND, show_can=False)
        elif state is None:
            person.set_owner_pose(model, interaction.TORSO_XY,
                                  interaction.SHOW_HAND, show_can=False)
        else:
            person.set_owner_pose(model, interaction.TORSO_XY,
                                  state["hand_xyz_m"], show_can=True)
            model.body_pos[can_body] = state["can_xyz_m"]
        gap, near = clearance(model, data, ape_body, ape_geom, np.asarray(xy))
        if gap < min_ape["signed_gap_m_capped_at_10cm"]:
            min_ape = {"signed_gap_m_capped_at_10cm": gap, "geom": near,
                       "time_s": round(t, 4), "phase": phase}
        prop_gap = float(mujoco.mj_geomDistance(model, data, ape_geom,
                                                prop_geom, .1, None))
        if prop_gap < min_prop["signed_gap_m_capped_at_10cm"]:
            min_prop = {"signed_gap_m_capped_at_10cm": prop_gap,
                        "time_s": round(t, 4), "phase": phase}
        if phase in ("owner_event", "route_capture", "hold_end"):
            room, person_ape = gap_to_room(model, data)
            if person_ape["gap_m_capped_at_50cm"] < min_person_ape["gap_m_capped_at_50cm"]:
                min_person_ape = {**person_ape, "time_s": round(t, 4)}
            if room["gap_m_capped_at_30cm"] < min_person_room["gap_m_capped_at_30cm"]:
                min_person_room = {**room, "time_s": round(t, 4)}
            if state not in ("rest", None):
                can_room = nearest_can_room(model, data)
                if can_room["gap_m_capped_at_10cm"] < min_can_room["gap_m_capped_at_10cm"]:
                    min_can_room = {**can_room, "time_s": round(t, 4)}
        samples += 1

    route = ep["route_episode"]
    for row in route["trace"]:
        start = np.asarray(row["actual_xy_m"])
        sample(float(row["time_s"]), start, "route_capture")
        requested = row.get("requested_xy_m")
        if requested is None:
            continue
        delta = ((np.asarray(requested) - np.asarray(row["estimated_xy_m"]))
                 * (1. + .30 / 5.89))
        for index, sub in enumerate(row["motion_substeps"]):
            fraction = (index if sub["stopped_before_move"] else index + 1) / 20
            sample(float(sub["time_s"]), start + fraction * delta, "executed_move")
    held = np.asarray(ep["final_actual_xy_m"])
    start_time = float(route["end_time_s"])
    end_time = float(ep["hold_until_s"])
    extra_times = [t for t, _ in events]
    times = np.unique(np.r_[np.arange(start_time + SUBSTEP_S, end_time + .01,
                                      SUBSTEP_S), extra_times, end_time])
    for t in times:
        phase = "owner_event" if any(abs(t - e[0]) < 1e-8 for e in events) else (
            "hold_end" if abs(t - end_time) < 1e-8 else "stationary_hold")
        sample(float(t), held, phase)
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frame_count = reader.count_frames()
    reader.close()
    return {"scenario_id": case["id"], "response": ep["response"],
            "depth": ep["depth"], "outcome": ep["outcome"],
            "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
            "nearest_ape_any": min_ape, "nearest_ape_prop": min_prop,
            "nearest_person_ape": min_person_ape,
            "nearest_person_room": min_person_room,
            "nearest_can_room": min_can_room,
            "ape_5cm_margin_pass": min_ape["signed_gap_m_capped_at_10cm"] >= .05,
            "person_room_5cm_pass": min_person_room["gap_m_capped_at_30cm"] >= .05,
            "can_room_nonoverlap_pass": min_can_room["gap_m_capped_at_10cm"] >= 0.,
            "video_sha_pass": sha256(video) == ep["video"]["sha256"],
            "video_frames_pass": frame_count == ep["video"]["frames"],
            "geometry_samples": samples}


def main():
    report = json.loads(TRIAL.read_text())
    results = []
    for ep in report["episodes"]:
        result = audit_episode(ep)
        results.append(result)
        print(result["scenario_id"], result["response"], result["depth"],
              result["outcome"], "ape", result["nearest_ape_any"],
              "person", result["nearest_person_room"],
              "can", result["nearest_can_room"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(TRIAL),
              "trial_source_sha_pass": report["source_sha256"] == sha256(Path(joined.__file__)),
              "person_source_sha_pass": report["person_source_sha256"] == sha256(Path(person.__file__)),
              "interaction_source_sha_pass": report["person_interaction_source_sha256"] == sha256(
                  Path(interaction.__file__)),
              "all_videos_pass": all(x["video_sha_pass"] and x["video_frames_pass"]
                                     for x in results),
              "scope": "MuJoCo signed geometry distances across executed route, owner hand/item states and stationary hold at 20 ms through dynamic crossing; scripted person and kinematic ape cylinder",
              "episodes": results}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
