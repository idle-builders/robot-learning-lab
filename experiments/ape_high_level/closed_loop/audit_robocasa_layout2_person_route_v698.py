"""Independent layout-2 geometry and saved-media check for the RGB-D owner run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_owner_transfer_v695 as transfer
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from audit_robocasa_person_joined_v498 import nearest_can_room
from screen_robocasa_owner_person_v489 import gap_to_room


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
TRIAL = ROOT / "artifacts/ape-robocasa-layout2-person-route-v697/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout2-person-route-v697/audit-v698.json"
DT = .02


def main():
    report = json.loads(TRIAL.read_text())
    ep = report["episode"]
    model = person.scene_with_person(transfer.PROP_START, layout_id=2,
                                     style_id=1, clutter_mode=0)
    torso = interaction.TORSO_XY + transfer.OWNER_SHIFT
    rest = interaction.REST_HAND + [*transfer.OWNER_SHIFT, 0.]
    person.set_owner_pose(model, torso, rest, show_can=False)
    fingerprint = hashlib.sha256(b"".join(array.tobytes() for array in
        (model.body_pos, model.geom_pos, model.geom_quat,
         model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    minima = {"gap_m_capped_at_10cm": .1, "time_s": None,
              "nearest_geom": None, "phase": None}
    samples = 0

    def sample(t, xy, hand, phase):
        nonlocal samples, minima
        if hand is None:
            person.set_owner_pose(model, torso, rest, show_can=False)
        else:
            hand = np.asarray(hand)
            person.set_owner_pose(model, torso, hand, show_can=True)
            direction = hand[:2] - torso
            direction /= np.linalg.norm(direction)
            model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * direction
        model.body_pos[model.body("route_obstacle").id, :2] = transfer.PROP_START
        gap, geom = clearance(model, data, ape_body, ape_geom, np.asarray(xy))
        if gap < minima["gap_m_capped_at_10cm"]:
            minima = {"gap_m_capped_at_10cm": gap, "time_s": round(t, 4),
                      "nearest_geom": geom, "phase": phase}
        samples += 1

    route_rows = ep["route_episode"]["trace"]
    first = route_rows[0]
    for t in np.arange(0., first["time_s"] + DT / 2, DT):
        sample(float(t), first["actual_xy_m"], None, "initial_perception_hold")
    for row in route_rows:
        start = np.asarray(row["actual_xy_m"])
        sample(row["time_s"], start, None, "route_capture")
        if row["motion_substeps"]:
            requested = np.asarray(row["requested_xy_m"])
            delta = (requested - np.asarray(row["estimated_xy_m"])) * (1. + .30 / 5.89)
            for i, sub in enumerate(row["motion_substeps"]):
                fraction = (i if sub["stopped_before_move"] else i + 1) / 20
                sample(sub["time_s"], start + fraction * delta, None, "route_motion_20ms")
    reached = next(row for row in ep["trace"]
                   if row.get("stage") == "fresh_person_rgbd_before_owner_request")
    arrival = ep["route_episode"]["end_true_xy_m"]
    for t in np.arange(ep["route_episode"]["end_time_s"] + DT,
                       reached["time_s"] + DT / 2, DT):
        sample(float(t), arrival, None, "fresh_person_hold")
    reach_rows = [row for row in ep["trace"] if row.get("stage") == "owner_reaches"]
    previous_time = reached["time_s"]
    previous_hand = rest.copy()
    for row in reach_rows:
        target = np.asarray(row["pose_evaluator"]["hand_xyz_m"])
        for t in np.arange(previous_time + DT, row["time_s"] + DT / 2, DT):
            alpha = np.clip((t - previous_time) / (row["time_s"] - previous_time), 0., 1.)
            sample(float(t), arrival, previous_hand + alpha * (target - previous_hand),
                   "owner_reach_20ms")
        previous_time, previous_hand = row["time_s"], target
    for row in ep["trace"]:
        if row.get("stage") == "memory_route_decision":
            sample(row["time_s"], row["evaluator_actual_xy_m"], previous_hand,
                   "memory_route_capture")
            for sub in row["motion_substeps"]:
                sample(sub["time_s"], sub["evaluator_actual_xy_m"], previous_hand,
                       "memory_motion_20ms")
        elif row.get("stage") in ("memory_viewpoint_reached", "memory_fresh_view"):
            sample(row["time_s"], row["evaluator_actual_xy_m"], previous_hand,
                   row["stage"])
    sample(ep["hold_until_s"], ep["final_actual_xy_m"], previous_hand, "final")
    room, _ = gap_to_room(model, data)
    can_room = nearest_can_room(model, data)
    video = Path(ep["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    packet_rows = [row for row in ep["trace"] if row.get("stage") == "memory_fresh_view"]
    packet_hashes_pass = all(
        sha256(Path(row["image_path"])) == row["image_sha256"] and
        sha256(Path(row["depth_path"])) == row["depth_sha256"]
        for row in packet_rows)
    result = {"schema": "robot-learning-lab.layout2-person-route-audit.v1",
              "source_sha256": sha256(SOURCE),
              "trial_report_sha256": sha256(TRIAL),
              "scene_fingerprint_pass": fingerprint == ep["scene_fingerprint"],
              "geometry_samples": samples,
              "minimum_signed_clearance": minima,
              "ape_5cm_margin_pass": minima["gap_m_capped_at_10cm"] >= .05,
              "person_room_5cm_pass": room["gap_m_capped_at_30cm"] >= .05,
              "can_room_nonoverlap_pass": can_room["gap_m_capped_at_10cm"] >= 0.,
              "video_sha_pass": sha256(video) == ep["video"]["sha256"],
              "video_frames_pass": frames == ep["video"]["frames"],
              "video_frames": frames,
              "saved_rgbd_packet_hashes_pass": packet_hashes_pass,
              "actor_memory_route_reached": any(row.get("stage") == "memory_viewpoint_reached" and
                                                  row.get("phase") == "return"
                                                  for row in ep["trace"]),
              "actor_fresh_metric_item_refound": ep["memory_return"]["item_status"] ==
                                                  "fresh_metric_item_refound",
              "scope": "Rebuilt layout2 owner scene, actual route/return 20ms positions, stationary perception holds, interpolated owner reach, MuJoCo 50cm proxy geometry, and saved media integrity. Static low prop; no dynamic owner or prop qualification."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
