"""V134p: full early retreat and RGB-D show with +0.6 rad left shoulder.

Tests the more observable arm orientation from V133p in the same scripted
episode. Motion and arm posture remain prescribed kinematics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
from audit_direct_structural_entry_v271 import V149, V150
from audit_protective_postures_v327 import self_clearance
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_color_grounded_entry_v151 import infer
from run_moving_low_clutter_v302 import STOOL_SIZE
import run_pre_show_reserve_screen_v100 as v100
import compare_early_arm_orientation_v133p as v133p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-forward-arm"
STATION = v128p.STATIONS["back12"]
SHOULDER = .6


def travel(seed: int, source_case: tuple, qpos: np.ndarray,
           estimate: np.ndarray) -> dict:
    room = OfferDriftRoom(case=source_case, pose_source="command",
                          branch=OUT / "travel")
    try:
        room.layout(seed, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.head = float(qpos[room.head_addr])
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        yaw = 2*math.atan2(qpos[6], qpos[3])
        minima = {"scene": (float("inf"), None, None, None),
                  "self": (float("inf"), None, None, None)}
        samples = []
        for tick in range(61):
            t = 8.0+.02*tick
            fraction = tick/60
            room.actual[:] = (qpos[0]+fraction*STATION[0],
                              qpos[1]+fraction*STATION[1], yaw)
            room.estimated[:] = estimate
            room.estimated[:2] += fraction*np.asarray(STATION)
            room.data.qpos[shoulder] = SHOULDER*fraction
            room.time_s = t
            set_person_pose(room.model, START_HAND_XYZ_M,
                            carried_item="loop_TomatoSauce")
            room.obstacle_xy[3] = (0., .306)
            room.model.body_pos[room.model.body("loop_stool").id] = (
                0., .306, STOOL_SIZE[2])
            room._forward()
            scene = room._current_obstacle_clearance()
            own = self_clearance(room, "left")
            if scene[0] < minima["scene"][0]:
                minima["scene"] = (float(scene[0]), t, *scene[1:])
            if own[0] < minima["self"][0]:
                minima["self"] = (float(own[0]), t, *own[1:])
            if tick % 5 == 0:
                samples.append({"time_s": round(t, 3),
                                "evaluator_scene_gap_m": round(scene[0], 5),
                                "evaluator_self_gap_m": round(own[0], 5)})
        return {key: {"minimum_gap_m": round(value[0], 5),
                      "time_s": round(value[1], 4),
                      "closest_pair": list(value[2:])}
                for key, value in minima.items()} | {"samples_100ms": samples}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    original_model = structural.structural_model
    original_out = v100.OUT
    original_sequences = v100.SEQUENCES.copy()
    original_directions = v100.DIRECTIONS.copy()
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    v100.OUT = OUT
    v100.SEQUENCES["wait"] = ("wait",)
    v100.DIRECTIONS["wait"] = (0., 0.)
    rows = []
    try:
        for seed in v128p.SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (v128p.V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            path = travel(seed, tuple(prior["case"]), qpos, estimate)
            station_qpos = qpos.copy()
            station_qpos[:2] += STATION
            station_estimate = estimate.copy()
            station_estimate[:2] += STATION
            probe = OfferDriftRoom(case=tuple(prior["case"]),
                                   pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            station_qpos[shoulder] = SHOULDER
            for case in v128p.CASES_TO_VALIDATE:
                screen = v100.run_one(seed, case, "wait",
                                      station_qpos, station_estimate,
                                      tuple(prior["case"]))
                image = screen["owner_image"]
                path_image = ROOT / image["actor_frame_path"]
                assert sha(path_image) == image["actor_frame_sha256"]
                with np.load(path_image) as frame:
                    rgb = Image.fromarray(frame["rgb"].copy())
                    depth = frame["depth_m"].copy()
                    pose = (frame["camera_origin_m"].copy(),
                            frame["camera_rotation"].copy())
                depth[(depth < .08) | (depth > 1.5)] = np.nan
                started = time.perf_counter()
                choice = infer(vision, exemplar, owner_hist,
                               rgb, depth, pose, None, None)
                elapsed = time.perf_counter()-started
                box = image["evaluator_truth_box"]
                overlap = (iou(choice["accepted_box"], box)
                           if choice["accepted_box"] is not None and
                           box is not None else 0.)
                row = {"seed": seed, "case": case,
                       "source_packet_path": str(source.relative_to(ROOT)),
                       "source_packet_sha256": sha(source),
                       "travel": path, "owner_show": screen,
                       "actor_accepted_box": choice["accepted_box"],
                       "evaluator_box_iou": round(float(overlap), 5),
                       "local_inference_wall_s": round(elapsed, 4)}
                rows.append(row)
                print(seed, case, "travel", path["scene"]["minimum_gap_m"],
                      "self", path["self"]["minimum_gap_m"],
                      "show", screen["minimum_scene_gap_m"],
                      "item", choice["accepted_box"] is not None,
                      flush=True)
    finally:
        v100.OUT = original_out
        v100.SEQUENCES.clear()
        v100.SEQUENCES.update(original_sequences)
        v100.DIRECTIONS.clear()
        v100.DIRECTIONS.update(original_directions)
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-early-forward-arm.v1",
              "source_sha256": sha(SOURCE),
              "v133_report_sha256": sha(v133p.OUT / "report.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "exemplar_sha256": sha(V149),
              "owner_hist_sha256": sha(V150),
              "scope": "Evaluator-only direct1.2s retreat with +0.6rad "
              "left shoulder from8.0s and self/scene gap sampling20ms. "
              "Owner waits to16.6s, shows through17.0s, and lower RGB-D "
              "at17.4s feeds the generic taught-item gate. Four block cases "
              "and two lighting seeds. A side block appears after travel. "
              "No actor-authorized route, motor dynamics, contact response "
              "or selected physical body is demonstrated.",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
