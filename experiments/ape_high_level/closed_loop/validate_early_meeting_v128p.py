"""V128p: replay two early meeting stations through travel and owner show.

Travel and shoulder motion are prescribed kinematics. The owner waits until
16.6s; this verifies geometry and rendered taught-item evidence only.
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
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
import run_pre_show_reserve_screen_v100 as v100
import screen_early_meeting_offset_v127p as v127p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-meeting-validation"
V98 = v100.V98
STATIONS = {"back12": (-.12, 0.), "back16_left4": (-.16, -.04)}
SEEDS = (170, 172)
CASES_TO_VALIDATE = tuple(CASES)
TARGET_SHOULDER_RAD = -.6
TRAVEL_START_S = 8.0
TRAVEL_END_S = 9.2


def travel(seed: int, source_case: tuple, qpos: np.ndarray,
           estimate: np.ndarray, offset_xy: tuple[float, float]) -> dict:
    room = OfferDriftRoom(case=source_case, pose_source="command",
                          branch=OUT / "travel")
    try:
        room.layout(seed, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.estimated[:] = estimate
        room.head = float(qpos[room.head_addr])
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        initial_shoulder = float(qpos[shoulder])
        yaw = 2*math.atan2(qpos[6], qpos[3])
        minimum = (float("inf"), None, None, None)
        samples = []
        for tick in range(61):
            t = TRAVEL_START_S + .02*tick
            fraction = tick/60
            room.actual[:] = (qpos[0] + fraction*offset_xy[0],
                              qpos[1] + fraction*offset_xy[1], yaw)
            room.estimated[:] = estimate
            room.estimated[:2] += fraction*np.asarray(offset_xy)
            room.data.qpos[shoulder] = (initial_shoulder + fraction*
                                        (TARGET_SHOULDER_RAD-initial_shoulder))
            room.time_s = t
            set_person_pose(room.model, START_HAND_XYZ_M,
                            carried_item="loop_TomatoSauce")
            room.obstacle_xy[3] = (0., .306)
            room.model.body_pos[room.model.body("loop_stool").id] = (
                0., .306, STOOL_SIZE[2])
            room._forward()
            gap, robot, obstacle = room._current_obstacle_clearance()
            if gap < minimum[0]:
                minimum = (float(gap), t, robot, obstacle)
            if tick % 5 == 0:
                samples.append({"time_s": round(t, 3),
                                "actual_xy_m": room.actual[:2].round(5).tolist(),
                                "evaluator_gap_m": round(float(gap), 5)})
        return {"minimum_scene_gap_m": round(minimum[0], 5),
                "minimum_time_s": round(minimum[1], 4),
                "closest_pair": list(minimum[2:]),
                "samples_100ms": samples}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v127 = json.loads((v127p.OUT / "screen.json").read_text())
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
    v100.SEQUENCES["wait"] = ("wait",)
    v100.DIRECTIONS["wait"] = (0., 0.)
    rows = []
    try:
        for seed in SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
                assert abs(float(packet["capture_time_s"])-7.6) < 1e-5
            for station, offset in STATIONS.items():
                path = travel(seed, tuple(prior["case"]), qpos, estimate,
                              offset)
                station_qpos = qpos.copy()
                station_qpos[:2] += offset
                station_estimate = estimate.copy()
                station_estimate[:2] += offset
                # Path reaches this joint target well before the owner starts.
                probe = OfferDriftRoom(case=tuple(prior["case"]),
                                       pose_source="command", branch=OUT)
                try:
                    shoulder = probe.model.joint(
                        "left_shoulder_pitch").qposadr[0]
                finally:
                    probe.close()
                station_qpos[shoulder] = TARGET_SHOULDER_RAD
                v100.OUT = OUT / station
                for case in CASES_TO_VALIDATE:
                    screen = v100.run_one(seed, case, "wait",
                                          station_qpos, station_estimate,
                                          tuple(prior["case"]))
                    frame_path = ROOT / screen["owner_image"]["actor_frame_path"]
                    if sha(frame_path) != screen["owner_image"]["actor_frame_sha256"]:
                        raise AssertionError("owner frame hash changed")
                    with np.load(frame_path) as frame:
                        rgb = Image.fromarray(frame["rgb"].copy())
                        depth = frame["depth_m"].copy()
                        pose = (frame["camera_origin_m"].copy(),
                                frame["camera_rotation"].copy())
                    depth[(depth < .08) | (depth > 1.5)] = np.nan
                    started = time.perf_counter()
                    choice = infer(vision, exemplar, owner_hist,
                                   rgb, depth, pose, None, None)
                    elapsed = time.perf_counter()-started
                    box = screen["owner_image"]["evaluator_truth_box"]
                    overlap = (iou(choice["accepted_box"], box)
                               if choice["accepted_box"] is not None and
                               box is not None else 0.)
                    prior_screen = next((row for row in v127["rows"]
                                         if row["case"] == case and
                                         row["meeting_offset_xy_m"] == list(offset)),
                                        None)
                    row = {"seed": seed, "station": station, "case": case,
                           "meeting_offset_xy_m": list(offset),
                           "source_packet_path": str(source.relative_to(ROOT)),
                           "source_packet_sha256": sha(source),
                           "travel": path, "owner_show": screen,
                           "v127_transition_gap_m": (prior_screen[
                               "minimum_scene_gap_m"]
                               if prior_screen is not None else None),
                           "actor_proposals": int(choice["proposals"]),
                           "actor_accepted_box": choice["accepted_box"],
                           "evaluator_box_iou": round(float(overlap), 5),
                           "local_inference_wall_s": round(elapsed, 4)}
                    rows.append(row)
                    print(seed, station, case,
                          "travel", path["minimum_scene_gap_m"],
                          "show", screen["minimum_scene_gap_m"],
                          "item", choice["accepted_box"] is not None,
                          "iou", row["evaluator_box_iou"], flush=True)
    finally:
        v100.OUT = original_out
        v100.SEQUENCES.clear()
        v100.SEQUENCES.update(original_sequences)
        v100.DIRECTIONS.clear()
        v100.DIRECTIONS.update(original_directions)
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-meeting-validation.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v127_report_sha256": sha(v127p.OUT / "screen.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "exemplar_sha256": sha(V149),
        "owner_hist_sha256": sha(V150),
        "scope": "Evaluator-only prescribed1.2s travel from the saved7.6s "
        "pose, starting at8.0s after a possible observation window, "
        "staging pose to two early owner-show stations in a clear scene. "
        "The hand-height side block, when present, appears by16.2s after "
        "travel. Owner waits until16.6s, shows over0.4s, lower RGB-D is "
        "captured17.4s, and moving prop is checked through18.2s. Generic "
        "taught-item gate runs on every rendered image. Clearance is "
        "evaluator truth. No actor selection, route-map support, learned "
        "locomotion, qualified posture or selected physical body is claimed.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
