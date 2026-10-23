"""V142p: transfer measured native-walk pose error to the owner meeting.

Three saved 50 Hz Direct Frame walks supply first-12cm foot/heading/range
error shapes. They perturb the prescribed historical owner-room retreat;
this is not an actual reverse policy in that room.
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
import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from pre_show_actor_planner import ROUTE_FREE_AGE_S, freshness_filtered
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_color_grounded_entry_v151 import infer
from run_moving_low_clutter_v302 import STOOL_SIZE
import run_pre_show_reserve_screen_v100 as v100
import compare_early_arm_orientation_v133p as v133p
import screen_rear_left_up_probe_v135p as v135p
import validate_early_forward_arm_v134p as v134p
import validate_early_meeting_v128p as v128p
import screen_early_retreat_observability_v129p as v129p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-native-pose-meeting"
NATIVE_REPORT = ROOT / "artifacts/ape-native-gait-heading/v1/report.json"
SEED = 170
CASES = ("clear", "hand_close")
DESIRED_OFFSET = np.asarray(v128p.STATIONS["back12"])
SHOULDER_RAD = v134p.SHOULDER


def native_errors() -> list[dict]:
    report = json.loads(NATIVE_REPORT.read_text())
    rows = []
    for source in report["cases"]:
        if source["arm"] != "range_360":
            continue
        trace_path = ROOT / source["trace"]
        if sha(trace_path) != source["trace_sha256"]:
            raise AssertionError("native trace changed")
        frames = json.loads(trace_path.read_text())
        start = np.asarray(frames[0]["native_xy_evaluator_only_m"])
        last = start.copy()
        arc = 0.
        selected = None
        for frame in frames[1:]:
            actual = np.asarray(frame["native_xy_evaluator_only_m"])
            arc += float(np.linalg.norm(actual-last))
            last = actual
            if arc >= .12:
                selected = frame
                break
        if selected is None:
            raise AssertionError("native walk never reached12cm")
        actual = np.asarray(selected["native_xy_evaluator_only_m"])
        estimate = np.asarray(selected["estimated_xy_m"])
        world_error = actual-estimate
        native_heading = math.atan2((actual-start)[1], (actual-start)[0])
        rotate = math.pi-native_heading
        c, s = math.cos(rotate), math.sin(rotate)
        backward_error = np.array([[c, -s], [s, c]]) @ world_error
        yaw_error = (float(selected["native_yaw_evaluator_only_rad"])-
                     float(selected["estimated_yaw_rad"]))
        rows.append({
            "native_case": source["case"],
            "trace_path": source["trace"],
            "trace_sha256": source["trace_sha256"],
            "frame": selected["frame"],
            "capture_time_s": selected["t_s"],
            "cumulative_native_arc_m": round(arc, 5),
            "native_estimator_reported_healthy": bool(
                selected["actor_belief_healthy"]),
            "world_actual_minus_estimated_xy_m": world_error.round(6).tolist(),
            "backward_aligned_actual_minus_estimated_xy_m": (
                backward_error.round(6).tolist()),
            "actual_minus_estimated_yaw_rad": round(yaw_error, 6),
        })
    if len(rows) != 3:
        raise AssertionError("expected three native paths")
    return rows


def path_screen(room: OfferDriftRoom, source_qpos: np.ndarray,
                source_estimate: np.ndarray,
                actual_offset: np.ndarray, yaw_error: float) -> dict:
    room.layout(SEED, absent=False, furnished=True)
    room.data.qpos[:] = source_qpos
    room.head = float(source_qpos[room.head_addr])
    shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
    start_yaw = 2*math.atan2(source_qpos[6], source_qpos[3])
    scene_min = (float("inf"), None, None, None)
    self_min = (float("inf"), None, None, None)
    samples = []
    for tick in range(61):
        t = 8.0+.02*tick
        fraction = tick/60
        yaw = start_yaw+fraction*yaw_error
        room.actual[:] = (source_qpos[0]+fraction*actual_offset[0],
                          source_qpos[1]+fraction*actual_offset[1], yaw)
        room.estimated[:] = source_estimate
        room.estimated[:2] += fraction*DESIRED_OFFSET
        room.data.qpos[shoulder] = fraction*SHOULDER_RAD
        room.time_s = t
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room.obstacle_xy[3] = (0., .306)
        room.model.body_pos[room.model.body("loop_stool").id] = (
            0., .306, STOOL_SIZE[2])
        room._forward()
        scene = room._current_obstacle_clearance()
        own = self_clearance(room, "left")
        if scene[0] < scene_min[0]:
            scene_min = (float(scene[0]), t, *scene[1:])
        if own[0] < self_min[0]:
            self_min = (float(own[0]), t, *own[1:])
        if tick % 5 == 0:
            samples.append({"time_s": round(t, 3),
                            "actual_pose_xyyaw": room.actual.round(5).tolist(),
                            "estimated_pose_xyyaw": room.estimated.round(5).tolist(),
                            "evaluator_scene_gap_m": round(scene[0], 5),
                            "evaluator_self_gap_m": round(own[0], 5)})
    return {"minimum_scene_gap_m": round(scene_min[0], 5),
            "minimum_scene_time_s": round(scene_min[1], 4),
            "closest_scene_pair": list(scene_min[2:]),
            "minimum_self_gap_m": round(self_min[0], 5),
            "minimum_self_time_s": round(self_min[1], 4),
            "closest_self_pair": list(self_min[2:]),
            "samples_100ms": samples}


def route_screen(robot: RobotGeometry, voxel_map,
                 qpos: np.ndarray, estimated: np.ndarray,
                 actual_offset: np.ndarray, yaw_error: float) -> dict:
    start = estimated.copy()
    end = estimated.copy()
    end[:2] += actual_offset
    end[2] += yaw_error
    result = {}
    shoulder = robot.model.joint("left_shoulder_pitch").qposadr[0]
    for label, angle in (("neutral", 0.), ("left_forward", SHOULDER_RAD)):
        posture_qpos = qpos.copy()
        posture_qpos[shoulder] = angle
        result[label] = {
            height: robot.swept_exterior_coverage(
                voxel_map, posture_qpos, start, end, v129p.DECISION_S,
                margin_m=.05, z_min_m=lo, z_max_m=hi)
            for height, lo, hi in (("lower", .02, .18),
                                   ("upper", .18, .43))}
    statuses = [band["status"] for posture in result.values()
                for band in posture.values()]
    return {"coverage_status": (
                "observed_free" if all(x == "free" for x in statuses) else
                "occupied" if "occupied" in statuses else "unknown"),
            "posture_samples": result}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    native = native_errors()
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == SEED)
    source = (v128p.V98 / "moving" / f"seed{SEED}-present" /
              f"seed{SEED}-present/side-probe-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    v129 = json.loads((v129p.OUT / "report.json").read_text())
    record = next(row for row in v129["rows"] if row["seed"] == SEED)
    v135 = json.loads((v135p.OUT / "report.json").read_text())
    probe_record = next(row for row in v135["rows"] if row["seed"] == SEED)
    original_model = structural.structural_model
    original_out = v100.OUT
    original_sequences = v100.SEQUENCES.copy()
    original_directions = v100.DIRECTIONS.copy()
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    v100.SEQUENCES["wait"] = ("wait",)
    v100.DIRECTIONS["wait"] = (0., 0.)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    try:
        room = OfferDriftRoom(case=tuple(prior["case"]),
                              pose_source="command", branch=OUT)
        try:
            base_map = v133p.map_at_eight(record)
            probe_path = ROOT / probe_record["new_packet_path"]
            assert sha(probe_path) == probe_record["new_packet_sha256"]
            probe_rays = json.loads(probe_path.read_text())[
                "ray_packets"]["rear_mid_left_up8"]
            sensor.add_packet(base_map, probe_rays, v129p.DECISION_S)
            base_map = freshness_filtered(
                base_map, v129p.DECISION_S, ROUTE_FREE_AGE_S)
            robot = RobotGeometry(room.model)
            shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
            for entry in native:
                aligned = np.asarray(entry[
                    "backward_aligned_actual_minus_estimated_xy_m"])
                for mirror in (False, True):
                    error = aligned.copy()
                    if mirror:
                        error[1] *= -1
                    yaw_error = float(entry[
                        "actual_minus_estimated_yaw_rad"])
                    if mirror:
                        yaw_error *= -1
                    actual_offset = DESIRED_OFFSET+error
                    path = path_screen(room, qpos, estimate,
                                       actual_offset, yaw_error)
                    route = route_screen(robot, base_map, qpos, estimate,
                                         actual_offset, yaw_error)
                    actual_qpos = qpos.copy()
                    actual_qpos[:2] += actual_offset
                    actual_qpos[3:7] = [math.cos(yaw_error/2),
                                           0., 0., math.sin(yaw_error/2)]
                    actual_qpos[shoulder] = SHOULDER_RAD
                    station_estimate = estimate.copy()
                    station_estimate[:2] += DESIRED_OFFSET
                    label = (entry["native_case"].replace("/", "_")+
                             ("-mirror" if mirror else "-direct"))
                    v100.OUT = OUT / label
                    for case in CASES:
                        screen = v100.run_one(SEED, case, "wait",
                                              actual_qpos, station_estimate,
                                              tuple(prior["case"]))
                        image_meta = screen["owner_image"]
                        frame_path = ROOT / image_meta["actor_frame_path"]
                        assert sha(frame_path) == image_meta[
                            "actor_frame_sha256"]
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
                        box = image_meta["evaluator_truth_box"]
                        overlap = (iou(choice["accepted_box"], box)
                                   if choice["accepted_box"] is not None
                                   and box is not None else 0.)
                        row = {"native_case": entry["native_case"],
                               "mirrored_lateral_error": mirror,
                               "case": case,
                               "actual_minus_estimated_xy_m": error.round(6).tolist(),
                               "actual_minus_estimated_yaw_rad": round(
                                   yaw_error, 6),
                               "actual_station_offset_xy_m": actual_offset.round(6).tolist(),
                               "native_error_source_frame": entry["frame"],
                               "actor_swept_route": route,
                               "prescribed_travel": path,
                               "owner_show": screen,
                               "actor_accepted_box": choice["accepted_box"],
                               "evaluator_box_iou": round(float(overlap), 5),
                               "local_inference_wall_s": round(elapsed, 4)}
                        rows.append(row)
                        print(label, case,
                              "route", route["coverage_status"],
                              "travel", path["minimum_scene_gap_m"],
                              "show", screen["minimum_scene_gap_m"],
                              "item", choice["accepted_box"] is not None,
                              flush=True)
        finally:
            room.close()
    finally:
        v100.OUT = original_out
        v100.SEQUENCES.clear()
        v100.SEQUENCES.update(original_sequences)
        v100.DIRECTIONS.clear()
        v100.DIRECTIONS.update(original_directions)
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-native-pose-meeting.v1",
              "source_sha256": sha(SOURCE),
              "native_heading_report_sha256": sha(NATIVE_REPORT),
              "v135_report_sha256": sha(v135p.OUT / "report.json"),
              "source_staging_packet_path": str(source.relative_to(ROOT)),
              "source_staging_packet_sha256": sha(source),
              "selected_visual_sha256": sha(selected_visual()),
              "exemplar_sha256": sha(V149),
              "owner_hist_sha256": sha(V150),
              "scope": "Transfer the first12cm actual-minus-estimated errors "
              "from three saved 50Hz forward Direct Frame walks with joint "
              "range scan correction into a prescribed12cm backward "
              "owner-room move. Signed and laterally mirrored errors are "
              "development sensitivity cases, not a reverse gait model "
              "or statistical uncertainty bound. Actor route map includes "
              "V135p's optimistic virtual three-ring plus rear-mid "
              "directed probe. Full-body and sampled arm self gaps use20ms "
              "kinematic motion; owner shows through18.2s, and generic "
              "taught-item inference uses rendered17.4s RGB-D. The native "
              "policy does not run in this room, and no selected body exists.",
              "native_error_sources": native,
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
