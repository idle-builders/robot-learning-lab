"""Measure camera-ray coverage of the ape's low front-left shell before show.

This reconstructs the same nine stationary pre-show instants as V167p, but
uses self-masked metric depth rays from the head and proposed chest cameras.
It asks whether an actor-known 5–10 cm low shell is observed as free, occupied
or unknown. The camera maps never receive prop or person labels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import screen_pre_show_range_admission_v167p as v167p
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-preshow-camera-shell"
CAMERAS = ("lower_rgb", "chest_low70_x45")
DEPTH_STRIDE = 8
SHELL_MARGINS_M = (.05, .10)


def front_left_low_shell(robot: RobotGeometry, actor_qpos: np.ndarray,
                         estimate: np.ndarray) -> np.ndarray:
    """Actor-geometry samples outside the known front-left low body solid."""
    robot.set_state(RobotGeometry.pose_qpos(actor_qpos, estimate))
    primitives = robot.primitives()
    c, s = math.cos(float(estimate[2])), math.sin(float(estimate[2]))
    unique = {}
    for point, normal in RobotGeometry.surface(primitives):
        dx, dy = point[:2]-estimate[:2]
        local_x, local_y = c*dx+s*dy, -s*dx+c*dy
        front_left_normal = ((c*normal[0]+s*normal[1]) +
                             (-s*normal[0]+c*normal[1]))
        if not (.015 < point[2] < .18 and local_x > -.03 and
                local_y > .02 and front_left_normal > .2):
            continue
        for margin in SHELL_MARGINS_M:
            sample = point+margin*normal
            if not (.015 < sample[2] < .18):
                continue
            if RobotGeometry.solid_distance(sample, primitives) <= .005:
                continue
            unique.setdefault(DepthVoxelMap.cell(sample), sample)
    return np.asarray(list(unique.values()), dtype=float).reshape(-1, 3)


def shell_state(voxel_map: DepthVoxelMap, samples: np.ndarray,
                time_s: float) -> dict:
    counts = {"free": 0, "occupied": 0, "unknown": 0}
    for point in samples:
        counts[voxel_map.state(point, time_s)] += 1
    return {**counts, "samples": len(samples),
            "all_observed_free": counts["free"] == len(samples),
            "last_observation_age_s": (None if voxel_map.last_time_s is None
                else round(time_s-voxel_map.last_time_s, 4))}


def merged(*maps: DepthVoxelMap) -> DepthVoxelMap:
    result = DepthVoxelMap()
    for source in maps:
        for name in ("free", "occupied"):
            target = getattr(result, name)
            for cell, stamp in getattr(source, name).items():
                target[cell] = max(target.get(cell, -math.inf), stamp)
        result.frames += source.frames
        if source.last_time_s is not None:
            result.last_time_s = max(result.last_time_s or -math.inf,
                                     source.last_time_s)
    return result


def scan_case(case: dict, prior: dict, staging: Path,
              range_rows: dict) -> dict:
    v160p.VariantRoom.variant = v160p.VARIANTS[case["variant"]]
    branch = OUT / case["id"]
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(staging, allow_pickle=False) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    room = v160p.VariantRoom(case=tuple(prior["case"]),
                             pose_source="command", branch=branch)
    masks = {}
    try:
        room.layout(case["seed"], absent=False, furnished=True)
        qpos[0] += case["station_x_m"]
        estimate[0] += case["station_x_m"]
        qpos[room.model.joint("left_shoulder_pitch").qposadr[0]] = .6
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        room.head = float(qpos[room.head_addr])
        robot = RobotGeometry(room.model)
        samples = front_left_low_shell(robot, qpos, estimate)
        if len(samples) == 0:
            raise AssertionError("Low front-left shell is empty")
        shell_path = branch / "actor-low-shell.npz"
        np.savez_compressed(shell_path,
                            shell_samples_xyz_m=samples,
                            actor_qpos=qpos,
                            estimated_pose_xyyaw=estimate)
        for camera in CAMERAS:
            masks[camera] = SelfDepthMask(
                room.model, camera=camera, width=W, height=H,
                agreement_m=.04)
        maps = {camera: DepthVoxelMap() for camera in CAMERAS}
        packet_rows = []
        for index, time_s in enumerate(v167p.SCAN_TIMES_S):
            room.time_s = time_s
            set_person_pose(room.model, START_HAND_XYZ_M,
                            carried_item="loop_TomatoSauce")
            room._forward()
            for camera in CAMERAS:
                rgb, depth = room.capture(camera, depth=True)
                if depth is None:
                    raise AssertionError("Expected metric camera depth")
                filtered, mask_stats = masks[camera].apply(
                    depth, room.data.qpos.copy(), max_depth_m=1.4)
                pose = room.camera_pose_estimate(camera)
                fovy = float(room.model.cam_fovy[
                    room.model.camera(camera).id])
                map_stats = maps[camera].observe(
                    filtered, pose, time_s, fovy_deg=fovy,
                    stride=DEPTH_STRIDE, max_depth_m=1.4,
                    self_masked=True)
                packet_path = (branch /
                               f"{camera}-self-masked-depth-{index:02d}.npz")
                np.savez_compressed(
                    packet_path,
                    filtered_depth_m=filtered.astype(np.float32),
                    camera_origin_xyz_m=pose[0],
                    camera_rotation=pose[1],
                    fovy_deg=fovy,
                    capture_time_s=time_s)
                image_path = None
                if index == len(v167p.SCAN_TIMES_S)-1:
                    image_path = branch / f"{camera}-last-pre-show.jpg"
                    rgb.save(image_path, quality=85)
                packet_rows.append({
                    "camera": camera,
                    "capture_time_s": time_s,
                    "packet_path": str(packet_path.relative_to(ROOT)),
                    "packet_sha256": sha(packet_path),
                    "last_image_path": (str(image_path.relative_to(ROOT))
                                        if image_path else None),
                    "last_image_sha256": (sha(image_path)
                                          if image_path else None),
                    "self_mask": mask_stats,
                    "map": map_stats,
                    "low_shell": shell_state(maps[camera], samples, time_s),
                })
        joint_map = merged(*maps.values())
        now = v167p.SCAN_TIMES_S[-1]
        final = {camera: shell_state(maps[camera], samples, now)
                 for camera in CAMERAS}
        final["both_cameras"] = shell_state(joint_map, samples, now)
        range_decisions = {}
        for occlusion in v167p.OCCLUSIONS:
            key = (case["variant"], case["seed"],
                   case["station_x_m"], "four_front_side", occlusion)
            range_decisions[occlusion] = range_rows[key][
                "pre_show_decision"]["provisional_admission"]
        return {
            "source_case_id": case["id"],
            "variant": case["variant"],
            "seed": case["seed"],
            "station_x_m": case["station_x_m"],
            "staging_packet": str(staging.relative_to(ROOT)),
            "staging_packet_sha256": sha(staging),
            "shell_packet": str(shell_path.relative_to(ROOT)),
            "shell_packet_sha256": sha(shell_path),
            "low_shell_samples": len(samples),
            "pre_show_camera_packets": packet_rows,
            "final_low_shell": final,
            "four_view_range_return_count_admission": range_decisions,
        }
    finally:
        for mask in masks.values():
            mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    cases_file = v167p.v165p.OUT / "report.json"
    cases = json.loads(cases_file.read_text())["source_cases"]
    hazards = [case for case in cases if case["class"] == "hazard"]
    range_file = v167p.OUT / "report.json"
    range_rows = json.loads(range_file.read_text())["rows"]
    by_range = {(row["variant"], row["seed"], row["station_x_m"],
                 row["layout"], row["occlusion"]): row
                for row in range_rows}
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for case in hazards:
            seed = case["seed"]
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            staging = (v100.V98 / "moving" / f"seed{seed}-present" /
                       f"seed{seed}-present/side-probe-00.npz")
            rows.append(scan_case(case, prior, staging, by_range))
            print(case["id"], rows[-1]["final_low_shell"], flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for camera in (*CAMERAS, "both_cameras"):
        group = [row["final_low_shell"][camera] for row in rows]
        summary[camera] = {
            "cases": len(group),
            "all_observed_free": sum(r["all_observed_free"] for r in group),
            "total_shell_cells": sum(r["samples"] for r in group),
            "free_shell_cells": sum(r["free"] for r in group),
            "occupied_shell_cells": sum(r["occupied"] for r in group),
            "unknown_shell_cells": sum(r["unknown"] for r in group),
        }
    report = {
        "schema": "robot-learning-lab.ape-preshow-camera-shell.v1",
        "source_sha256": sha(SOURCE),
        "depth_voxel_source_sha256": sha(SOURCE.with_name("depth_voxel.py")),
        "self_mask_source_sha256": sha(SOURCE.with_name("self_depth_mask.py")),
        "v98_report_sha256": sha(v98_file),
        "v167_report_sha256": sha(range_file),
        "v165_report_sha256": sha(cases_file),
        "current_design_status": pointer["status"],
        "camera_views": CAMERAS,
        "scan_times_s": v167p.SCAN_TIMES_S,
        "depth_stride": DEPTH_STRIDE,
        "shell_margins_m": SHELL_MARGINS_M,
        "scope": "Nine 10Hz pre-show RGB-D captures per camera in each of "
                 "eight reconstructed stationary held-owner scenes. A "
                 "joint-geometry-predicted self mask removes visible ape "
                 "pixels before 4cm ray mapping; 5/10cm actor-body surface "
                 "samples in the front-left low region are queried as "
                 "free/occupied/unknown. Neither camera uses evaluator "
                 "identity or future prop path. This is a conservative "
                 "coverage diagnostic at an imposed historical pose, not "
                 "a final admission threshold, whole-body route certificate, "
                 "selected mount or physical action result. V167 four-view "
                 "range return-count decisions are shown for contrast, not "
                 "equated to shell coverage.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
