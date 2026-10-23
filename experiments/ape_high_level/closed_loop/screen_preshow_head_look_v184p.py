"""Test whether a feasible head-yaw look observes the low front-left shell.

The held body's head is set to candidate joint angles at one pre-show instant;
this is a viewpoint screen, not a timed or physically executed head skill.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from run import H, ROOT, W, sha
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import screen_preshow_camera_shell_v181p as v181p
import bound_preshow_camera_misses_v183p as v183p
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-preshow-head-look"
HEAD_YAWS_RAD = (-1.5, -.75, 0., .75, 1.2, 1.5)
TIME_S = 16.2


def map_saved_chest(row: dict) -> tuple[DepthVoxelMap, DepthVoxelMap]:
    saved = next(p for p in row["pre_show_camera_packets"]
                 if p["camera"] == "chest_low70_x45" and
                 p["capture_time_s"] == TIME_S)
    path = ROOT / saved["packet_path"]
    if sha(path) != saved["packet_sha256"]:
        raise AssertionError("Saved chest depth changed")
    with np.load(path, allow_pickle=False) as packet:
        depth = packet["filtered_depth_m"].copy()
        origin = packet["camera_origin_xyz_m"].copy()
        rotation = packet["camera_rotation"].copy()
        fovy = float(packet["fovy_deg"])
    returned = DepthVoxelMap()
    returned.observe(depth, (origin, rotation), TIME_S, fovy_deg=fovy,
                     stride=v181p.DEPTH_STRIDE, self_masked=True)
    ideal = DepthVoxelMap()
    ideal.free = returned.free.copy()
    ideal.occupied = returned.occupied.copy()
    ideal.frames = returned.frames
    ideal.last_time_s = returned.last_time_s
    v183p.add_simulator_miss_rays(ideal, depth, origin, rotation, fovy,
                                  TIME_S)
    return returned, ideal


def score(voxel_map: DepthVoxelMap,
          shell: np.ndarray) -> dict:
    counts = v183p.states(voxel_map, shell, TIME_S)
    return {**counts, "samples": len(shell)}


def scan_case(source: dict, prior: dict) -> dict:
    case_id = source["source_case_id"]
    branch = OUT / case_id
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / source["shell_packet"],
                 allow_pickle=False) as packet:
        shell = packet["shell_samples_xyz_m"].copy()
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    chest_returned, chest_ideal = map_saved_chest(source)
    v160p.VariantRoom.variant = v160p.VARIANTS[source["variant"]]
    room = v160p.VariantRoom(case=tuple(prior["case"]),
                             pose_source="command", branch=branch)
    mask = None
    try:
        room.layout(source["seed"], absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        room.time_s = TIME_S
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        joint = room.model.joint("head_yaw")
        if any(angle < joint.range[0] or angle > joint.range[1]
               for angle in HEAD_YAWS_RAD):
            raise AssertionError("Candidate head yaw exceeds joint limit")
        mask = SelfDepthMask(room.model, camera="lower_rgb",
                             width=W, height=H, agreement_m=.04)
        rows = []
        for index, yaw in enumerate(HEAD_YAWS_RAD):
            room.head = yaw
            room._forward()
            rgb, depth = room.capture("lower_rgb", depth=True)
            if depth is None:
                raise AssertionError("Expected head metric depth")
            filtered, mask_stats = mask.apply(
                depth, room.data.qpos.copy(), max_depth_m=1.4)
            origin, rotation = room.camera_pose_estimate("lower_rgb")
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            packet_path = branch / f"head-yaw-{index:02d}.npz"
            image_path = branch / f"head-yaw-{index:02d}.jpg"
            np.savez_compressed(
                packet_path,
                filtered_depth_m=filtered.astype(np.float32),
                camera_origin_xyz_m=origin,
                camera_rotation=rotation,
                fovy_deg=fovy,
                capture_time_s=TIME_S,
                head_yaw_rad=yaw)
            rgb.save(image_path, quality=85)
            head_returned = DepthVoxelMap()
            head_returned.observe(
                filtered, (origin, rotation), TIME_S,
                fovy_deg=fovy, stride=v181p.DEPTH_STRIDE,
                self_masked=True)
            head_ideal = DepthVoxelMap()
            head_ideal.free = head_returned.free.copy()
            head_ideal.occupied = head_returned.occupied.copy()
            head_ideal.frames = head_returned.frames
            head_ideal.last_time_s = head_returned.last_time_s
            miss = v183p.add_simulator_miss_rays(
                head_ideal, filtered, origin, rotation, fovy, TIME_S)
            rows.append({
                "head_yaw_rad": yaw,
                "packet_path": str(packet_path.relative_to(ROOT)),
                "packet_sha256": sha(packet_path),
                "rgb_path": str(image_path.relative_to(ROOT)),
                "rgb_sha256": sha(image_path),
                "self_mask": mask_stats,
                "simulator_far_rays": miss["simulator_far_rays"],
                "head_return_only": score(head_returned, shell),
                "head_optimistic_miss_as_free": score(head_ideal, shell),
                "both_return_only": score(v181p.merged(
                    head_returned, chest_returned), shell),
                "both_optimistic_miss_as_free": score(v181p.merged(
                    head_ideal, chest_ideal), shell),
            })
        return {
            "source_case_id": case_id,
            "variant": source["variant"],
            "seed": source["seed"],
            "station_x_m": source["station_x_m"],
            "source_shell_packet": source["shell_packet"],
            "source_shell_packet_sha256": source[
                "shell_packet_sha256"],
            "baseline_head_yaw_rad": float(qpos[room.head_addr]),
            "head_joint_limit_rad": joint.range.tolist(),
            "shell_cells": len(shell),
            "rows": rows,
        }
    finally:
        if mask is not None:
            mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v181p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for item in source["rows"]:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == item["seed"])
            rows.append(scan_case(item, prior))
            print(item["source_case_id"], [(r["head_yaw_rad"],
                r["both_optimistic_miss_as_free"]["free"])
                for r in rows[-1]["rows"]], flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for angle in HEAD_YAWS_RAD:
        group = [next(r for r in case["rows"] if r["head_yaw_rad"] == angle)
                 for case in rows]
        summary[str(angle)] = {
            "cases": len(group),
            "both_return_only_free_cells": sum(r[
                "both_return_only"]["free"] for r in group),
            "both_optimistic_free_cells": sum(r[
                "both_optimistic_miss_as_free"]["free"] for r in group),
            "both_optimistic_unknown_cells": sum(r[
                "both_optimistic_miss_as_free"]["unknown"] for r in group),
        }
    report = {
        "schema": "robot-learning-lab.ape-preshow-head-look.v1",
        "source_sha256": sha(SOURCE),
        "v181_report_sha256": sha(source_file),
        "v183_source_sha256": sha(Path(v183p.__file__)),
        "head_yaw_candidates_rad": HEAD_YAWS_RAD,
        "time_s": TIME_S,
        "scope": "Six head-yaw joint-limit-respecting viewpoints at the same "
                 "imposed 16.2s historical held pose in eight matched "
                 "scenes. The chest view is fixed. Return-only camera rays "
                 "and an optimistic simulator-far-miss-as-free bound query "
                 "the actor-known 39-cell low front-left shell. Head motion "
                 "is teleported, not timed or executed by a selected body. "
                 "No real camera no-return, selected sensor or invitation "
                 "permission follows.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
