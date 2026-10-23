"""Compare owner-facing head and downward chest RGB-D near-body coverage.

Both cameras render the same held articulated-owner scene at 10Hz. Metric
depth is projected through the actor's estimated camera pose (body estimate
plus joint state) into the same front-left low region used by V149p range.
The frozen 1.8cm/0.2s closing threshold is an exploratory cross-modality
screen, not a calibrated camera warning or a physical sensor selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_structural_entry_v271 import V149
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p
import run_live_articulated_owner_v343 as owner
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
from screen_standoff_front_cue_v149p import MIN_CLOSING_M_PER_0P2S
import stress_standoff_range_packets_v162p as v162p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-camera-depth-screen"
CAMERAS = ("lower_rgb", "chest_low70_x45")
TIMES_S = tuple(round(16.8+.1*i, 4) for i in range(16))
DEPTH_STRIDE = 4
DEPTH_OFFSET = 2
MAX_DEPTH_M = 1.4
MIN_SUPPORT = 30


def project_low_region(depth: np.ndarray,
                       pose: tuple[np.ndarray, np.ndarray],
                       estimated: np.ndarray, fovy_deg: float) -> tuple[int,
                                                                       float | None,
                                                                       int]:
    height, width = depth.shape
    focal = height/(2*math.tan(math.radians(fovy_deg)/2))
    rows, cols = np.mgrid[DEPTH_OFFSET:height:DEPTH_STRIDE,
                          DEPTH_OFFSET:width:DEPTH_STRIDE]
    sampled = depth[rows, cols]
    valid = np.isfinite(sampled) & (sampled > .08) & (sampled < MAX_DEPTH_M)
    rays = np.stack(((cols-width/2)/focal,
                     -(rows-height/2)/focal,
                     -np.ones_like(sampled)), axis=-1)[valid]
    origin, rotation = pose
    points = origin+(rays*sampled[valid, None])@rotation.T
    delta = points[:, :2]-estimated[:2]
    c, s = math.cos(float(estimated[2])), math.sin(float(estimated[2]))
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = points[:, 2]
    selected = x[(x > .08) & (x < .50) &
                 (y > .05) & (y < .50) &
                 (z > .015) & (z < .10)]
    return (len(selected),
            round(float(np.quantile(selected, .1)), 5)
            if len(selected) >= MIN_SUPPORT else None,
            int(np.count_nonzero(valid)))


def scan_case(case: dict, prior: dict, staging: Path) -> list[dict]:
    if case["class"] == "hazard":
        v160p.VariantRoom.variant = v160p.VARIANTS[case["variant"]]
        room_type = v160p.VariantRoom
    else:
        v148p.StandoffRoom.profile = case["variant"]
        room_type = v148p.StandoffRoom
    branch = OUT / case["id"]
    branch.mkdir(parents=True, exist_ok=True)
    with np.load(staging, allow_pickle=False) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    room = room_type(case=tuple(prior["case"]),
                     pose_source="command", branch=branch)
    try:
        room.layout(case["seed"], absent=False, furnished=True)
        qpos[0] += case["station_x_m"]
        estimate[0] += case["station_x_m"]
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        qpos[shoulder] = .6
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        start_head = float(qpos[room.head_addr])
        room.head = start_head
        by_camera = {name: [] for name in CAMERAS}
        for index, time_s in enumerate(TIMES_S):
            room.time_s = time_s
            shown = min(1., max(0., (time_s-16.6)/.4))
            hand = (START_HAND_XYZ_M+
                    shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M))
            set_person_pose(room.model, hand,
                            carried_item="loop_TomatoSauce")
            if time_s >= 17.-1e-6:
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
            look = min(1., max(0., (time_s-17.)/.4))
            room.head = start_head*(1.-look)
            room._forward()
            for camera in CAMERAS:
                folder = branch / camera
                folder.mkdir(parents=True, exist_ok=True)
                rgb, depth = room.capture(camera, depth=True)
                if depth is None:
                    raise AssertionError("Expected metric depth")
                camera_pose = room.camera_pose_estimate(camera)
                fovy = float(room.model.cam_fovy[
                    room.model.camera(camera).id])
                support, p10, valid_pixels = project_low_region(
                    depth, camera_pose, room.estimated.copy(), fovy)
                rgb_path = folder / f"rgb-{index:02d}.jpg"
                depth_path = folder / f"depth-{index:02d}.npz"
                rgb.save(rgb_path, quality=83)
                np.savez_compressed(
                    depth_path,
                    sampled_depth_m=depth[
                        DEPTH_OFFSET::DEPTH_STRIDE,
                        DEPTH_OFFSET::DEPTH_STRIDE].astype(np.float32),
                    full_shape_hw=np.asarray(depth.shape),
                    estimated_pose_xyyaw=room.estimated.copy(),
                    camera_origin_xyz_m=camera_pose[0],
                    camera_rotation=camera_pose[1],
                    fovy_deg=fovy,
                    capture_time_s=time_s)
                by_camera[camera].append({
                    "capture_time_s": time_s,
                    "rgb_path": str(rgb_path.relative_to(ROOT)),
                    "rgb_sha256": sha(rgb_path),
                    "depth_packet_path": str(depth_path.relative_to(ROOT)),
                    "depth_packet_sha256": sha(depth_path),
                    "valid_depth_samples": valid_pixels,
                    "front_left_low_depth_support": support,
                    "front_left_low_p10_x_m": p10,
                    "closing_0p2s_m": None,
                    "cue_now": False,
                })
        output = []
        for camera, samples in by_camera.items():
            for index in range(2, len(samples)):
                older = samples[index-2]["front_left_low_p10_x_m"]
                current = samples[index]["front_left_low_p10_x_m"]
                if older is None or current is None:
                    continue
                closing = round(older-current, 5)
                samples[index]["closing_0p2s_m"] = closing
                samples[index]["cue_now"] = (
                    closing >= MIN_CLOSING_M_PER_0P2S)
            first = next((s["capture_time_s"] for s in samples
                          if s["cue_now"]), None)
            breach = case["first_below_5cm_s"]
            camera_id = room.model.camera(camera).id
            output.append({
                "source_case_id": case["id"],
                "class": case["class"],
                "camera": camera,
                "camera_body": room.model.body(
                    int(room.model.cam_bodyid[camera_id])).name,
                "camera_local_xyz_m": room.model.cam_pos[camera_id].tolist(),
                "camera_fovy_deg": float(room.model.cam_fovy[camera_id]),
                "first_pose_compensated_depth_cue_s": first,
                "evaluator_first_below_5cm_s": breach,
                "timely_hazard_warning": (
                    case["class"] == "hazard" and
                    first is not None and breach is not None and
                    first <= breach),
                "false_control_warning": (
                    case["class"] == "control" and first is not None),
                "minimum_supported_17p2_to_18p2": min(
                    s["front_left_low_depth_support"] for s in samples
                    if 17.2-1e-6 <= s["capture_time_s"] <= 18.2+1e-6),
                "samples": samples,
            })
        return output
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    prior_file = v162p.OUT / "report.json"
    source = json.loads(prior_file.read_text())
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for case in source["source_cases"]:
            seed = case["seed"]
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            staging = (v100.V98 / "moving" / f"seed{seed}-present" /
                       f"seed{seed}-present/side-probe-00.npz")
            rows.extend(scan_case(case, prior, staging))
            print(case["id"],
                  [(r["camera"], r[
                      "first_pose_compensated_depth_cue_s"])
                   for r in rows[-2:]], flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for camera in CAMERAS:
        group = [r for r in rows if r["camera"] == camera]
        hazard = [r for r in group if r["class"] == "hazard"]
        control = [r for r in group if r["class"] == "control"]
        summary[camera] = {
            "hazards": len(hazard),
            "timely_hazard_warnings": sum(
                r["timely_hazard_warning"] for r in hazard),
            "controls": len(control),
            "false_control_warnings": sum(
                r["false_control_warning"] for r in control),
            "no_low_roi_support_during_17p2_to_18p2": sum(
                r["minimum_supported_17p2_to_18p2"] < MIN_SUPPORT
                for r in group),
        }
    report = {
        "schema": "robot-learning-lab.ape-standoff-camera-depth-screen.v1",
        "source_sha256": sha(SOURCE),
        "v162_report_sha256": sha(prior_file),
        "v98_report_sha256": sha(v98_file),
        "current_design_status": pointer["status"],
        "cameras": CAMERAS,
        "capture_times_s": TIMES_S,
        "depth_stride": DEPTH_STRIDE,
        "minimum_low_roi_depth_samples": MIN_SUPPORT,
        "frozen_closing_threshold_m_per_0p2s":
            MIN_CLOSING_M_PER_0P2S,
        "scope": "Rendered metric RGB-D from an owner-facing head camera "
                 "and fixed downward chest candidate at matched 10Hz "
                 "stationary-body scenes. Actor estimated pose and joint "
                 "configuration compensate head motion before a front-left "
                 "low-world-region depth statistic. This is a sensor role "
                 "screen, not a live response or object/person classifier. "
                 "The camera pose uses ideal joint state and saved estimated "
                 "base pose; depth optics/material failures and pose error "
                 "are absent. The 30-sample support and range-like closing "
                 "threshold are exploratory, not a selected camera rule. "
                 "Evaluator body-gap crossings score timing only.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
