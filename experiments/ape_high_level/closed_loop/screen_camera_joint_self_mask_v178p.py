"""Replay saved RGB-D warning packets with an actor joint-geometry self mask.

The mask uses only estimated base pose, the staged joint configuration and
known robot collision solids. Evaluator segmentation is not an actor input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
import direct_structural_rooms_v270 as structural
from robot_geometry import RobotGeometry
from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import screen_standoff_camera_depth_v171p as v171p
import stress_standoff_range_packets_v162p as v162p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-camera-joint-self-mask"
MASK_MARGIN_M = .01


def projected_low_points(packet) -> tuple[np.ndarray, np.ndarray]:
    depth = packet["sampled_depth_m"].astype(float)
    height, width = map(int, packet["full_shape_hw"])
    rows, cols = np.mgrid[v171p.DEPTH_OFFSET:height:v171p.DEPTH_STRIDE,
                          v171p.DEPTH_OFFSET:width:v171p.DEPTH_STRIDE]
    focal = height/(2*math.tan(math.radians(float(packet["fovy_deg"]))/2))
    valid = np.isfinite(depth) & (depth > .08) & (depth < v171p.MAX_DEPTH_M)
    rays = np.stack(((cols-width/2)/focal,
                     -(rows-height/2)/focal,
                     -np.ones_like(depth)), axis=-1)[valid]
    xyz = packet["camera_origin_xyz_m"] + (
        rays*depth[valid, None]) @ packet["camera_rotation"].T
    estimate = packet["estimated_pose_xyyaw"]
    delta = xyz[:, :2]-estimate[:2]
    c, s = math.cos(float(estimate[2])), math.sin(float(estimate[2]))
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = xyz[:, 2]
    low = ((x > .08) & (x < .50) &
           (y > .05) & (y < .50) &
           (z > .015) & (z < .10))
    return xyz[low], x[low]


def scan_case(case: dict, source_rows: dict, prior: dict,
              staging: Path) -> list[dict]:
    if case["class"] == "hazard":
        v160p.VariantRoom.variant = v160p.VARIANTS[case["variant"]]
        room_type = v160p.VariantRoom
    else:
        v148p.StandoffRoom.profile = case["variant"]
        room_type = v148p.StandoffRoom
    with np.load(staging, allow_pickle=False) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    room = room_type(case=tuple(prior["case"]),
                     pose_source="command", branch=OUT)
    try:
        room.layout(case["seed"], absent=False, furnished=True)
        qpos[0] += case["station_x_m"]
        estimate[0] += case["station_x_m"]
        qpos[room.model.joint("left_shoulder_pitch").qposadr[0]] = .6
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        start_head = float(qpos[room.head_addr])
        robot = RobotGeometry(room.model)
        output = []
        for camera in v171p.CAMERAS:
            source = source_rows[camera]
            samples = []
            for index, saved in enumerate(source["samples"]):
                time_s = saved["capture_time_s"]
                look = min(1., max(0., (time_s-17.)/.4))
                qpos[room.head_addr] = start_head*(1.-look)
                actor_qpos = RobotGeometry.pose_qpos(qpos, estimate)
                robot.set_state(actor_qpos)
                packet_path = ROOT / saved["depth_packet_path"]
                if sha(packet_path) != saved["depth_packet_sha256"]:
                    raise AssertionError("Saved actor depth packet changed")
                with np.load(packet_path, allow_pickle=False) as packet:
                    xyz, x = projected_low_points(packet)
                    if float(packet["capture_time_s"]) != time_s:
                        raise AssertionError("Packet time changed")
                old_p10 = (round(float(np.quantile(x, .1)), 5)
                           if len(x) >= v171p.MIN_SUPPORT else None)
                if (len(x) != saved["front_left_low_depth_support"] or
                    old_p10 != saved["front_left_low_p10_x_m"]):
                    raise AssertionError("Saved actor low-depth statistic changed")
                distances = RobotGeometry.solid_distances(
                    xyz, robot.primitives())
                keep = distances > MASK_MARGIN_M
                masked_x = x[keep]
                p10 = (round(float(np.quantile(masked_x, .1)), 5)
                       if len(masked_x) >= v171p.MIN_SUPPORT else None)
                samples.append({
                    "time_s": time_s,
                    "source_depth_packet_path": saved["depth_packet_path"],
                    "source_depth_packet_sha256": saved[
                        "depth_packet_sha256"],
                    "low_roi_support": len(x),
                    "joint_self_masked_low_pixels": int(np.count_nonzero(~keep)),
                    "masked_low_roi_support": len(masked_x),
                    "masked_p10_x_m": p10,
                    "closing_0p2s_m": None,
                    "cue_now": False,
                })
            for index in range(2, len(samples)):
                current = samples[index]["masked_p10_x_m"]
                older = samples[index-2]["masked_p10_x_m"]
                if current is not None and older is not None:
                    delta = round(older-current, 5)
                    samples[index]["closing_0p2s_m"] = delta
                    samples[index]["cue_now"] = (
                        delta >= v171p.MIN_CLOSING_M_PER_0P2S)
            first = next((s["time_s"] for s in samples if s["cue_now"]), None)
            output.append({
                "source_case_id": case["id"],
                "class": case["class"],
                "camera": camera,
                "unmasked_first_cue_s": source[
                    "first_pose_compensated_depth_cue_s"],
                "joint_self_masked_first_cue_s": first,
                "evaluator_first_below_5cm_s": case["first_below_5cm_s"],
                "timely_hazard_warning": (case["class"] == "hazard" and
                    first is not None and case["first_below_5cm_s"] is not None
                    and first <= case["first_below_5cm_s"]),
                "false_control_warning": (
                    case["class"] == "control" and first is not None),
                "samples": samples,
            })
        return output
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v171p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    by_case = {}
    for row in source["rows"]:
        by_case.setdefault(row["source_case_id"], {})[row["camera"]] = row
    case_file = v162p.OUT / "report.json"
    cases = json.loads(case_file.read_text())["source_cases"]
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for case in cases:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == case["seed"])
            seed = case["seed"]
            staging = (v100.V98 / "moving" / f"seed{seed}-present" /
                       f"seed{seed}-present/side-probe-00.npz")
            rows.extend(scan_case(case, by_case[case["id"]],
                                  prior, staging))
            print(case["id"], flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for camera in v171p.CAMERAS:
        group = [r for r in rows if r["camera"] == camera]
        hazards = [r for r in group if r["class"] == "hazard"]
        controls = [r for r in group if r["class"] == "control"]
        summary[camera] = {
            "hazards": len(hazards),
            "timely_hazard_warnings": sum(
                r["timely_hazard_warning"] for r in hazards),
            "controls": len(controls),
            "false_control_warnings": sum(
                r["false_control_warning"] for r in controls),
            "masked_low_roi_pixels": sum(
                s["joint_self_masked_low_pixels"] for r in group
                for s in r["samples"]),
            "first_cues_changed": sum(
                r["unmasked_first_cue_s"] !=
                r["joint_self_masked_first_cue_s"] for r in group),
        }
    result = {
        "schema": "robot-learning-lab.ape-camera-joint-self-mask-screen.v1",
        "source_sha256": sha(SOURCE),
        "v171_report_sha256": sha(source_file),
        "v162_report_sha256": sha(case_file),
        "mask_margin_m": MASK_MARGIN_M,
        "scope": "Actor-available estimated base pose, staged joint state and known "
                 "collision solids mask points within 1 cm in all 512 saved "
                 "RGB-D packets. This is an offline actor-input replay on the "
                 "historical body, not an optics/pose calibration or physical "
                 "safety claim. Evaluator body-gap labels only score timing.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
