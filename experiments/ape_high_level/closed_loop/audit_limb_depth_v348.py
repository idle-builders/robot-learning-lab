"""V348: actor RGB-D replay of the person/mannequin challenge frames.

The saved DINO proposals choose image regions. Re-rendered RGB must exactly
match the retained actor image before depth is used. Person and mannequin
labels, geometry, and reach type enter only after the actor measurements.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import challenge_person_belief_v346 as v346
from challenge_interaction_motion_v347 import MovingItemRoom
import direct_structural_rooms_v270 as structural
from articulated_person_scene import (NEAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_localization_v297 import depth_variant
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose
from robot_geometry import RobotGeometry


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v348"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V346 = ROOT / "artifacts/ape-closed-loop-cove50-v346/person-belief-challenge.json"
V347 = ROOT / "artifacts/ape-closed-loop-cove50-v347/interaction-motion-challenge.json"
DEPTH_CHANGE_M = .05


def region(box: list[float], height: int, width: int) -> tuple[slice, slice]:
    x0, y0, x1, y1 = box
    return (slice(max(0, math.floor(y0)), min(height, math.ceil(y1))),
            slice(max(0, math.floor(x0)), min(width, math.ceil(x1))))


def actor_arm_box(sample: dict) -> list[float] | None:
    arms = [d for d in sample["actor_lower_limb_detections"]
            if "arm" in d["label"] or "hand" in d["label"]]
    return max(arms, key=lambda d: d["score"])["box_xyxy"] if arms else None


def point_summary(depth: np.ndarray, box: list[float] | None,
                  camera_position: np.ndarray, camera_rotation: np.ndarray,
                  fovy_deg: float) -> dict | None:
    if box is None:
        return None
    height, width = depth.shape
    ys, xs = region(box, height, width)
    crop = depth[ys, xs]
    valid = np.isfinite(crop) & (crop > 0)
    if not valid.any():
        return {"valid_pixels": 0, "front_world_xyz_m": None,
                "depth_q20_m": None}
    values = crop[valid]
    front = valid & (crop <= np.quantile(values, .25))
    rows, cols = np.where(front)
    rows += ys.start
    cols += xs.start
    selected_depth = depth[rows, cols]
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rays = np.stack(((cols - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(selected_depth)), axis=1)
    points = camera_position + (rays * selected_depth[:, None]) @ camera_rotation.T
    return {"valid_pixels": int(valid.sum()),
            "depth_q20_m": round(float(np.quantile(values, .2)), 4),
            "front_world_xyz_m": np.round(np.median(points, axis=0), 4).tolist()}


def depth_change(a: np.ndarray, b: np.ndarray,
                 box: list[float] | None) -> dict | None:
    if box is None:
        return None
    ys, xs = region(box, *a.shape)
    prior, current = a[ys, xs], b[ys, xs]
    both = np.isfinite(prior) & np.isfinite(current)
    changed = both & (np.abs(current - prior) > DEPTH_CHANGE_M)
    return {"both_valid_pixels": int(both.sum()),
            "changed_pixels": int(changed.sum()),
            "changed_fraction_of_both_valid": round(
                float(changed.sum() / both.sum()), 4) if both.any() else None}


def changed_surface(a: np.ndarray, b: np.ndarray, box: list[float] | None,
                    camera_position: np.ndarray, camera_rotation: np.ndarray,
                    fovy_deg: float, robot: RobotGeometry) -> dict | None:
    """Project newly closer pixels, excluding uncovered distant background."""
    if box is None:
        return None
    height, width = a.shape
    ys, xs = region(box, height, width)
    prior, current = a[ys, xs], b[ys, xs]
    valid = (np.isfinite(prior) & np.isfinite(current) &
             (current < prior - DEPTH_CHANGE_M))
    rows, cols = np.where(valid)
    if not len(rows):
        return {"newly_closer_pixels": 0, "nonself_points": 0,
                "closest_body_gap_m": None, "body_gap_q10_m": None,
                "median_changed_xyz_m": None}
    rows = rows[::4] + ys.start
    cols = cols[::4] + xs.start
    distances = b[rows, cols]
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rays = np.stack(((cols - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(distances)), axis=1)
    points = camera_position + (rays * distances[:, None]) @ camera_rotation.T
    height_mask = (points[:, 2] >= .08) & (points[:, 2] <= .65)
    points = points[height_mask]
    if not len(points):
        return {"newly_closer_pixels": int(valid.sum()), "nonself_points": 0,
                "closest_body_gap_m": None, "body_gap_q10_m": None,
                "median_changed_xyz_m": None}
    gaps = RobotGeometry.solid_distances(points, robot.primitives())
    points, gaps = points[gaps >= 0.], gaps[gaps >= 0.]
    return {"newly_closer_pixels": int(valid.sum()),
            "nonself_points": len(points),
            "closest_body_gap_m": round(float(np.min(gaps)), 4)
            if len(gaps) else None,
            "body_gap_q10_m": round(float(np.quantile(gaps, .1)), 4)
            if len(gaps) else None,
            "median_changed_xyz_m": np.round(np.median(points, axis=0), 4).tolist()
            if len(points) else None}


def replay_case(row: dict) -> dict:
    seed = row["seed"]
    name = row["case"]
    moving_negative = "moving_item" in name
    mannequin = "static_mannequin" in name
    positive = row["positive_person"]
    variant = row["variant"]
    branch = OUT / name
    branch.mkdir(parents=True, exist_ok=True)
    trace = json.loads((V323 / f"seed{seed}-present" /
                        f"seed{seed}-present/episode-trace.json").read_text())
    original_model = structural.structural_model
    if positive:
        structural.structural_model = (
            lambda kind, side: add_person(
                original_model(kind, side),
                shirt_rgba=variant["shirt_rgba"]))
    elif mannequin:
        structural.structural_model = (
            lambda kind, side: add_person(original_model(kind, side)))
    elif variant["coatstand"]:
        structural.structural_model = (
            lambda kind, side: v346.add_coatstand(original_model(kind, side)))
    try:
        room = (MovingItemRoom if moving_negative else LateCorridorRoom)(
            case=tuple(trace["case"]), pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimated_pose(trace)
        room.head = variant["head_yaw_rad"]
        room.data.qpos[room.model.joint(
            "left_shoulder_pitch").qposadr[0]] = -.8
        room.model.light_diffuse[:] *= variant["light_scale"]
        offset = np.asarray(variant["offset_xyz_m"]) if positive else None
        if positive:
            room.model.body_pos[room.model.body("loop_person").id] = offset
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        robot = RobotGeometry(room.model)
        captures = []
        for index, sample in enumerate(row["actor_samples"]):
            room.time_s = sample["time_s"]
            room._update_moving_obstacle()
            if positive:
                fraction = (room.time_s - 16.2) / .4
                hand = START_HAND_XYZ_M + fraction * (
                    NEAR_HAND_XYZ_M - START_HAND_XYZ_M)
                set_person_pose(room.model, hand)
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] += offset
            room._forward()
            rgb, ideal = room.capture("lower_rgb", depth=True)
            with np.load(ROOT / sample["actor_packet_path"]) as saved:
                difference = np.max(np.abs(np.asarray(rgb).astype(np.int16) -
                                           saved["lower_rgb"].astype(np.int16)), axis=2)
                rgb_exact = bool(np.count_nonzero(difference) == 0)
                qpos_exact = bool(np.array_equal(room.data.qpos, saved["actor_qpos"]))
            if not (rgb_exact and qpos_exact):
                raise AssertionError(f"RGB/qpos replay changed for {name} {index}: "
                                     f"rgb={rgb_exact} qpos={qpos_exact} "
                                     f"pixels={np.count_nonzero(difference)} "
                                     f"max={np.max(difference)}")
            short = depth_variant(ideal, "short", seed, 500 + index)
            long = depth_variant(ideal, "noisy_long", seed, 500 + index)
            position, rotation = room.camera_pose_estimate("lower_rgb")
            packet = branch / f"rgbd-{index:02d}.npz"
            np.savez_compressed(packet, lower_short_depth_m=short,
                                lower_noisy_long_depth_m=long,
                                camera_origin_estimated_m=position,
                                camera_rotation_estimated=rotation,
                                capture_time_s=room.time_s)
            box = actor_arm_box(sample)
            captures.append({
                "time_s": sample["time_s"],
                "actor_depth_packet_path": str(packet.relative_to(ROOT)),
                "actor_depth_packet_sha256": sha(packet),
                "rgb_and_qpos_pixel_exact_replay": True,
                "actor_limb_box_xyxy": box,
                "short_roi": point_summary(short, box, position, rotation, fovy),
                "long_roi": point_summary(long, box, position, rotation, fovy),
                "short_depth": short, "long_depth": long})
        first_box = next((c["actor_limb_box_xyxy"] for c in captures
                          if c["actor_limb_box_xyxy"] is not None), None)
        changes = []
        for after_index in range(1, len(captures)):
            before, after = captures[after_index - 1], captures[after_index]
            with np.load(ROOT / after["actor_depth_packet_path"]) as data:
                position = data["camera_origin_estimated_m"].copy()
                rotation = data["camera_rotation_estimated"].copy()
            with np.load(ROOT / row["actor_samples"][after_index][
                    "actor_packet_path"]) as actor:
                robot.set_state(actor["actor_qpos"])
            changes.append({"from_time_s": before["time_s"],
                            "to_time_s": after["time_s"],
                            "short": depth_change(before["short_depth"],
                                                  after["short_depth"], first_box),
                            "long": depth_change(before["long_depth"],
                                                 after["long_depth"], first_box),
                            "long_newly_closer_surface": changed_surface(
                                before["long_depth"], after["long_depth"],
                                first_box, position, rotation, fovy, robot)})
        for capture in captures:
            capture.pop("short_depth")
            capture.pop("long_depth")
        return {"case": name, "seed": seed, "positive_person": positive,
                "actor_samples": captures,
                "first_limb_box_depth_changes": changes,
                "evaluator_minimum_person_gap_m": row.get(
                    "evaluator_minimum_person_gap_m")}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior346 = json.loads(V346.read_text())
    prior347 = json.loads(V347.read_text())
    rows = [replay_case(row) for row in (
        prior346["rows"] + prior347["moving_negative_rows"])]
    report = {
        "scope": "Pixel-exact lower-RGB re-render of 18 V346/V347 actor scenes, now pairing each DINO arm proposal with hypothetical 7–50 cm short or 8–250 cm noisy long depth. 5 cm depth-change measurements use only the first lower arm box and consecutive actor depth frames; semantic person labels and true clearance are evaluator-only. No motion gate or physical authorization is selected from this exploratory audit.",
        "source_sha256": sha(SOURCE),
        "v346_report_sha256": sha(V346),
        "v347_report_sha256": sha(V347),
        "depth_change_threshold_m": DEPTH_CHANGE_M,
        "rows": rows,
    }
    (OUT / "limb-depth-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("pixel-exact", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
