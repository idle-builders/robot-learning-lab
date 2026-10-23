"""V374: move the owner's hand independently during the distant show.

Reuses V370's selected-body exposure/camera/detector harness with a scoped
scene-motion adapter: the torso stays fixed while two-segment arm IK moves
the side-held can. Robust long-depth point continuity is scored afterward.
The adapter restores V370 globals before returning; no old run is rewritten.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import challenge_moving_distant_show_v370 as base
from articulated_person_scene import (ITEM_SIDE_OFFSET_M,
                                      START_HAND_XYZ_M, set_person_pose)
from audit_direct_structural_entry_v271 import V149, V150
from robust_item_depth import locate_box_center
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v374"
SEEDS = (170, 172)
PATTERNS = ("hand_gentle", "hand_faster")


def hand_offset(pattern: str, elapsed_s: float) -> np.ndarray:
    if pattern == "hand_gentle":
        period, amp_x, amp_y = .8, .02, .05
    elif pattern == "hand_faster":
        period, amp_x, amp_y = .4, .05, .10
    else:
        raise ValueError(pattern)
    wave = math.sin(2 * math.pi * elapsed_s / period)
    return np.asarray((amp_x * wave, amp_y * wave, 0.))


def hand_scene(room, pattern: str, elapsed_s: float) -> None:
    room.time_s = 16.2 + elapsed_s
    room._update_moving_obstacle()
    room.head = min(.8, 2. * max(0., elapsed_s))
    room.model.body_pos[room.model.body("loop_person").id] = (0., 0., 0.)
    set_person_pose(room.model, START_HAND_XYZ_M + hand_offset(
        pattern, elapsed_s), carried_item="loop_TomatoSauce")
    room._forward()


def augment(row: dict) -> dict:
    points = []
    errors = []
    for sample in row["samples"]:
        with np.load(ROOT / sample["actor_packet_path"]) as packet:
            pose = (packet["camera_origin_estimated_m"].copy(),
                    packet["camera_rotation_estimated"].copy())
            depth = packet["noisy_long_depth_m"].copy()
        estimate = locate_box_center(
            sample["actor_choice"]["accepted_box"], pose, depth,
            fovy_deg=67.)
        position = estimate["world_xyz_m"]
        points.append(position)
        if position is not None and sample[
                "evaluator_accepted_presented_can"]:
            true_item = (START_HAND_XYZ_M + ITEM_SIDE_OFFSET_M +
                         hand_offset(row["pattern"],
                                     sample["capture_time_s"] - 16.2))
            errors.append(float(np.linalg.norm(np.asarray(position) -
                                               true_item)))
        sample["actor_robust_long_depth"] = estimate
    links = sum(a is not None and b is not None and
                float(np.linalg.norm(np.asarray(a) - np.asarray(b))) <= .12
                for a, b in zip(points[:-1], points[1:]))
    row["actor_robust_long_position_frames"] = sum(p is not None
                                                    for p in points)
    row["actor_robust_3d_links"] = links
    row["evaluator_median_robust_position_error_m"] = (
        None if not errors else round(float(np.median(errors)), 5))
    row["evaluator_max_robust_position_error_m"] = (
        None if not errors else round(max(errors), 5))
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    original_scene = base.set_scene
    original_out = base.OUT
    try:
        base.set_scene = hand_scene
        base.OUT = OUT
        rows = [augment(base.run_case(seed, pattern, vision, exemplar,
                                      owner_hist))
                for seed in SEEDS for pattern in PATTERNS]
    finally:
        base.set_scene = original_scene
        base.OUT = original_out
    report = {
        "scope": "Selected Cove-50 guarded blocked-route no-contact item-show screen using V370's 0→+.8rad kinematic head slew, 10Hz RGB-D,20ms three-render exposure and released taught-object vision. V374's scoped scene adapter leaves the owner torso still while articulated two-segment arm and can move independently in gentle/faster lateral trajectories. Robust long-depth point and <=12cm adjacent association scored on the same actor packets. Person and item dynamics, optics and head skill remain procedural; exact identity still needs owner confirmation. V370 video overlay label is inherited from the reused render harness.",
        "source_sha256": sha(SOURCE),
        "render_harness_source_sha256": sha(Path(base.__file__)),
        "robust_depth_source_sha256": sha(Path(
            locate_box_center.__code__.co_filename)),
        "rows": rows}
    (OUT / "independent-hand-show.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], "correct", row["evaluator_correct_frames"],
              "robust", row["actor_robust_long_position_frames"],
              "links", row["actor_robust_3d_links"],
              "person", row["evaluator_min_person_gap_m"], flush=True)


if __name__ == "__main__":
    main()
