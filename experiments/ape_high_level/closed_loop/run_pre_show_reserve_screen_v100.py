"""V100: screen pre-show meeting moves across travel, owner view and hold.

This is evaluator feasibility, not an actor-selected plan. A cooperative owner
waits for the requested reposition before moving the item into view. The low
prop keeps its V98 absolute trajectory, so extra motion consumes real reserve.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_blocked_owner_help_v317 as owner_panel
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v100"
V98 = ROOT / "artifacts/ape-closed-loop-cove50-v98"
START_S = 16.2
STEP_DURATION_S = .4
COMMAND_STEP_M = .08
ACTUAL_FACTOR = .55
OWNER_MOVE_S = .4
LOOK_S = .4
INFERENCE_S = .8
DIRECTIONS = {
    "right": (0., -1.),
    "back": (-1., 0.),
    "back_right": (-math.sqrt(.5), -math.sqrt(.5)),
}
SEQUENCES = {
    "hold": (),
    "right1": ("right",),
    "back2_diag2": ("back", "back", "back_right", "back_right"),
    "diag3_back1": ("back_right", "back_right", "back_right", "back"),
    "back3_right2": ("back", "back", "back", "right", "right"),
    "back2_diag3": ("back", "back", "back_right", "back_right", "back_right"),
}


def prop_y(time_s: float) -> float:
    return .306 + min(1., max(0., (time_s - 17.2) / .8)) * (.210 - .306)


def commanded_offset(sequence: tuple[str, ...], elapsed_s: float) -> np.ndarray:
    offset = np.zeros(2)
    for step, name in enumerate(sequence):
        fraction = min(1., max(0., (elapsed_s - step*STEP_DURATION_S) /
                                STEP_DURATION_S))
        offset += COMMAND_STEP_M*fraction*np.asarray(DIRECTIONS[name])
    return offset


def run_one(seed: int, case: str, sequence_name: str,
            qpos: np.ndarray, estimate: np.ndarray,
            source_case: tuple) -> dict:
    room = OfferDriftRoom(case=source_case, pose_source="command",
                          branch=OUT / "scene")
    try:
        room.layout(seed, absent=False, furnished=True)
        obstacle = CASES[case]
        if obstacle is not None:
            position, size = obstacle
            room.model.body_pos[room.model.body("loop_box").id] = position
            room.model.geom("loop_box_geom").size[:] = size
        room.data.qpos[:] = qpos
        actual = np.array((qpos[0], qpos[1],
                           2*math.atan2(qpos[6], qpos[3])))
        start_head = float(qpos[room.head_addr])
        room.head = start_head
        sequence = SEQUENCES[sequence_name]
        owner_start = START_S + STEP_DURATION_S*len(sequence)
        owner_end = owner_start + OWNER_MOVE_S
        image_time = owner_end + LOOK_S
        recognition_end = image_time + INFERENCE_S
        end_time = max(recognition_end, 18.0)
        steps = int(round((end_time-START_S)/.02))
        branch = OUT / f"seed{seed}-{case}-{sequence_name}"
        branch.mkdir(parents=True, exist_ok=True)
        video_path = branch / "pre-show-reserve.mp4"
        record_video = (seed == 170 and case in ("clear", "hand_close")
                        and sequence_name in ("hold", "back2_diag2"))
        writer_context = (imageio.get_writer(
            video_path, fps=5, codec="libx264", quality=7)
            if record_video else nullcontext(None))
        minimum = (float("inf"), None, None, None)
        first_below = None
        sampled = []
        with writer_context as writer:
            for tick in range(steps+1):
                time_s = START_S + .02*tick
                offset = commanded_offset(sequence, time_s-START_S)
                room.actual[:] = actual
                room.actual[:2] += ACTUAL_FACTOR*offset
                room.estimated[:] = estimate
                room.estimated[:2] += offset
                room.time_s = time_s
                look_fraction = min(1., max(0.,
                    (time_s-owner_end)/LOOK_S))
                room.head = start_head*(1.-look_fraction)
                fraction = min(1., max(0., (time_s-owner_start)/OWNER_MOVE_S))
                hand = (START_HAND_XYZ_M + fraction *
                        (FAR_HAND_XYZ_M - START_HAND_XYZ_M))
                set_person_pose(room.model, hand,
                                carried_item="loop_TomatoSauce")
                if time_s >= owner_end - 1e-6:
                    room.model.body_pos[room.model.body(
                        "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                y = prop_y(time_s)
                room.obstacle_xy[3] = (0., y)
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    0., y, STOOL_SIZE[2])
                room._forward()
                gap, robot_geom, obstacle_geom = room._current_obstacle_clearance()
                if gap < minimum[0]:
                    minimum = (float(gap), time_s, robot_geom, obstacle_geom)
                if first_below is None and gap < .05:
                    first_below = time_s
                if tick % 5 == 0:
                    sampled.append({"time_s": round(time_s, 4),
                                    "actual_pose_xyyaw": room.actual.round(5).tolist(),
                                    "evaluator_gap_m": round(float(gap), 5),
                                    "phase": ("reposition" if time_s < owner_start
                                              else "owner_show" if time_s < owner_end
                                              else "look" if time_s < image_time
                                              else "inference_hold")})
                if writer is not None and tick % 10 == 0:
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    writer.append_data(owner_panel.panel(
                        room, rgb, f"V100 {sequence_name} {case} | "
                        f"t{time_s:.1f} | gap {100*gap:.1f}cm"))
                if abs(time_s-image_time) < 1e-6:
                    rgb, depth = room.capture("lower_rgb", depth=True)
                    camera_origin, camera_rotation = room.camera_pose_estimate(
                        "lower_rgb")
                    frame_path = branch / "owner-image.npz"
                    np.savez_compressed(
                        frame_path, rgb=np.asarray(rgb), depth_m=depth,
                        camera_origin_m=camera_origin,
                        camera_rotation=camera_rotation,
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=time_s)
                    if record_video:
                        rgb.save(branch / "owner-image.png")
                    box = room.truth_box("TomatoSauce", "lower_rgb")
                    valid_depth = 0
                    if box is not None:
                        x0, y0, x1, y1 = map(int, box)
                        patch = depth[max(0,y0):max(0,y1),
                                      max(0,x0):max(0,x1)]
                        valid_depth = int(np.count_nonzero(
                            np.isfinite(patch) & (patch >= .08) & (patch <= 1.5)))
                    image = {"capture_time_s": round(time_s, 4),
                             "evaluator_truth_box": box,
                             "evaluator_valid_item_depth_pixels": valid_depth,
                             "camera": "lower_rgb",
                             "actor_frame_path": str(frame_path.relative_to(ROOT)),
                             "actor_frame_sha256": sha(frame_path)}
        if "image" not in locals():
            raise AssertionError("missing owner image")
        return {"seed": seed, "case": case,
                "sequence": sequence_name,
                "commands": sequence,
                "owner_start_s": round(owner_start, 4),
                "image_time_s": round(image_time, 4),
                "recognition_end_s": round(recognition_end, 4),
                "end_time_s": round(end_time, 4),
                "final_actual_pose_xyyaw": room.actual.round(5).tolist(),
                "minimum_scene_gap_m": round(minimum[0], 5),
                "minimum_time_s": round(minimum[1], 4),
                "closest_pair": list(minimum[2:]),
                "first_below_5cm_s": (round(first_below, 4)
                                      if first_below is not None else None),
                "owner_image": image,
                "video": ({"path": str(video_path.relative_to(ROOT)),
                           "sha256": sha(video_path)}
                          if record_video else None),
                "sampled_path": sampled}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
                if abs(float(packet["capture_time_s"]) - START_S) > 1e-5:
                    raise AssertionError("pre-show source time changed")
            for case in ("clear", "low_close", "hand_close",
                         "hand_clear_offset"):
                for sequence_name in SEQUENCES:
                    row = run_one(seed, case, sequence_name, qpos, estimate,
                                  tuple(prior["case"]))
                    row["source_packet_path"] = str(source.relative_to(ROOT))
                    row["source_packet_sha256"] = sha(source)
                    rows.append(row)
    finally:
        structural.structural_model = old_model
    report = {"schema": "robot-learning-lab.ape-pre-show-reserve-screen.v1",
              "source_sha256": sha(SOURCE),
              "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "sequences": SEQUENCES,
              "direction_xy": DIRECTIONS,
              "command_step_m": COMMAND_STEP_M,
              "actual_motion_factor": ACTUAL_FACTOR,
              "scope": "Evaluator-only pre-show meeting-pose screen from V98's saved 16.2s articulated-owner state. A cooperative scripted owner waits while the ape performs fixed bounded kinematic steps, then offers the item. The low prop follows its unchanged absolute 17.2-18.0s approach. Samples full-body scene gap through travel, presentation, look, an 0.8s held inference and at least the complete prop approach; checks rendered lower RGB-D item visibility with evaluator box. No actor-sensor corridor authorization, image-identity inference or physical skill is claimed.",
              "rows": rows}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for case in ("clear", "hand_close"):
        print(case, [(r["sequence"], r["minimum_scene_gap_m"],
                       r["owner_image"]["evaluator_valid_item_depth_pixels"])
                     for r in rows if r["seed"] == 170 and r["case"] == case])
    print(path)


if __name__ == "__main__":
    main()
