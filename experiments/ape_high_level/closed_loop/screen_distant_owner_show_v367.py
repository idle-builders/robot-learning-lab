"""V367: can the selected ape verify a taught item without a close handoff?

At the live guarded blocked-route pose, an articulated owner holds the can
beside the starting hand and does not reach. The actor tries its existing
fresh RGB-D taught-object pipeline under bounded lower-head looks plus the
upper social camera. True item/person geometry only scores afterward.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
import run_blocked_owner_help_v317 as owner_module
from run_color_grounded_entry_v151 import infer
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v367"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
HEAD_YAWS_RAD = (0., .4, .8, 1.2)
CAMERAS = ("lower_rgb", "upper_rgb")


def run_case(seed: int, vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray) -> dict:
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V353 / "compliant_far" / f"seed{seed}-present" /
             f"seed{seed}-present")
    trace_path = prior / "episode-trace.json"
    invite_path = prior / "handoff-invitation.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(invite_path) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimate
        room.data.qpos[:] = qpos
        room.time_s = 16.2
        room._update_moving_obstacle()
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        person_geoms = RobotGeometry(room.model).geom_ids
        video_path = branch / "distant-owner-show.mp4"
        views = []
        with imageio.get_writer(video_path, fps=2, codec="libx264",
                                quality=7) as writer:
            for yaw in HEAD_YAWS_RAD:
                room.head = yaw  # Active look surrogate; no native head timing.
                room._forward()
                for camera in CAMERAS:
                    rgb, depth = room.capture(camera, depth=True)
                    depth[(depth < .08) | (depth > 1.5)] = np.nan
                    pose = room.camera_pose_estimate(camera)
                    choice = infer(vision, exemplar, owner_hist, rgb, depth,
                                   pose, None, None)
                    truth = room.truth_box("TomatoSauce", camera)
                    correct = (choice["accepted_box"] is not None and
                               truth is not None and iou(
                                   choice["accepted_box"], truth) >= .25)
                    label = f"yaw{yaw:+.1f}-{camera}"
                    packet_path = branch / f"actor-{label}.npz"
                    np.savez_compressed(
                        packet_path, rgb=np.asarray(rgb), depth_m=depth,
                        camera_origin_estimated_m=pose[0],
                        camera_rotation_estimated=pose[1],
                        actor_qpos=room.pose_data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
                    gap = gap_to_person(room, person_geoms)[0]
                    views.append({
                        "head_yaw_rad": yaw, "camera": camera,
                        "actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "actor_packet_sha256": sha(packet_path),
                        "actor_choice": choice,
                        "evaluator_truth_box": truth,
                        "evaluator_accepted_presented_can": correct,
                        "evaluator_person_gap_m": round(gap, 5)})
                    writer.append_data(owner_module.panel(
                        room, rgb, f"V367 seed{seed} no-contact show "
                        f"{camera} head{yaw:+.1f} "
                        f"{'found' if correct else 'unknown'}"))
        return {"seed": seed, "source_invitation_sha256": sha(invite_path),
                "source_episode_trace_sha256": sha(trace_path),
                "views": views,
                "video_path": str(video_path.relative_to(ROOT)),
                "video_sha256": sha(video_path)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = [run_case(seed, vision, exemplar, owner_hist) for seed in SEEDS]
    report = {
        "scope": "Selected Cove-50 Direct Frame V353 guarded blocked-route invitation state, articulated owner holding the taught can 8cm beside the original non-reaching hand, no base or person motion. Four direct-qpos lower-head yaw samples0/.4/.8/1.2rad plus upper camera, released DINO/hue candidate and RGB-D world point. True can box and robot/person gap score after actor choice. This is a no-contact visual recovery screen; speech/listening, owner comprehension, native head slew and stopping absent. No final sensor selection.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows}
    (OUT / "distant-owner-show.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print("seed", row["seed"], [
            (v["head_yaw_rad"], v["camera"])
            for v in row["views"] if v["evaluator_accepted_presented_can"]],
            "min person", min(v["evaluator_person_gap_m"]
                              for v in row["views"]), flush=True)


if __name__ == "__main__":
    main()
