"""V335 moving owner-hand geometry against neutral/protective ape postures.

This is an evaluator-only human-volume screen after V333's owner request.
It does not give true hand pose or collision distance to the actor.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from owner_hand_scene import (END_BODY_XYZ_M, HAND_LOCAL_XYZ_M,
                              OWNER_BODY, START_BODY_XYZ_M, add_owner_hand)
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import ParkedLateRoom
from run_shin_intrusion_posture_v324 import (actual_pose, estimated_pose,
                                              panel)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v335"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
REACH_END_X_M = {"near": .20, "far": END_BODY_XYZ_M[0]}
START_S = 16.2
DURATION_S = .4


def gap_to(room, robot_geoms: tuple[int, ...], names: tuple[str, ...]) -> tuple:
    return min((float(mujoco.mj_geomDistance(
        room.model, room.data, a, room.model.geom(name).id, 10., None)),
                room.model.geom(a).name, name)
               for a in robot_geoms for name in names)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_model = structural.structural_model
    structural.structural_model = (lambda kind, side:
                                   add_owner_hand(original_model(kind, side)))
    try:
        for seed in SEEDS:
            trace_path = (V323 / f"seed{seed}-present" /
                          f"seed{seed}-present/episode-trace.json")
            trace = json.loads(trace_path.read_text())
            actual, estimated = actual_pose(trace), estimated_pose(trace)
            for moving, room_type in ((True, LateCorridorRoom),
                                      (False, ParkedLateRoom)):
                for reach, end_x in REACH_END_X_M.items():
                    label = (f"seed{seed}-" + ("moving" if moving else "parked") +
                             f"-{reach}")
                    branch = OUT / label
                    branch.mkdir(parents=True, exist_ok=True)
                    room = room_type(case=tuple(trace["case"]),
                                     pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.actual[:] = actual
                        room.estimated[:] = estimated
                        room.head = 1.2
                        room.data.qpos[room.model.joint(
                            "left_shoulder_pitch").qposadr[0]] = -.8 if moving else 0.
                        robot = RobotGeometry(room.model)
                        robot_geoms = robot.geom_ids
                        human = ("loop_owner_forearm_geom", "loop_owner_hand_geom")
                        target = ("loop_TomatoSauce_geom",)
                        prop = ("loop_stool_geom",)
                        minimum = {name: (float("inf"), "", "")
                                   for name in ("human", "item", "prop", "all_scene")}
                        frames = []
                        for index, fraction in enumerate(np.linspace(0., 1., 21)):
                            room.time_s = START_S + DURATION_S * fraction
                            room._update_moving_obstacle()
                            start = np.asarray(START_BODY_XYZ_M)
                            end = np.asarray((end_x, END_BODY_XYZ_M[1],
                                              END_BODY_XYZ_M[2]))
                            hand_body = start + fraction * (end - start)
                            room.model.body_pos[room.model.body(OWNER_BODY).id] = hand_body
                            hand_center = hand_body + HAND_LOCAL_XYZ_M
                            room.model.body_pos[room.model.body(
                                "loop_TomatoSauce").id] = hand_center
                            room._forward()
                            gaps = {"human": gap_to(room, robot_geoms, human),
                                    "item": gap_to(room, robot_geoms, target),
                                    "prop": gap_to(room, robot_geoms, prop),
                                    "all_scene": room._current_obstacle_clearance()}
                            for name, entry in gaps.items():
                                minimum[name] = min(minimum[name], entry)
                            if seed == SEEDS[0] and moving and index % 2 == 0:
                                frames.append(panel(
                                    np.asarray(room.capture_external()),
                                    f"{label} | t{room.time_s:.2f}s | "
                                    f"human gap {gaps['human'][0]:.3f}m"))
                        video_path = None
                        if frames:
                            video_path = branch / "owner-reach.mp4"
                            with imageio.get_writer(video_path, fps=6,
                                                    codec="libx264", quality=7) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                        row = {"case": label, "seed": seed,
                               "moving_prop_evaluator_setup": moving,
                               "owner_reach": reach,
                               "owner_end_body_xyz_m_evaluator_setup": end.tolist(),
                               "ape_left_shoulder_pitch_rad": -.8 if moving else 0.,
                               "minimum_evaluator_clearances": {
                                   name: {"distance_m": round(entry[0], 5),
                                          "closest_geoms": [entry[1], entry[2]]}
                                   for name, entry in minimum.items()},
                               "video_path": (str(video_path.relative_to(ROOT))
                                              if video_path else None),
                               "video_sha256": sha(video_path) if video_path else None,
                               "v323_trace_sha256": sha(trace_path)}
                        rows.append(row)
                        print(label, "human", round(minimum["human"][0], 5),
                              "all", round(minimum["all_scene"][0], 5), flush=True)
                    finally:
                        room.close()
    finally:
        structural.structural_model = original_model
    report = {"scope": "V335 evaluator-only selected Direct Frame geometry screen with a visible scripted owner forearm capsule, hand sphere and carried can moving for0.4s during a show-me gesture. Compare a near reach to a farther offer, each with the moving-prop protective shoulder posture or parked-prop neutral posture. Actor sensors and a human-aware action gate are not implemented here; scene exact distances label the risk. Human arm biomechanics, timing variability, intent and physical compliance are unmodeled.",
              "source_sha256": sha(SOURCE),
              "owner_hand_scene_source_sha256": sha(Path(add_owner_hand.__code__.co_filename)),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "owner-reach-posture-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
