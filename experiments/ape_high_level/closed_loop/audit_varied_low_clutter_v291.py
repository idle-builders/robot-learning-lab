"""V291: frozen varied Direct Frame rooms for near-floor range margins."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_c1_plane_mount_v237 import MOUNTS, scan as c1_scan
from audit_lateral_clutter_sensors_v283 import virtual_cloud
from audit_shin_range_v190 import scan as shin_scan
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v291"
# Predeclared before evaluating any sensor return. Tuples are:
# seed, structure, crate center x/y, full box x/y/z, body yaw.
CASES = (
    (139, "open", -.05, -.32, .10, .08, .02, 0.),
    (140, "open", .06, .35, .10, .08, .04, -.35),
    (141, "open", 0., -.39, .16, .06, .08, .35),
    (142, "screen", .08, -.36, .08, .10, .04, -.35),
    (143, "screen", -.06, .33, .10, .08, .02, .35),
    (144, "screen", .03, .42, .12, .08, .08, .6),
    (145, "doorway", -.07, -.37, .10, .10, .04, -.6),
    (146, "doorway", .07, .39, .10, .08, .08, .6),
    (147, "raised", .02, -.34, .10, .08, .02, 0.),
    (148, "raised", -.04, .40, .12, .08, .04, -.35),
)
STEP_M = .08
SLIP = .88
THRESHOLDS_M = (.055, .065)


def motion(pose: np.ndarray, direction: int) -> np.ndarray:
    angle = float(pose[2])
    return direction * STEP_M * np.array((-math.sin(angle), math.cos(angle)))


def point_gap(robot: RobotGeometry, qpos: np.ndarray, pose: np.ndarray,
              delta: np.ndarray, cloud: np.ndarray) -> float:
    result = math.inf
    for fraction in np.linspace(0., 1., 21):
        future = pose.copy()
        future[:2] += fraction * delta
        robot.set_state(robot.pose_qpos(qpos, future))
        if len(cloud):
            result = min(result, float(np.min(RobotGeometry.solid_distances(
                cloud, robot.primitives()))))
    return result


def model_gap(room, obstacles, bodies, pose, delta) -> tuple[float, str, str]:
    closest = (math.inf, "", "")
    for fraction in np.linspace(0., 1., 21):
        room.actual[:] = pose
        room.actual[:2] += fraction * SLIP * delta
        room._forward()
        row = full_body_clearance(room, obstacles, bodies)
        if row[0] < closest[0]:
            closest = row
    return closest


def score(rows: list[dict], sensor: str, threshold: float) -> dict:
    pool = [row for row in rows if row["evaluator_valid_start_5cm"]]
    key = str(threshold)
    return {"eligible_actions": len(pool),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in pool),
            "false_safe": sum(row["actor_pass"][sensor][key] and
                              not row["evaluator_safe_5cm"] for row in pool),
            "false_block": sum(not row["actor_pass"][sensor][key] and
                               row["evaluator_safe_5cm"] for row in pool),
            "passed": sum(row["actor_pass"][sensor][key] for row in pool)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    preflight = []
    observations = []
    for case in CASES:
        seed, kind, crate_x, crate_y, size_x, size_y, height, yaw = case
        room = DirectStructuralRoom(kind=kind, mode="absent",
                                    side=1 if seed % 2 else -1)
        robot = RobotGeometry(room.model)
        try:
            room.layout(seed, absent=True, furnished=True)
            crate = room.model.geom("loop_crate_geom")
            crate.size[:] = (size_x / 2, size_y / 2, height / 2)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, crate_y, height / 2)
            room._forward()
            obstacles = scene_geoms(room)
            bodies = body_geoms(room, obstacles)
            crate_id = crate.id
            nearest_other = min((float(mujoco.mj_geomDistance(
                room.model, room.data, crate_id, gid, 10., None)),
                                 room.model.geom(gid).name)
                                for gid in obstacles if gid != crate_id)
            hazard_direction = -1 if crate_y < 0 else 1
            starts = []
            for pose_index in range(3):
                pose = np.array((0., 0., yaw))
                pose[:2] += pose_index * SLIP * motion(pose, hazard_direction)
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                start_gap = float(full_body_clearance(room, obstacles, bodies)[0])
                starts.append({"pose_index": pose_index, "pose_xyyaw": pose.tolist(),
                               "start_clearance_m": round(start_gap, 5),
                               "valid_start_5cm": start_gap >= .05})
                evaluator = {direction: model_gap(room, obstacles, bodies,
                                                  pose, motion(pose, direction))
                             for direction in (-1, 1)}
                room.actual[:] = pose
                room.estimated[:] = pose
                room._forward()
                qpos = room.data.qpos.copy()
                virtual = np.concatenate((virtual_cloud(room, .04),
                                           virtual_cloud(room, .11)))
                c1 = {name: c1_scan(room, pose, MOUNTS[name])
                      for name in ("front_30cm", "front_11cm")}
                shin, shin_counts = shin_scan(room, robot, {
                    "case": f"v291-seed{seed}-pose{pose_index}"})
                clouds = {"virtual_4_11": virtual,
                          "front_30cm": c1["front_30cm"][0],
                          "front_11cm": c1["front_11cm"][0],
                          "shin_two": shin["two"],
                          "shin_four": shin["four"]}
                observations.append({"seed": seed, "pose_index": pose_index,
                                     "valid_return_points": {name: len(cloud)
                                                             for name, cloud in clouds.items()},
                                     "c1_self_first": {name: c1[name][1]["self_first"]
                                                       for name in c1},
                                     "shin_counts": shin_counts})
                for direction in (-1, 1):
                    delta = motion(pose, direction)
                    gaps = {name: point_gap(robot, qpos, pose, delta, cloud)
                            for name, cloud in clouds.items()}
                    truth = evaluator[direction]
                    rows.append({"seed": seed, "kind": kind,
                                 "crate_full_height_m": height,
                                 "pose_index": pose_index,
                                 "pose_xyyaw": pose.tolist(),
                                 "direction": direction,
                                 "evaluator_valid_start_5cm": start_gap >= .05,
                                 "evaluator_safe_5cm": truth[0] >= .05,
                                 "evaluator_min_clearance_m": round(truth[0], 5),
                                 "evaluator_limiting_scene_geom": truth[2],
                                 "actor_point_gap_m": {name: None if not math.isfinite(gap)
                                                       else round(gap, 5)
                                                       for name, gap in gaps.items()},
                                 "actor_pass": {name: {str(threshold):
                                                       gap >= threshold
                                                       for threshold in THRESHOLDS_M}
                                                for name, gap in gaps.items()}})
            preflight.append({"seed": seed, "kind": kind,
                              "crate_center_xy_m": (crate_x, crate_y),
                              "crate_full_size_xyz_m": (size_x, size_y, height),
                              "body_yaw_rad": yaw,
                              "nearest_other_scene_gap_m": round(nearest_other[0], 5),
                              "nearest_other_scene_geom": nearest_other[1],
                              "starts": starts})
            room.capture_external().save(OUT / f"seed{seed}-{kind}.png")
            print(seed, kind, "height_cm", int(height * 100),
                  "valid_starts", sum(x["valid_start_5cm"] for x in starts),
                  "scene_gap", round(nearest_other[0], 3), flush=True)
        finally:
            room.close()
    sensors = rows[0]["actor_pass"]
    scores = {name: {str(threshold): score(rows, name, threshold)
                     for threshold in THRESHOLDS_M} for name in sensors}
    report = {"scope": "V291 preregistered ten new selected Direct Frame low-object layouts (seeds139-148): open/screen/doorway/raised topology, full 2/4/8cm object heights, varied x/y footprint and body yaw. Evaluator validates scene spacing and full-body 5cm start/action clearance before actor sensor use; only valid starts count in scores. Compare same-action 8cm local side steps with .88 slip under virtual4+11, C1-class front30/11 self-first and two/four noisy articulated shin 8x8 views. Freeze 5.5cm and 6.5cm point margins; no tuning within cases. Save renders, counts, actor gaps and post-hoc labels. Kinematic static room and point-only pass; no explicit observed-free, timing, native stop or live task.",
              "source_sha256": sha(SOURCE), "selected_visual_sha256": sha(selected_visual()),
              "case_specs": CASES, "preflight": preflight,
              "observations": observations, "rows": rows, "scores": scores}
    (OUT / "varied-low-clutter-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
