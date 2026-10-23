"""V308: action-level cost of using dense side depth as raw obstacle points.

Replay V307 actor frames at t=7.6s for one 8cm positive-side step. Compare
four-shin, all short-depth points and an evaluator-only prop-pixel oracle to
the independently sampled moving full-body clearance. The oracle is a sensor
information bound, never an actor input or deployable segmentation algorithm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_lateral_clutter_depth_v285 import project
from audit_lateral_clutter_sensors_v283 import actor_gap
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v308"
V307 = ROOT / "artifacts/ape-closed-loop-cove50-v307/moving-prop-visibility-audit.json"
MARGIN_M = .065
EVALUATOR_MARGIN_M = .05
STEP_M = .08
SLIP = .94
START_T = 7.6


def score(rows: list[dict], key: str) -> dict:
    valid = [row for row in rows if row["evaluator_valid_start_5cm"]]
    return {"valid_start_actions": len(valid),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in valid),
            "false_safe": sum(row["actor_pass"][key] and
                              not row["evaluator_safe_5cm"] for row in valid),
            "false_block": sum(not row["actor_pass"][key] and
                               row["evaluator_safe_5cm"] for row in valid),
            "passed": sum(row["actor_pass"][key] for row in valid)}


def main() -> None:
    prior = json.loads(V307.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in prior["cases"]:
        case = tuple(entry["case"])
        seed, mode, *_ = case
        sample = entry["observations"][1]
        room = HeldoutMovingRoom(case=case, pose_source="command",
                                 branch=OUT)
        robot = RobotGeometry(room.model)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            room.time_s = START_T
            room.actual[:] = (0., .0464, 0.)
            room.estimated[:] = (0., .08, 0.)
            room.head = 0.
            room._update_moving_obstacle()
            room._forward()
            qpos = room.data.qpos.copy()
            pose = room.estimated.copy()
            shin_local = np.load(sample["shin_actor_frame_path"])["four"]
            shin = shin_local.copy()
            shin[:, :2] += pose[:2]
            clouds = {"shin_four": shin}
            count = {"shin_four": len(shin)}
            for frame in sample["lower"]:
                yaw = frame["head_yaw_rad"]
                actor = np.load(frame["actor_frame_path"])
                label = np.load(frame["evaluator_label_path"])[
                    "prop_pixel_mask"]
                camera_pose = (actor["camera_origin_estimated_m"],
                               actor["camera_rotation_estimated"])
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                all_points = project(actor["depth_m"], camera_pose, fovy)
                oracle_points = project(np.where(label, actor["depth_m"], np.nan),
                                        camera_pose, fovy)
                key = f"head_{yaw:.1f}"
                clouds[key] = all_points
                clouds[f"{key}_oracle_prop_only"] = oracle_points
                clouds[f"shin_plus_{key}"] = np.concatenate((shin, all_points))
                count[key] = len(all_points)
                count[f"{key}_oracle_prop_only"] = len(oracle_points)
                count[f"shin_plus_{key}"] = len(clouds[f"shin_plus_{key}"])
            actor_gaps = {name: actor_gap(robot, qpos, pose, cloud, 1)
                          for name, cloud in clouds.items()}
            obstacles = scene_geoms(room)
            bodies = body_geoms(room, obstacles)
            start_gap = float(full_body_clearance(room, obstacles, bodies)[0])
            closest = (float("inf"), "", "", 0.)
            for fraction in np.linspace(0., 1., 21):
                room.time_s = START_T + .4 * fraction
                room._update_moving_obstacle()
                room.actual[:] = (0., .0464 + SLIP * STEP_M * fraction, 0.)
                room._forward()
                gap, robot_name, scene_name = full_body_clearance(
                    room, obstacles, bodies)
                if gap < closest[0]:
                    closest = (float(gap), robot_name, scene_name,
                               float(fraction))
            rows.append({"case": case,
                         "actor_point_count": count,
                         "actor_point_gap_m": {name: None if not np.isfinite(gap)
                                               else round(float(gap), 5)
                                               for name, gap in actor_gaps.items()},
                         "actor_pass": {name: bool(gap >= MARGIN_M)
                                        for name, gap in actor_gaps.items()},
                         "evaluator_start_clearance_m": round(start_gap, 5),
                         "evaluator_valid_start_5cm": start_gap >= EVALUATOR_MARGIN_M,
                         "evaluator_min_dynamic_clearance_m": round(closest[0], 5),
                         "evaluator_limiting_robot_geom": closest[1],
                         "evaluator_limiting_scene_geom": closest[2],
                         "evaluator_fraction_at_minimum": closest[3],
                         "evaluator_safe_5cm": closest[0] >= EVALUATOR_MARGIN_M})
            print(seed, mode, "truth", round(closest[0], 4),
                  "shin", round(actor_gaps["shin_four"], 4),
                  "head1.2", round(actor_gaps["head_1.2"], 4),
                  "oracle", round(actor_gaps["head_1.2_oracle_prop_only"], 4),
                  flush=True)
        finally:
            room.close()
    scores = {name: score(rows, name) for name in rows[0]["actor_pass"]}
    report = {"scope": "V308 frozen V307 actor sensor frames at t7.6s before a hypothetical 8cm positive-side step, selected Direct Frame six changed moving-prop rooms. Compare 6.5cm point-gap decisions from four articulated shin rays, all self-masked 7–50cm lower depth at +1.2/+1.5rad, their union, and evaluator-only prop-labeled depth points as an information upper bound. Hidden evaluator advances body and scripted prop every20ms at .94 slip and checks all collision solids against a 5cm margin; invalid starts excluded. Head look actuation time and native stop are omitted. Oracle prop mask is never an actor sensor or deployable algorithm; all-depth false blocks expose the need for floor/self separation and moving-cluster association before using dense points. Six actions cannot establish general safety.",
              "source_sha256": sha(SOURCE),
              "v307_sha256": sha(V307),
              "selected_visual_sha256": sha(selected_visual()),
              "actor_margin_m": MARGIN_M,
              "evaluator_margin_m": EVALUATOR_MARGIN_M,
              "step_m": STEP_M, "slip_evaluator_only": SLIP,
              "rows": rows, "scores": scores}
    (OUT / "moving-depth-action-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
