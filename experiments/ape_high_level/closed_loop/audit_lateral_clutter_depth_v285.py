"""V285: paired low-object range/depth observations on the V282 fork."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_lateral_clutter_sensors_v283 import (ACTOR_POINT_MARGIN_M,
                                                DIRECTIONS, POSE_Y, actor_gap)
from audit_shin_range_v190 import scan as shin_scan
from preflight_lateral_clutter_v282 import SEED, make_room
from robot_geometry import RobotGeometry
from run import ROOT, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v285"
BASE = ROOT / "artifacts/ape-closed-loop-cove50-v283/lateral-clutter-sensor-audit.json"
CAMERAS = ("lower_rgb", "chest_low50_x45")


def project(depth: np.ndarray, camera_pose: tuple[np.ndarray, np.ndarray],
            fovy_deg: float) -> np.ndarray:
    height, width = depth.shape
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rows, cols = np.mgrid[4:height:8, 4:width:8]
    d = depth[rows, cols]
    valid = np.isfinite(d) & (d > .08) & (d < 1.5)
    rays = np.stack(((cols - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(d)), axis=-1)[valid]
    origin, rotation = camera_pose
    points = origin + (rays * d[valid, None]) @ rotation.T
    return points[(points[:, 2] >= .02) & (points[:, 2] <= .55)]


def capture(room, mask: SelfDepthMask, camera: str, qpos: np.ndarray,
            short: bool) -> tuple[np.ndarray, dict]:
    rgb, depth = room.capture(camera, depth=True)
    if short:
        depth[(depth < .07) | (depth > .5)] = np.nan
    filtered, info = mask.apply(depth, qpos)
    pose = room.camera_pose_estimate(camera)
    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
    points = project(filtered, pose, fovy)
    return points, {**info, "points": len(points),
                    "camera_origin_m": pose[0].tolist(),
                    "camera_fovy_deg": fovy}


def main() -> None:
    prior = json.loads(BASE.read_text())
    if len(prior["rows"]) != 6:
        raise ValueError("V283 action set changed")
    OUT.mkdir(parents=True, exist_ok=True)
    room = make_room(SEED, "present")
    robot = RobotGeometry(room.model)
    head_mask = SelfDepthMask(room.model, camera="lower_rgb",
                              width=640, height=480)
    chest_mask = SelfDepthMask(room.model, camera="chest_low50_x45",
                               width=640, height=480)
    rows = []
    observations = []
    try:
        for index, y in enumerate(POSE_Y):
            pose = np.array((0., y, 0.))
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            qpos = room.data.qpos.copy()
            shin, shin_counts = shin_scan(room, robot,
                                          {"case": f"v285-seed{SEED}-pose{index}"})
            clouds = {"shin_four": shin["four"]}
            counts = {"shin_four": shin_counts}
            for short in (False, True):
                name = "short" if short else "ideal"
                clouds[f"chest_{name}"], counts[f"chest_{name}"] = capture(
                    room, chest_mask, "chest_low50_x45", qpos, short)
                head_clouds = []
                for yaw in (-1.2, 0., 1.2):
                    room.head = yaw
                    room._forward()
                    head_qpos = room.data.qpos.copy()
                    cloud, count = capture(room, head_mask, "lower_rgb",
                                           head_qpos, short)
                    key = f"head_yaw{yaw:+.1f}_{name}"
                    clouds[key], counts[key] = cloud, count
                    head_clouds.append(cloud)
                clouds[f"head_sweep_{name}"] = np.concatenate(head_clouds)
                counts[f"head_sweep_{name}"] = {
                    "points": len(clouds[f"head_sweep_{name}"])}
            room.head = 0.
            room._forward()
            qpos = room.data.qpos.copy()
            for name in ("head_sweep_ideal", "head_sweep_short",
                         "chest_ideal", "chest_short"):
                clouds[f"shin_four_plus_{name}"] = np.concatenate(
                    (shin["four"], clouds[name]))
            observations.append({"pose_y_m": y,
                                 "counts": counts,
                                 "low_points_near_clutter_evaluator_only": {
                                     name: int(np.sum((np.abs(cloud[:, 0]) <= .07) &
                                                      (np.abs(cloud[:, 1] + .35) <= .06) &
                                                      (cloud[:, 2] <= .09)))
                                     for name, cloud in clouds.items()}})
            for direction in DIRECTIONS:
                gaps = {name: actor_gap(robot, qpos, pose, cloud, direction)
                        for name, cloud in clouds.items()}
                baseline = next(row for row in prior["rows"]
                                if row["pose_y_m"] == y and
                                row["direction"] == direction)
                rows.append({"pose_y_m": y, "direction": direction,
                             "evaluator_valid_start_5cm": baseline[
                                 "evaluator_valid_start_5cm"],
                             "actor_point_gap_m": {key: round(value, 5)
                                                   for key, value in gaps.items()},
                             "actor_pass": {key: value >= ACTOR_POINT_MARGIN_M
                                            for key, value in gaps.items()},
                             "evaluator_safe_5cm": baseline["evaluator_safe_5cm"],
                             "evaluator_min_clearance_m": baseline[
                                 "evaluator_min_clearance_m"]})
            print("pose", index, {key: counts[key]["points"] for key in
                  ("chest_ideal", "chest_short", "head_sweep_ideal",
                   "head_sweep_short")}, flush=True)
    finally:
        head_mask.close()
        chest_mask.close()
        room.close()
    keys = rows[0]["actor_pass"]
    def score(pool: list[dict]) -> dict:
        return {key: {"false_safe": sum(row["actor_pass"][key] and
                                        not row["evaluator_safe_5cm"] for row in pool),
                      "false_block": sum(not row["actor_pass"][key] and
                                         row["evaluator_safe_5cm"] for row in pool),
                      "passed": sum(row["actor_pass"][key] for row in pool)}
                for key in keys}
    scores = score(rows)
    valid_start_scores = score([row for row in rows
                                if row["evaluator_valid_start_5cm"]])
    report = {"scope": "V285 same six V283 selected-Direct-Frame lateral actions, static first-hit MuJoCo depth captured at fixed low-chest and lower-head yaw -1.2/0/+1.2; ideal 8-150cm or short 7-50cm mask, predicted self-depth agreement mask, stride8 point projection 2-55cm height, with/without four noisy shin views. Actor 6.5cm swept point gap; evaluator V283 5cm full-body labels only after decision. Head sweep is sequential but represented as same-pose fresh union: age and motion during looks are not modeled. No real stereo holes, specular error, physical module fit or native walking. Point-only pass is not observed-free proof.",
              "source_sha256": sha(SOURCE), "v283_sha256": sha(BASE),
              "observations": observations, "rows": rows, "scores": scores,
              "valid_start_scores": valid_start_scores}
    (OUT / "lateral-clutter-depth-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
