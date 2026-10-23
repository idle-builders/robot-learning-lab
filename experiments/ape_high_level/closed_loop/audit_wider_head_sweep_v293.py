"""V293: paired ±1.2/±1.5 head-depth looks in V291 varied rooms."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_lateral_clutter_depth_v285 import project
from audit_shin_range_v190 import scan as shin_scan
from audit_varied_low_clutter_v291 import CASES, motion, point_gap
from direct_structural_rooms_v270 import DirectStructuralRoom
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v293"
V291 = ROOT / "artifacts/ape-closed-loop-cove50-v291/varied-low-clutter-audit.json"
HEAD_ANGLES = (1.2, 1.5)
POINT_MARGIN_M = .065
MIN_USABLE_POINTS = 10


def capture(room, mask, seed: int, pose_index: int, direction: int,
            angle: float, *, output_root: Path = OUT) -> tuple[np.ndarray, dict]:
    room.head = direction * angle
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    depth[(depth < .07) | (depth > .5)] = np.nan
    filtered, info = mask.apply(depth, room.data.qpos.copy())
    origin, rotation = room.camera_pose_estimate("lower_rgb")
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    points = project(filtered, (origin, rotation), fovy)
    path = output_root / f"seed{seed}-pose{pose_index}-dir{direction:+d}-yaw{angle:.1f}.npz"
    np.savez_compressed(path, rgb=np.asarray(rgb), depth_short_m=filtered,
                        camera_origin_m=origin, camera_rotation=rotation,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        actor_qpos=room.data.qpos.copy())
    return points, {"points": len(points),
                    "masked_self_pixels": info["masked_self_pixels"],
                    "source_sha256": sha(path)}


def score(rows: list[dict], key: str) -> dict:
    pool = [row for row in rows if row["evaluator_valid_start_5cm"]]
    return {"eligible_actions": len(pool),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in pool),
            "false_safe": sum(row["actor_pass"][key] and
                              not row["evaluator_safe_5cm"] for row in pool),
            "false_block": sum(not row["actor_pass"][key] and
                               row["evaluator_safe_5cm"] for row in pool),
            "passed": sum(row["actor_pass"][key] for row in pool)}


def main() -> None:
    prior = json.loads(V291.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        seed, kind, crate_x, crate_y, size_x, size_y, height, yaw = case
        room = DirectStructuralRoom(kind=kind, mode="absent",
                                    side=1 if seed % 2 else -1)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
        try:
            room.layout(seed, absent=True, furnished=True)
            geom = room.model.geom("loop_crate_geom")
            geom.size[:] = (size_x / 2, size_y / 2, height / 2)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, crate_y, height / 2)
            hazard = -1 if crate_y < 0 else 1
            for pose_index in range(3):
                pose = np.array((0., 0., yaw))
                pose[:2] += pose_index * .88 * motion(pose, hazard)
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                shin, _ = shin_scan(room, robot, {
                    "case": f"v291-seed{seed}-pose{pose_index}"})
                for direction in (-1, 1):
                    frames = []
                    clouds = []
                    for angle in HEAD_ANGLES:
                        cloud, info = capture(room, mask, seed, pose_index,
                                              direction, angle)
                        clouds.append(cloud)
                        frames.append(info)
                    qpos = room.data.qpos.copy()
                    by_name = {"head_1p2": clouds[0],
                               "head_1p5": clouds[1],
                               "head_sweep": np.concatenate(clouds),
                               "shin_four_plus_head_sweep":
                                   np.concatenate((shin["four"], *clouds))}
                    gaps = {name: point_gap(robot, qpos, pose,
                                            motion(pose, direction), cloud)
                            for name, cloud in by_name.items()}
                    baseline = next(row for row in prior["rows"]
                                    if row["seed"] == seed and
                                    row["pose_index"] == pose_index and
                                    row["direction"] == direction)
                    pass_rows = {name: gap >= POINT_MARGIN_M
                                 for name, gap in gaps.items()}
                    pass_rows["head_sweep_unknown_stop"] = (
                        sum(len(cloud) for cloud in clouds) >= MIN_USABLE_POINTS and
                        pass_rows["head_sweep"])
                    rows.append({"seed": seed, "kind": kind,
                                 "full_height_m": height,
                                 "pose_index": pose_index, "direction": direction,
                                 "evaluator_valid_start_5cm": baseline[
                                     "evaluator_valid_start_5cm"],
                                 "evaluator_safe_5cm": baseline[
                                     "evaluator_safe_5cm"],
                                 "evaluator_min_clearance_m": baseline[
                                     "evaluator_min_clearance_m"],
                                 "v291_shin_four_point_gap_m": baseline[
                                     "actor_point_gap_m"]["shin_four"],
                                 "head_frames": frames,
                                 "actor_point_gap_m": {name: None if not math.isfinite(gap)
                                                       else round(gap, 5)
                                                       for name, gap in gaps.items()},
                                 "actor_pass": pass_rows})
            print(seed, kind, flush=True)
        finally:
            mask.close()
            room.close()
    keys = rows[0]["actor_pass"]
    scores = {key: score(rows, key) for key in keys}
    report = {"scope": "V293 frozen V291 ten varied Direct Frame low-object rooms and 56 valid-start actions; side-looking short 7-50cm lower-head rendered depth at local yaw±1.2 and ±1.5 (selected yaw joint limit), separately and unioned, with predicted self mask and stride8 point projection. Compare 6.5cm point rule with V291 four-shin points and head-fusion. Report an additional head-only empty/under10-points unknown-stop rule. Actor sees RGB/depth, encoder qpos and estimated pose, not object labels; evaluator labels copied only after point decisions. Static same-pose sequential looks, ideal depth/no holes or capture age, package and head clearances unqualified; point-only pass is not observed-free proof. This is a follow-up to V292's targeted failure diagnosis, so it is not an independent held-out split.",
              "source_sha256": sha(SOURCE), "v291_sha256": sha(V291),
              "head_angles_abs_rad": HEAD_ANGLES,
              "rows": rows, "scores": scores}
    (OUT / "wider-head-sweep-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
