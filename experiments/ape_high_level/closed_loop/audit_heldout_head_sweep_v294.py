"""V294: fresh Direct Frame room split for the wider head-depth hypothesis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_shin_range_v190 import scan as shin_scan
from audit_varied_low_clutter_v291 import model_gap, motion, point_gap
from audit_wider_head_sweep_v293 import capture as head_capture
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v294"
V293 = ROOT / "artifacts/ape-closed-loop-cove50-v293/wider-head-sweep-audit.json"
# Frozen before rendering/sensor decisions. Same tuple units as V291:
# seed, structure, center x/y, full box x/y/z, body yaw.
CASES = (
    (149, "open", .09, -.29, .08, .06, .02, -.55),
    (150, "doorway", -.09, .32, .12, .06, .04, .55),
    (151, "screen", -.02, -.44, .08, .12, .02, .20),
    (152, "raised", .10, .36, .06, .08, .08, -.50),
    (153, "open", -.10, -.41, .14, .08, .04, .75),
    (154, "screen", .00, .31, .10, .04, .02, .10),
    (155, "doorway", .07, -.38, .08, .10, .08, -.75),
    (156, "raised", -.08, .45, .12, .06, .04, -.20),
    (157, "screen", -.03, -.34, .04, .04, .02, .55),
    (158, "open", .04, .39, .08, .08, .04, -.55),
)
ANGLES = (1.2, 1.5)
MARGIN_M = .065
MIN_HEAD_POINTS = 10


def score(rows: list[dict], key: str) -> dict:
    eligible = [row for row in rows if row["scene_valid"] and
                row["evaluator_valid_start_5cm"]]
    return {"eligible_actions": len(eligible),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in eligible),
            "false_safe": sum(row["actor_pass"][key] and
                              not row["evaluator_safe_5cm"] for row in eligible),
            "false_block": sum(not row["actor_pass"][key] and
                               row["evaluator_safe_5cm"] for row in eligible),
            "passed": sum(row["actor_pass"][key] for row in eligible)}


def main() -> None:
    prior = json.loads(V293.read_text())
    if prior["scores"]["head_1p5"]["false_safe"] != 0:
        raise ValueError("Development head hypothesis changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    preflight = []
    for case in CASES:
        seed, kind, crate_x, crate_y, size_x, size_y, height, yaw = case
        room = DirectStructuralRoom(kind=kind, mode="absent",
                                    side=1 if seed % 2 else -1)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
        try:
            room.layout(seed, absent=True, furnished=True)
            crate = room.model.geom("loop_crate_geom")
            crate.size[:] = (size_x / 2, size_y / 2, height / 2)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, crate_y, height / 2)
            room._forward()
            obstacles = scene_geoms(room)
            bodies = body_geoms(room, obstacles)
            nearest_scene = min((float(mujoco.mj_geomDistance(
                room.model, room.data, crate.id, gid, 10., None)),
                                 room.model.geom(gid).name)
                                for gid in obstacles if gid != crate.id)
            scene_valid = nearest_scene[0] >= 0.
            hazard = -1 if crate_y < 0 else 1
            starts = []
            for pose_index in range(3):
                pose = np.array((0., 0., yaw))
                pose[:2] += pose_index * .88 * motion(pose, hazard)
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                start_gap = float(full_body_clearance(room, obstacles, bodies)[0])
                labels = {direction: model_gap(room, obstacles, bodies,
                                               pose, motion(pose, direction))
                          for direction in (-1, 1)}
                starts.append({"pose_index": pose_index,
                               "pose_xyyaw": pose.tolist(),
                               "start_clearance_m": round(start_gap, 5),
                               "valid_start_5cm": start_gap >= .05})
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                shin, shin_counts = shin_scan(room, robot, {
                    "case": f"v294-seed{seed}-pose{pose_index}"})
                for direction in (-1, 1):
                    clouds, frames = [], []
                    for angle in ANGLES:
                        cloud, info = head_capture(room, mask, seed,
                                                   pose_index, direction,
                                                   angle, output_root=OUT)
                        clouds.append(cloud)
                        frames.append(info)
                    qpos = room.data.qpos.copy()
                    head = np.concatenate(clouds)
                    fused = np.concatenate((shin["four"], head))
                    candidate = {"shin_four": shin["four"],
                                 "head_1p2": clouds[0],
                                 "head_1p5": clouds[1],
                                 "head_sweep": head,
                                 "shin_four_plus_head_sweep": fused}
                    gaps = {name: point_gap(robot, qpos, pose,
                                            motion(pose, direction), cloud)
                            for name, cloud in candidate.items()}
                    passes = {name: gap >= MARGIN_M for name, gap in gaps.items()}
                    passes["head_sweep_unknown_stop"] = (
                        len(head) >= MIN_HEAD_POINTS and passes["head_sweep"])
                    truth = labels[direction]
                    rows.append({"seed": seed, "kind": kind,
                                 "crate_full_height_m": height,
                                 "pose_index": pose_index,
                                 "direction": direction,
                                 "scene_valid": scene_valid,
                                 "evaluator_valid_start_5cm": start_gap >= .05,
                                 "evaluator_safe_5cm": truth[0] >= .05,
                                 "evaluator_min_clearance_m": round(truth[0], 5),
                                 "evaluator_limiting_scene_geom": truth[2],
                                 "actor_point_gap_m": {name: None if not math.isfinite(gap)
                                                       else round(gap, 5)
                                                       for name, gap in gaps.items()},
                                 "actor_pass": passes,
                                 "head_frames": frames,
                                 "shin_valid_returns": len(shin["four"]),
                                 "shin_self_first": shin_counts.get("two_self_first", 0) +
                                                    shin_counts.get("four_self_first", 0)})
            room.actual[:] = (0., 0., yaw)
            room.head = 0.
            room._forward()
            room.capture_external().save(OUT / f"seed{seed}-{kind}.png")
            preflight.append({"seed": seed, "kind": kind,
                              "crate_center_xy_m": (crate_x, crate_y),
                              "crate_full_size_xyz_m": (size_x, size_y, height),
                              "body_yaw_rad": yaw,
                              "scene_valid": scene_valid,
                              "nearest_scene_gap_m": round(nearest_scene[0], 5),
                              "nearest_scene_geom": nearest_scene[1],
                              "starts": starts})
            print(seed, kind, "valid_scene", scene_valid,
                  "valid_starts", sum(x["valid_start_5cm"] for x in starts),
                  flush=True)
        finally:
            mask.close()
            room.close()
    scores = {key: score(rows, key) for key in rows[0]["actor_pass"]}
    report = {"scope": "V294 fresh predeclared seeds149-158 and ten selected Direct Frame open/doorway/screen/raised low-clutter variants after V293 wider-head hypothesis. Full 2/4/8cm boxes vary x/y, footprint and body yaw; independently check scene spacing, start and swept full-body 5cm clearance before actor observations. Compare four noisy articulated shin 8x8 views, side-looking 7-50cm self-masked lower-head depth at local yaw±1.2 and ±1.5, their union and an empty/under10-return unknown-stop rule at frozen 6.5cm point margin. Save every RGB/depth image, estimated pose, actor qpos, point gap and post-hoc evaluator label. A point pass never certifies the full swept space. Static room, ideal rendered depth, kinematic step, no native stop/capture-age model; these are fresh layouts but not physical-domain validation.",
              "source_sha256": sha(SOURCE), "v293_sha256": sha(V293),
              "selected_visual_sha256": sha(selected_visual()),
              "case_specs": CASES, "preflight": preflight,
              "rows": rows, "scores": scores}
    (OUT / "heldout-head-sweep-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
