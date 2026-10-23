"""V309: self-mask tolerance tradeoff for noisy side depth on Direct Frame.

Re-render V307's t7.6 +1.2rad lower-head frame with the same 1cm Gaussian
depth noise and 15% holes. Freeze 2.5/4/6cm self agreement alternatives, then
score their point gate and evaluator-labeled real prop pixels separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_direct_localization_v297 import depth_variant
from audit_lateral_clutter_depth_v285 import project
from audit_lateral_clutter_sensors_v283 import actor_gap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import CASES, HeldoutMovingRoom
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v309"
V307 = ROOT / "artifacts/ape-closed-loop-cove50-v307/moving-prop-visibility-audit.json"
V308 = ROOT / "artifacts/ape-closed-loop-cove50-v308/moving-depth-action-audit.json"
AGREEMENTS_M = (.025, .04, .06)
HEAD_YAW_RAD = 1.2
TIME_S = 7.6
ACTOR_MARGIN_M = .065


def segmented_scene(room) -> tuple[np.ndarray, np.ndarray]:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera="lower_rgb")
    segmented = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    gid = room.model.geom("loop_stool_geom").id
    label = ((segmented[:, :, 0] == gid) &
             (segmented[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
    return segmented, label


def limiting_pixel(robot: RobotGeometry, qpos: np.ndarray,
                   pose: np.ndarray, depth: np.ndarray,
                   camera_pose: tuple[np.ndarray, np.ndarray],
                   fovy_deg: float) -> dict:
    """Post-hoc image provenance of the point governing the raw depth gate."""
    height, width = depth.shape
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rows, cols = np.mgrid[4:height:8, 4:width:8]
    sampled = depth[rows, cols]
    valid = np.isfinite(sampled) & (sampled > .08) & (sampled < 1.5)
    v, u, distances = rows[valid], cols[valid], sampled[valid]
    rays = np.stack(((u - width / 2) / focal,
                     -(v - height / 2) / focal,
                     -np.ones_like(distances)), axis=-1)
    origin, rotation = camera_pose
    points = origin + (rays * distances[:, None]) @ rotation.T
    valid_height = (points[:, 2] >= .02) & (points[:, 2] <= .55)
    points, u, v = points[valid_height], u[valid_height], v[valid_height]
    best = (float("inf"), -1, 0.)
    for fraction in np.linspace(0., 1., 21):
        future = pose.copy()
        future[1] += .08 * fraction
        robot.set_state(robot.pose_qpos(qpos, future))
        gaps = RobotGeometry.solid_distances(points, robot.primitives())
        index = int(np.argmin(gaps))
        if gaps[index] < best[0]:
            best = (float(gaps[index]), index, float(fraction))
    return {"gap_m": round(best[0], 5),
            "pixel_uv": [int(u[best[1]]), int(v[best[1]])],
            "projected_point_m": points[best[1]].round(5).tolist(),
            "step_fraction": best[2]}


def score(rows: list[dict], agreement: float) -> dict:
    key = str(agreement)
    return {"actions": len(rows),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in rows),
            "false_safe": sum(row["alternatives"][key]["actor_pass"] and
                              not row["evaluator_safe_5cm"] for row in rows),
            "false_block": sum(not row["alternatives"][key]["actor_pass"] and
                               row["evaluator_safe_5cm"] for row in rows),
            "total_real_prop_pixels_masked": sum(
                row["alternatives"][key]["evaluator_prop_pixels_masked"]
                for row in rows)}


def main() -> None:
    prior = json.loads(V307.read_text())
    action = json.loads(V308.read_text())
    labels = {row["case"][0]: row for row in action["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=OUT)
        robot = RobotGeometry(room.model)
        masks = {agreement: SelfDepthMask(
            room.model, camera="lower_rgb", width=W, height=H,
            agreement_m=agreement) for agreement in AGREEMENTS_M}
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            room.time_s = TIME_S
            room.actual[:] = (0., .0464, 0.)
            room.estimated[:] = (0., .08, 0.)
            room.head = HEAD_YAW_RAD
            room._update_moving_obstacle()
            room._forward()
            _, ideal = room.capture("lower_rgb", depth=True)
            if ideal is None:
                raise AssertionError("Rendered depth unavailable")
            measured = depth_variant(ideal, "short", seed, 22)
            rng = np.random.default_rng(seed * 1009 + 41 + 12)
            measured += rng.normal(0., .01, measured.shape).astype(measured.dtype)
            measured[rng.random(measured.shape) < .15] = np.nan
            camera_pose = room.camera_pose_estimate("lower_rgb")
            actor_qpos = room.pose_data.qpos.copy()
            segmented, label = segmented_scene(room)
            alternatives = {}
            for agreement, mask in masks.items():
                filtered, info = mask.apply(measured, actor_qpos,
                                             max_depth_m=2.5)
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                points = project(filtered, camera_pose, fovy)
                # The body would return to yaw0 before this step. Pose is the
                # actor's estimate; the evaluator label never enters the gap.
                qpos = actor_qpos.copy()
                qpos[room.head_addr] = 0.
                gap = actor_gap(robot, qpos, room.estimated.copy(), points, 1)
                key = str(agreement)
                alternatives[key] = {
                    "actor_points": len(points),
                    "actor_point_gap_m": round(float(gap), 5),
                    "actor_pass": bool(gap >= ACTOR_MARGIN_M),
                    "self_masked_pixels": info["masked_self_pixels"],
                    "closer_surface_pixels_preserved":
                        info["closer_surface_pixels_preserved"],
                    "evaluator_prop_pixels_total": int(label.sum()),
                    "evaluator_prop_valid_depth_pixels": int((label &
                        np.isfinite(filtered)).sum()),
                    "evaluator_prop_pixels_masked": int((label &
                        np.isfinite(measured) & ~np.isfinite(filtered)).sum())}
                if agreement == .04:
                    limiting = limiting_pixel(robot, qpos, room.estimated.copy(),
                                              filtered, camera_pose, fovy)
                    if abs(limiting["gap_m"] - round(float(gap), 5)) > 1e-5:
                        raise AssertionError("Pixel provenance and gate disagree")
                    u, v = limiting["pixel_uv"]
                    gid = int(segmented[v, u, 0])
                    limiting["evaluator_rendered_geom"] = (
                        room.model.geom(gid).name
                        if int(segmented[v, u, 1]) ==
                        int(mujoco.mjtObj.mjOBJ_GEOM) and 0 <= gid < room.model.ngeom
                        else None)
                    limiting["evaluator_is_moving_prop"] = bool(label[v, u])
                    alternatives[key]["evaluator_limiting_pixel"] = limiting
            row = {"case": case,
                   "evaluator_safe_5cm": labels[seed]["evaluator_safe_5cm"],
                   "evaluator_min_dynamic_clearance_m": labels[seed][
                       "evaluator_min_dynamic_clearance_m"],
                   "alternatives": alternatives}
            if seed == 146:
                # A concrete scorer-only diagnosis of one self ghost. The
                # image pixel belongs to the Direct Frame forearm visual.
                v, u = 468, 236
                reference = masks[AGREEMENTS_M[0]]
                reference.data.qpos[:] = actor_qpos
                mujoco.mj_forward(reference.model, reference.data)
                reference.renderer.enable_depth_rendering()
                reference.renderer.update_scene(
                    reference.data, camera="lower_rgb",
                    scene_option=reference.option)
                predicted = reference.renderer.render().copy()
                reference.renderer.disable_depth_rendering()
                gid = int(segmented[v, u, 0])
                row["evaluator_seed146_forearm_pixel"] = {
                    "pixel_uv": [u, v],
                    "measured_depth_m": round(float(measured[v, u]), 5),
                    "predicted_self_depth_m": round(float(predicted[v, u]), 5),
                    "difference_m": round(float(measured[v, u] -
                                                predicted[v, u]), 5),
                    "segmented_geom": room.model.geom(gid).name}
            rows.append(row)
            print(seed, mode, {key: (value["actor_pass"],
                                    value["evaluator_prop_pixels_masked"])
                               for key, value in alternatives.items()}, flush=True)
        finally:
            for mask in masks.values():
                mask.close()
            room.close()
    scores = {str(agreement): score(rows, agreement)
              for agreement in AGREEMENTS_M}
    report = {"scope": "V309 revealed V307 t7.6s +1.2rad short-depth frames in six selected Direct Frame moving low-prop rooms. Replay identical 7–50cm rendered depth with deterministic 1cm Gaussian noise/15% holes through self-mask agreement bands2.5/4/6cm. Actor sees only estimated pose, joint model and filtered depth; evaluator prop segmentation counts real obstacle pixels lost to each mask and V308 labels score 5cm dynamic action safety. A wider tolerance may suppress noisy self ghosts but can erase a genuinely closer prop; no agreement band is promoted without a new split, active camera timing and native stop. The seed146 forearm pixel is a post-hoc scorer diagnosis, not an actor label.",
              "source_sha256": sha(SOURCE), "v307_sha256": sha(V307),
              "v308_sha256": sha(V308),
              "selected_visual_sha256": sha(selected_visual()),
              "agreements_m": AGREEMENTS_M, "rows": rows, "scores": scores}
    (OUT / "self-depth-noise-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
