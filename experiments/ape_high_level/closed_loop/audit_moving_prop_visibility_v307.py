"""V307: which Direct Frame views observe an approaching side floor prop?

Paired snapshots at t=7.2/7.6s in the frozen V305 rooms compare four shin
first-hit views with upper long depth and deliberate lower short-depth looks.
Exact prop identity/geometry is used only by the evaluator after captures.
Head looks here are counterfactual simultaneous snapshots, not free live scans.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_localization_v297 import depth_variant
from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import CASES, HeldoutMovingRoom, scheduled_y
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v307"
TIMED_POSES = ((7.2, 0., 0.), (7.6, .0464, .08))
LOWER_YAWS = (1.2, 1.5)
SHORT_RANGE_M = (.07, .50)
LONG_RANGE_M = (.08, 2.50)


def prop_mask(room, camera: str) -> np.ndarray:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    segmented = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    gid = room.model.geom("loop_stool_geom").id
    return ((segmented[:, :, 0] == gid) &
            (segmented[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def camera_frame(room, mask: SelfDepthMask, camera: str,
                 depth_mode: str, seed: int, pose_index: int,
                 angle: float | None, branch: Path) -> dict:
    room.head = 0. if angle is None else angle
    room._forward()
    rgb, ideal = room.capture(camera, depth=True)
    if ideal is None:
        raise AssertionError("Rendered depth unavailable")
    depth = depth_variant(ideal, "short" if depth_mode == "short" else "noisy_long",
                          seed, pose_index * 10 + (0 if angle is None else round(angle * 10)))
    if depth_mode == "short":
        rng = np.random.default_rng(seed * 1009 + pose_index * 41 +
                                    round((angle or 0) * 10))
        depth += rng.normal(0., .01, depth.shape).astype(depth.dtype)
        depth[rng.random(depth.shape) < .15] = np.nan
    # Self prediction uses the actor's estimated base pose and joint/head
    # encoders. It receives neither actual XY nor the evaluator prop mask.
    camera_origin, rotation = room.camera_pose_estimate(camera)
    filtered, self_info = mask.apply(depth, room.pose_data.qpos.copy(),
                                     max_depth_m=LONG_RANGE_M[1])
    label = prop_mask(room, camera)  # evaluator-only after actor capture
    frame_name = (f"pose{pose_index}-{camera}-"
                  f"yaw{0 if angle is None else round(angle*10):02d}")
    actor_path = branch / f"{frame_name}-actor.npz"
    label_path = branch / f"{frame_name}-evaluator-label.npz"
    np.savez_compressed(actor_path, rgb=np.asarray(rgb), depth_m=filtered,
                        camera_origin_estimated_m=camera_origin,
                        camera_rotation_estimated=rotation,
                        joint_qpos=room.pose_data.qpos[7:],
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
    np.savez_compressed(label_path, prop_pixel_mask=label)
    return {"camera": camera, "head_yaw_rad": angle,
            "depth_mode": depth_mode,
            "actor_frame_path": str(actor_path.relative_to(ROOT)),
            "actor_frame_sha256": sha(actor_path),
            "evaluator_label_path": str(label_path.relative_to(ROOT)),
            "evaluator_label_sha256": sha(label_path),
            "evaluator_prop_rgb_pixels": int(label.sum()),
            "evaluator_prop_valid_depth_pixels": int((label &
                np.isfinite(filtered)).sum()),
            "evaluator_prop_depth_fraction": round(float((label &
                np.isfinite(filtered)).sum()) / max(1, int(label.sum())), 4),
            "total_valid_depth_pixels": int(np.isfinite(filtered).sum()),
            "self_masked_pixels": self_info["masked_self_pixels"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    montage_rows = []
    for case in CASES:
        seed, mode, *_ = case
        branch = OUT / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=branch)
        robot = RobotGeometry(room.model)
        lower_mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
        upper_mask = SelfDepthMask(room.model, camera="upper_rgb", width=W, height=H)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            observations = []
            for pose_index, (time_s, actual_y, estimated_y) in enumerate(TIMED_POSES):
                room.time_s = time_s
                room.actual[:] = (0., actual_y, 0.)
                room.estimated[:] = (0., estimated_y, 0.)
                room.head = 0.
                room._update_moving_obstacle()
                room._forward()
                clouds, counts = shin_scan(room, robot, {
                    "case": f"v307-seed{seed}-pose{pose_index}"})
                prop_center = np.array((0., scheduled_y(time_s, case)))
                rays = clouds["four"]
                labeled = (np.linalg.norm(rays[:, :2] - prop_center, axis=1) <= .11
                           if len(rays) else np.zeros(0, dtype=bool))
                local = rays.copy()
                local[:, :2] -= room.actual[:2]
                scan_path = branch / f"pose{pose_index}-shin-body-local.npz"
                np.savez_compressed(scan_path, four=local,
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    head_encoder_rad=room.head,
                                    capture_time_s=time_s)
                upper = camera_frame(room, upper_mask, "upper_rgb", "long",
                                     seed, pose_index, None, branch)
                lower = [camera_frame(room, lower_mask, "lower_rgb", "short",
                                      seed, pose_index, angle, branch)
                         for angle in LOWER_YAWS]
                observations.append({
                    "capture_time_s": time_s,
                    "actor_estimated_xyyaw": room.estimated.tolist(),
                    "evaluator_actual_xyyaw": room.actual.tolist(),
                    "evaluator_prop_center_y_m": round(float(prop_center[1]), 5),
                    "shin_actor_frame_path": str(scan_path.relative_to(ROOT)),
                    "shin_actor_frame_sha256": sha(scan_path),
                    "shin_valid_returns": len(rays),
                    "shin_counts": dict(counts),
                    "evaluator_shin_returns_on_prop": int(labeled.sum()),
                    "evaluator_shin_prop_median_y_m": (
                        round(float(np.median(rays[labeled, 1])), 5)
                        if np.any(labeled) else None),
                    "upper": upper, "lower": lower})
            # A small reviewer view: the raw side camera plus exact scorer
            # outline. The outline is never included in actor-frame files.
            preview = np.load(observations[1]["lower"][-1]["actor_frame_path"])
            label = np.load(observations[1]["lower"][-1]["evaluator_label_path"])[
                "prop_pixel_mask"]
            canvas = Image.fromarray(preview["rgb"].copy())
            if label.any():
                ys, xs = np.where(label)
                ImageDraw.Draw(canvas).rectangle((int(xs.min()), int(ys.min()),
                                                  int(xs.max()), int(ys.max())),
                                                 outline=(255, 40, 20), width=3)
            montage_rows.append((seed, canvas.resize((320, 240)), observations))
            cases.append({"case": case, "observations": observations})
            print(seed, mode, "shin prop returns",
                  [o["evaluator_shin_returns_on_prop"] for o in observations],
                  "lower valid pixels at t7.6",
                  [f["evaluator_prop_valid_depth_pixels"]
                   for f in observations[1]["lower"]], flush=True)
        finally:
            lower_mask.close()
            upper_mask.close()
            room.close()
    montage = Image.new("RGB", (960, 520), (247, 245, 239))
    draw = ImageDraw.Draw(montage)
    for index, (seed, canvas, obs) in enumerate(montage_rows):
        x = (index % 3) * 320
        y = (index // 3) * 260
        montage.paste(canvas, (x, y + 20))
        draw.text((x + 6, y + 3), f"seed {seed}: prop pixels "
                  f"{obs[1]['lower'][-1]['evaluator_prop_valid_depth_pixels']}",
                  fill=(20, 25, 30))
    montage_path = OUT / "side-depth-visibility-contact-sheet.png"
    montage.save(montage_path)
    report = {"scope": "V307 paired pose snapshots t7.2/7.6s in the six frozen V305 selected Direct Frame moving-floor-prop rooms. At each actor command/actual pose, compare noisy four-shin first-hit local rays, hypothetical upper head 8–250cm RGB-D and side-looking lower 7–50cm RGB-D at +1.2/+1.5rad with 1cm depth noise/15% holes and estimated-pose self masking. Raw actor RGB/depth/rays saved separately from evaluator segmentation masks and prop labels; label counts quantify visibility, not autonomous object tracking. Head yaw alternatives are captured at the same timestamp without modeling 0.4s actuation, so this is an observability upper bound, not an executable sensor schedule or physical package. Kinematic base and ideal MuJoCo materials remain.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "case_specs": CASES, "timed_poses": TIMED_POSES,
              "lower_yaws_rad": LOWER_YAWS,
              "contact_sheet_path": str(montage_path.relative_to(ROOT)),
              "contact_sheet_sha256": sha(montage_path),
              "cases": cases}
    (OUT / "moving-prop-visibility-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
