"""V288: fuse self-masked head/chest depth rays with shin sweep coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_lateral_clutter_depth_v285 import BASE as V283
from audit_lateral_clutter_sensors_v283 import DIRECTIONS, POSE_Y
from audit_shin_swept_coverage_v287 import (observe as shin_observe,
                                            nominal_free, proposed_exterior)
from preflight_lateral_clutter_v282 import SEED, make_room
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v288"
V285 = ROOT / "artifacts/ape-closed-loop-cove50-v285/lateral-clutter-depth-audit.json"
V287 = ROOT / "artifacts/ape-closed-loop-cove50-v287/shin-swept-coverage-audit.json"
DEPTH_HEADROOM_M = .02
HEAD_YAWS = (-1.2, 0., 1.2)


def capture(room, mask: SelfDepthMask, camera: str, label: str) -> dict:
    _, measured = room.capture(camera, depth=True)
    filtered, mask_info = mask.apply(measured, room.data.qpos.copy())
    short = filtered.copy()
    short[(short < .07) | (short > .5)] = np.nan
    origin, rotation = room.camera_pose_estimate(camera)
    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
    path = OUT / f"{label}.npz"
    np.savez_compressed(path, depth_ideal_m=filtered, depth_short_m=short,
                        camera_origin_m=origin, camera_rotation=rotation,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        actor_qpos=room.data.qpos.copy(), capture_time_s=room.time_s,
                        camera_fovy_deg=fovy)
    return {"ideal": filtered, "short": short, "origin": origin,
            "rotation": rotation, "fovy_deg": fovy, "source_sha256": sha(path),
            "masked_self_pixels": mask_info["masked_self_pixels"]}


def depth_free(samples: np.ndarray, frames: list[dict], mode: str) -> np.ndarray:
    free = np.zeros(len(samples), dtype=bool)
    if not len(samples):
        return free
    for frame in frames:
        depth = frame[mode]
        origin, rotation = frame["origin"], frame["rotation"]
        local = (samples - origin) @ rotation
        z = -local[:, 2]
        focal = depth.shape[0] / (2 * math.tan(math.radians(frame["fovy_deg"]) / 2))
        col = local[:, 0] / np.maximum(z, 1e-6) * focal + depth.shape[1] / 2
        row = depth.shape[0] / 2 - local[:, 1] / np.maximum(z, 1e-6) * focal
        visible = ((z > .08) & (col >= 0) & (col < depth.shape[1] - 1) &
                   (row >= 0) & (row < depth.shape[0] - 1))
        indices = np.flatnonzero(visible)
        if not len(indices):
            continue
        c = np.floor(col[indices]).astype(int)
        r = np.floor(row[indices]).astype(int)
        # All four surrounding pixels must have a farther valid first hit.
        # A hole or a surface edge therefore remains unknown.
        d = np.stack((depth[r, c], depth[r, c + 1],
                      depth[r + 1, c], depth[r + 1, c + 1]), axis=1)
        free[indices] |= np.all(np.isfinite(d) &
                                (d >= z[indices, None] + DEPTH_HEADROOM_M),
                                axis=1)
    return free


def main() -> None:
    prior = json.loads(V285.read_text())
    v283 = json.loads(V283.read_text())
    if len(prior["rows"]) != 6:
        raise ValueError("V285 action set changed")
    OUT.mkdir(parents=True, exist_ok=True)
    room = make_room(SEED, "present")
    robot = RobotGeometry(room.model)
    masks = {camera: SelfDepthMask(room.model, camera=camera, width=W, height=H)
             for camera in ("lower_rgb", "chest_low50_x45")}
    observations = []
    rows = []
    try:
        for index, y in enumerate(POSE_Y):
            pose = np.array((0., y, 0.))
            room.actual[:] = pose
            room.estimated[:] = pose
            room.head = 0.
            room._forward()
            qpos = room.data.qpos.copy()
            shins, shin_counts = shin_observe(room,
                                              f"v285-seed{SEED}-pose{index}")
            chest = capture(room, masks["chest_low50_x45"],
                            "chest_low50_x45", f"pose{index}-chest")
            heads = []
            for yaw in HEAD_YAWS:
                room.head = yaw
                room._forward()
                heads.append(capture(room, masks["lower_rgb"], "lower_rgb",
                                     f"pose{index}-head{yaw:+.1f}"))
            room.head = 0.
            room._forward()
            observations.append({"pose_y_m": y, "shin_counts": shin_counts,
                                 "depth_sources": {"chest": chest["source_sha256"],
                                                   **{f"head{yaw:+.1f}": frame[
                                                       "source_sha256"]
                                                      for yaw, frame in zip(HEAD_YAWS, heads)}}})
            for direction in DIRECTIONS:
                samples = proposed_exterior(robot, qpos, pose, direction)
                free_shin_two = nominal_free(samples, shins["two"])
                free_shin_four = nominal_free(samples, shins["four"])
                free_head_short = depth_free(samples, heads, "short")
                free_head_ideal = depth_free(samples, heads, "ideal")
                free_chest_short = depth_free(samples, [chest], "short")
                free_chest_ideal = depth_free(samples, [chest], "ideal")
                masks_by_source = {"shin_two": free_shin_two,
                                   "shin_four": free_shin_four,
                                   "head_sweep_short": free_head_short,
                                   "head_sweep_ideal": free_head_ideal,
                                   "chest_short": free_chest_short,
                                   "chest_ideal": free_chest_ideal,
                                   "shin_four_plus_head_short":
                                       free_shin_four | free_head_short,
                                   "shin_four_plus_head_ideal":
                                       free_shin_four | free_head_ideal,
                                   "shin_four_plus_head_chest_ideal":
                                       free_shin_four | free_head_ideal |
                                       free_chest_ideal}
                baseline = next(row for row in prior["rows"]
                                if row["pose_y_m"] == y and row["direction"] == direction)
                gaps = baseline["actor_point_gap_m"]
                point_pass = {
                    "shin_two": False,  # V283 two-shin result below
                    "shin_four": gaps["shin_four"] >= .065,
                    "head_sweep_short": gaps["head_sweep_short"] >= .065,
                    "head_sweep_ideal": gaps["head_sweep_ideal"] >= .065,
                    "chest_short": gaps["chest_short"] >= .065,
                    "chest_ideal": gaps["chest_ideal"] >= .065,
                    "shin_four_plus_head_short":
                        gaps["shin_four_plus_head_sweep_short"] >= .065,
                    "shin_four_plus_head_ideal":
                        gaps["shin_four_plus_head_sweep_ideal"] >= .065,
                    "shin_four_plus_head_chest_ideal":
                        min(gaps["shin_four"], gaps["head_sweep_ideal"],
                            gaps["chest_ideal"]) >= .065}
                # V285 has four-shin points only; V283 supplies the paired
                # two-shin point decision from the same selected-body pose.
                old = next(row for row in v283["rows"]
                           if row["pose_y_m"] == y and row["direction"] == direction)
                point_pass["shin_two"] = old["actor_pass"]["shin_two"]
                layouts = {}
                for name, free in masks_by_source.items():
                    bands = {}
                    for low, high in ((.015, .16), (.16, .32), (.32, .55)):
                        mask = (samples[:, 2] >= low) & (samples[:, 2] < high)
                        bands[f"{low:.3f}-{high:.3f}m"] = {
                            "samples": int(mask.sum()),
                            "nominal_free": int(np.sum(free & mask))}
                    layouts[name] = {"point_pass": point_pass[name],
                                     "nominal_free_surface": int(free.sum()),
                                     "unknown_surface": int((~free).sum()),
                                     "full_coverage_pass": bool(len(samples) and
                                                                free.all() and
                                                                point_pass[name]),
                                     "bands": bands}
                rows.append({"pose_y_m": y, "direction": direction,
                             "surface_samples": len(samples),
                             "evaluator_valid_start_5cm": baseline[
                                 "evaluator_valid_start_5cm"],
                             "evaluator_safe_5cm": baseline["evaluator_safe_5cm"],
                             "layouts": layouts})
                print("pose", index, "direction", direction,
                      "covered", {name: result["nominal_free_surface"]
                                  for name, result in layouts.items()}, flush=True)
    finally:
        for mask in masks.values():
            mask.close()
        room.close()
    def score(pool: list[dict]) -> dict:
        return {name: {"false_safe": sum(row["layouts"][name]["full_coverage_pass"] and
                                         not row["evaluator_safe_5cm"] for row in pool),
                       "false_block": sum(not row["layouts"][name]["full_coverage_pass"] and
                                          row["evaluator_safe_5cm"] for row in pool),
                       "passed": sum(row["layouts"][name]["full_coverage_pass"] for row in pool)}
                for name in rows[0]["layouts"]}
    scores = score(rows)
    valid_start_scores = score([row for row in rows
                                if row["evaluator_valid_start_5cm"]])
    report = {"scope": "V288 same six V283 Direct Frame low-clutter actions. Fuse nominal four-shin first-hit angular-cell free coverage with self-masked lower-head depth at yaw-1.2/0/+1.2 and fixed chest depth, each ideal8-150cm or short7-50cm mask. A surface point is depth-free only if four surrounding pixels have valid first hits >2cm beyond it; holes/self occlusion remain unknown. Query all newly entered leading selected-body 5cm-margin surface samples z1.5-55cm; require full coverage and V285/V283 6.5cm point pass, then compare evaluator labels. Source frames saved. This is still a same-pose, ideal-rendered optical diagnostic without calibrated beam footprint, asynchronous age, walking, or native stop; it can reject unsupported steps but cannot certify safety.",
              "source_sha256": sha(SOURCE), "v285_sha256": sha(V285),
              "v287_sha256": sha(V287),
              "v283_sha256": sha(V283), "observations": observations,
              "rows": rows, "scores": scores,
              "valid_start_scores": valid_start_scores}
    (OUT / "depth-shin-coverage-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
