"""Use an independent current RGB-D look to check a borderline low-ray veto.

The prior actor and 3 cm inward low-source registration stress are frozen.
At a range-track-directed low camera look, depth projects only from the
actor's current estimated pose. A clean near-surface circle can release a
low-only corridor veto with an added 1 cm depth reserve. Missing/ambiguous
depth preserves the original veto. The gimbal aim has no simulated actuation
time, and the ideal eye rig has no selected-body self occlusion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import novel_range_object as tracking
from run import ROOT, sha
from run_spec_planar_route_v202p import StatusGrid
import run_low_prop_sensor_join_v206p as v206
import run_low_prop_role_fault_v212p as v212
import run_low_prop_capture_age_v227p as v227
import run_low_prop_local_corridor_v230p as v230
import run_low_prop_capture_corridor_v229p as v229
import run_low_prop_aligned_capture_v232p as v232
import run_low_prop_held_pose_v224p as v224


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-depth-crosscheck-v236p"
LOW_BIAS_XY_M = np.array([0., -.03])
CAMERA_Z_M = .20
DEPTH_RANGE_M = (.20, 1.50)
DEPTH_NOISE_M = .01
DEPTH_HOLE_FRACTION = .15
DEPTH_CLEAR_RESERVE_M = .01
FOVY_DEG = 70.
WIDTH, HEIGHT = 320, 240


def rotation_toward(estimated_xy, track_center):
    forward = np.asarray(track_center) - np.asarray(estimated_xy)
    forward /= np.linalg.norm(forward)
    right = np.array([forward[1], -forward[0], 0.])
    up = np.array([0., 0., 1.])
    back = np.cross(right, up)
    rotation = np.stack((right, up, back), axis=1)
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return rotation, xyzw[[3, 0, 1, 2]]


def depth_center(packet, estimated_xy, low_center, radius):
    """Actor-side near-surface fit; no simulator object ID or true pose."""
    depth = packet["measured_depth_m"]
    height, width = depth.shape
    focal = height / (2 * math.tan(math.radians(FOVY_DEG) / 2))
    rows, cols = np.mgrid[4:height:8, 4:width:8]
    values = depth[rows, cols]
    valid = np.isfinite(values) & (values >= DEPTH_RANGE_M[0]) & (
        values <= DEPTH_RANGE_M[1])
    rays = np.stack(((cols - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(values)), axis=-1)[valid]
    origin = np.r_[estimated_xy, CAMERA_Z_M]
    points = origin + (rays * values[valid, None]) @ packet["rotation"].T
    low_center = np.asarray(low_center)
    radial = np.linalg.norm(points[:, :2] - origin[:2], axis=1)
    expected_front = np.linalg.norm(low_center - origin[:2]) - radius
    chosen = (np.linalg.norm(points[:, :2] - low_center, axis=1) < .45) & (
        points[:, 2] > .04) & (points[:, 2] < .18) & (
        radial < expected_front + .04)
    candidate = points[chosen, :2]
    if len(candidate) < 12:
        return {"status": "too_few_depth_points", "points": len(candidate)}
    fit = least_squares(
        lambda center: np.linalg.norm(candidate - center, axis=1) - radius,
        low_center, bounds=(low_center - .08, low_center + .08),
        loss="soft_l1", f_scale=.02)
    center = fit.x
    residual = float(np.median(np.abs(np.linalg.norm(candidate - center,
                                                      axis=1) - radius)))
    if not fit.success or residual > .012:
        return {"status": "ambiguous_depth_shape", "points": len(candidate),
                "median_residual_m": residual}
    return {"status": "depth_confirmed_circle", "points": len(candidate),
            "median_residual_m": residual, "center_xy_m": center.tolist(),
            "low_depth_center_difference_m": float(np.linalg.norm(
                center - low_center))}


def camera_timeline(captures, decisions, case_id):
    frames = []
    for capture, decision in zip(captures, decisions):
        panel = np.zeros((2 * HEIGHT, WIDTH, 3), dtype=np.uint8)
        source_index = decision["depth_source_capture_index"]
        delivered = captures[source_index] if source_index is not None else None
        if delivered is not None and delivered["packet"] is not None:
            with np.load(ROOT / delivered["packet"]["path"],
                         allow_pickle=False) as packet:
                panel[:HEIGHT] = packet["rgb"]
                depth = packet["measured_depth_m"]
            valid = np.isfinite(depth)
            scaled = np.zeros(depth.shape, np.uint8)
            scaled[valid] = np.clip((depth[valid] - DEPTH_RANGE_M[0]) /
                                    (DEPTH_RANGE_M[1] - DEPTH_RANGE_M[0]) *
                                    255, 0, 255).astype(np.uint8)
            color = cv2.cvtColor(cv2.applyColorMap(
                255 - scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            color[~valid] = 0
            panel[HEIGHT:] = color
        frame = Image.fromarray(panel)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, WIDTH, 26), fill="#101619")
        draw.text((7, 8), f"{case_id}  now={capture['time_s']:.1f}s "
                  f"view={delivered['time_s']:.1f}s" if delivered is not None
                  else f"{case_id}  now={capture['time_s']:.1f}s no view",
                  fill="white")
        draw.rectangle((0, HEIGHT, WIDTH, HEIGHT + 26), fill="#101619")
        draw.text((7, HEIGHT + 8), decision["decision"], fill="white")
        frames.extend((np.asarray(frame), np.asarray(frame)))
    path = OUT / "inward_bias_depth" / case_id / "current-low-rgbd-timeline.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=5, codec="libx264", quality=7)
    return {"path": str(path), "sha256": sha(path), "frames": len(frames),
            "decision_interval_s": .4}


def one(case, prior, vision, exemplar, *, depth_delay_steps=0,
        depth_age_limit_s=None):
    if depth_delay_steps < 0:
        raise ValueError("Depth delay must be nonnegative")
    base_plane, base_plan = v206.plane, StatusGrid.plan
    base_update, base_v230_one = tracking.NovelRangeObject.update, v230.one
    old_v232_out = v232.OUT
    state = {"estimated_xy": None, "last_track_center": None,
             "last_track_radius": None, "depth_history": [],
             "renderer": None, "model": None}
    captures, decisions, projections, vetoes = [], [], [], []
    branch = OUT / "inward_bias_depth"

    def biased_update(tracker, ranges, actor_capture_pose, capture_time_s):
        injected = np.asarray(actor_capture_pose) + LOW_BIAS_XY_M
        result = base_update(tracker, ranges, injected, capture_time_s)
        projections.append({"capture_time_s": capture_time_s,
                            "actor_capture_pose_xy_m": np.asarray(
                                actor_capture_pose).tolist(),
                            "low_projection_pose_xy_m": injected.tolist(),
                            "track_status": result["status"]})
        return result

    def observed_plane(model, data, actual_xy, height_m, beams, minimum_m, rng):
        values, statuses = base_plane(
            model, data, actual_xy, height_m, beams, minimum_m, rng)
        if height_m != .10 or beams != 72:
            return values, statuses
        index = len(captures)
        previous_center = state["last_track_center"]
        previous_estimate = state["estimated_xy"]
        record = {"decision_index": index, "time_s": round(index * .4, 4),
                  "status": "no_prior_track", "packet": None}
        captured_depth = None
        if previous_center is not None and previous_estimate is not None and (
            np.linalg.norm(np.asarray(previous_center) - previous_estimate) <= 1.5):
            if state["renderer"] is None:
                state["renderer"] = mujoco.Renderer(
                    model, height=HEIGHT, width=WIDTH)
                state["model"] = model
            elif state["model"] is not model:
                raise AssertionError("Depth renderer model changed")
            camera = model.camera("owner_view").id
            original_pos = model.cam_pos[camera].copy()
            original_quat = model.cam_quat[camera].copy()
            rotation, quaternion = rotation_toward(
                previous_estimate, previous_center)
            model.cam_pos[camera] = [actual_xy[0], actual_xy[1], CAMERA_Z_M]
            model.cam_quat[camera] = quaternion
            mujoco.mj_forward(model, data)
            renderer = state["renderer"]
            renderer.update_scene(data, camera="owner_view")
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="owner_view")
            ideal = renderer.render().copy()
            renderer.disable_depth_rendering()
            model.cam_pos[camera] = original_pos
            model.cam_quat[camera] = original_quat
            mujoco.mj_forward(model, data)
            noise = np.random.default_rng(236000 + index)
            measured = ideal + noise.normal(0., DEPTH_NOISE_M,
                                            ideal.shape).astype(ideal.dtype)
            measured[(measured < DEPTH_RANGE_M[0]) | (
                measured > DEPTH_RANGE_M[1]) | (
                noise.random(measured.shape) < DEPTH_HOLE_FRACTION)] = np.nan
            path = branch / "depth-packets" / case["id"] / f"look-{index:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, rgb=rgb, measured_depth_m=measured,
                                rotation=rotation, commanded_camera_quat=quaternion,
                                previous_actor_estimated_xy_m=previous_estimate,
                                previous_actor_track_center_xy_m=previous_center,
                                decision_index=index, capture_time_s=index * .4)
            record = {"decision_index": index, "time_s": round(index * .4, 4),
                      "status": "current_rgbd_look", "packet": {
                          "path": str(path.relative_to(ROOT)),
                          "sha256": sha(path)},
                      "previous_actor_track_center_xy_m": previous_center,
                      "previous_actor_estimated_xy_m": previous_estimate.tolist()}
            captured_depth = {"measured_depth_m": measured,
                              "rotation": rotation,
                              "actor_capture_pose_xy_m": None}
        captures.append(record)
        state["depth_history"].append(captured_depth)
        return values, statuses

    def depth_checked_one(case, prior, vision, exemplar):
        old_v227_out = v227.OUT
        raw_plan = StatusGrid.plan

        def depth_checked_plan(grid, origin_xy, goal_xy):
            result = raw_plan(grid, origin_xy, goal_xy)
            requested, _, path = result
            index = len(grid.track_history) - 1
            if index != len(captures) - 1:
                raise AssertionError("Depth packet lost decision alignment")
            track = grid.track_history[-1]
            center, radius = track["center_xy_m"], track["radius_m"]
            current_depth = state["depth_history"][index]
            if current_depth is not None:
                current_depth["actor_capture_pose_xy_m"] = np.asarray(
                    origin_xy).copy()
                captures[index]["actor_capture_pose_xy_m"] = np.asarray(
                    origin_xy).tolist()
            source_index = index - depth_delay_steps
            delivered_depth = (state["depth_history"][source_index]
                               if source_index >= 0 else None)
            age_s = (round(depth_delay_steps * .4, 4)
                     if delivered_depth is not None else None)
            depth_fresh = (delivered_depth is not None and
                           (depth_age_limit_s is None or
                            age_s <= depth_age_limit_s))
            row = {"scan_index": index, "time_s": round(index * .4, 4),
                   "low_track_status": track["status"],
                   "depth_capture_status": captures[index]["status"],
                   "depth_source_capture_index": (source_index if
                       delivered_depth is not None else None),
                   "depth_age_s": age_s,
                   "depth_age_status": ("missing" if delivered_depth is None
                                        else "fresh" if depth_fresh else "stale"),
                   "decision": result[1], "depth_fit": None,
                   "low_minimum_m": None, "depth_minimum_m": None,
                   "required_m": None}
            if requested is not None and path and center is not None and (
                radius is not None and track["status"] in
                ("tracked", "shape_confirmed")):
                cells = [grid.xy(cell) for cell in path if
                         np.linalg.norm(grid.xy(cell) - origin_xy) <=
                         v230.LOOKAHEAD_M]
                if cells:
                    row["lookahead_cells_xy_m"] = [
                        np.asarray(point).tolist() for point in cells]
                    minimum = min(float(np.linalg.norm(
                        point - np.asarray(center))) for point in cells)
                    required = (v229.BODY_RADIUS_M + float(radius) +
                                v229.REQUIRED_GAP_M +
                                v229.HISTORICAL_STOP_PATH_M)
                    row["low_minimum_m"], row["required_m"] = minimum, required
                    if minimum < required:
                        if depth_fresh:
                            fit = depth_center(
                                delivered_depth,
                                delivered_depth["actor_capture_pose_xy_m"],
                                center, float(radius))
                            row["depth_fit"] = fit
                            if fit["status"] == "depth_confirmed_circle":
                                depth_minimum = min(float(np.linalg.norm(
                                    point - fit["center_xy_m"])) for point in cells)
                                row["depth_minimum_m"] = depth_minimum
                                if depth_minimum >= required + DEPTH_CLEAR_RESERVE_M:
                                    row["decision"] = "depth_clears_low_veto"
                                else:
                                    row["decision"] = "ask_owner_clear_corridor"
                            else:
                                row["decision"] = "ask_owner_clear_corridor"
                        else:
                            row["decision"] = "ask_owner_clear_corridor"
                        if row["decision"] == "ask_owner_clear_corridor":
                            result = (None, "ask_owner_clear_corridor", [])
                            vetoes.append({"scan_index": index,
                                           "time_s": row["time_s"],
                                           "track_status": track["status"],
                                           "track_center_xy_m": center,
                                           "track_radius_m": radius,
                                           "minimum_lookahead_center_separation_m": minimum,
                                           "required_separation_m": required,
                                           "lookahead_m": v230.LOOKAHEAD_M,
                                           "decision": result[1]})
            state["estimated_xy"] = np.asarray(origin_xy).copy()
            state["last_track_center"] = center
            state["last_track_radius"] = radius
            decisions.append(row)
            return result

        StatusGrid.plan = depth_checked_plan
        v227.OUT = v230.OUT
        try:
            result = v227.one(case, "capture_age_gate", prior, vision, exemplar)
        finally:
            StatusGrid.plan = raw_plan
            v227.OUT = old_v227_out
        result["low_corridor_vetoes"] = vetoes
        result["low_depth_decisions"] = decisions
        result["low_depth_captures"] = captures
        result["capture_age_profile"] = "capture_age_and_depth_checked_corridor"
        return result

    tracking.NovelRangeObject.update = biased_update
    v206.plane = observed_plane
    v230.one = depth_checked_one
    v232.OUT = OUT
    try:
        result = v232.one(case, "aligned_age_local", prior, vision, exemplar)
    finally:
        tracking.NovelRangeObject.update = base_update
        v206.plane = base_plane
        v230.one = base_v230_one
        v232.OUT = old_v232_out
        if state["renderer"] is not None:
            state["renderer"].close()
    result["low_projection_bias_xy_m"] = LOW_BIAS_XY_M.tolist()
    result["low_projection_events"] = projections
    result["depth_delivery_delay_steps"] = depth_delay_steps
    result["depth_age_limit_s"] = depth_age_limit_s
    result["low_depth_timeline_video"] = camera_timeline(
        captures, decisions, case["id"])
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    prior = np.load(v206.dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_window, old_margin = v212.WINDOW_S, v206.route.MARGIN_M
    v212.WINDOW_S = (999., 1000.)
    v206.route.MARGIN_M = .15
    vision = v206.dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v206.dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case in v224.CASES:
            result = one(case, prior, vision, exemplar)
            episodes.append(result)
            print(case["id"], result["route_state"], result["outcome"],
                  "depth_release", sum(d["decision"] == "depth_clears_low_veto"
                                       for d in result["low_depth_decisions"]),
                  "gap", round(result["route_episode"][
                      "min_true_prop_gap_m"], 3), flush=True)
    finally:
        v212.WINDOW_S = old_window
        v206.route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-depth-crosscheck.v1",
        "source_sha256": sha(SOURCE),
        "v232_source_sha256": sha(Path(v232.__file__)),
        "v230_source_sha256": sha(Path(v230.__file__)),
        "inward_bias_report_sha256": sha(
            ROOT / "artifacts/ape-robocasa-low-prop-capture-pose-bias-v234p/report.json"),
        "current_design_status": pointer["status"],
        "low_projection_bias_xy_m": LOW_BIAS_XY_M.tolist(),
        "depth_view": {"height_m": CAMERA_Z_M, "width": WIDTH,
                       "height": HEIGHT, "fovy_deg": FOVY_DEG,
                       "valid_range_m": DEPTH_RANGE_M,
                       "gaussian_noise_std_m": DEPTH_NOISE_M,
                       "random_hole_fraction": DEPTH_HOLE_FRACTION,
                       "clearance_reserve_m": DEPTH_CLEAR_RESERVE_M},
        "changed_factor": "At each decision after a prior low track, render "
                          "a current actor-aimed low RGB-D view. At a low-only "
                          "local veto, fit near-surface depth points near the "
                          "low track from actor-estimated current pose. Release "
                          "only if fit is clean and the depth center clears "
                          "the same next-60cm path with 1cm extra reserve; "
                          "otherwise retain the low veto.",
        "scope": "Paired with V234p's inward 3cm low-projection bias on "
                 "three near-fast complete owner episodes. Current depth "
                 "has 1cm Gaussian noise, 15% holes and a 20–150cm window. "
                 "The track-directed camera has instantaneous ideal yaw, "
                 "no selected-body mount/self mask, and no native stop. "
                 "This is a sensor-fusion ceiling, not hardware proof.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
