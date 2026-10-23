"""First owner-approach loop inside a photographed indoor mesh.

The owner can be a simple moving proxy or a rigidly moved photogrammetry scan.
The actor receives rendered RGB-D, two-height range, a human teaching click,
optional odometry, and accepted/rejected motion feedback. Simulator geometry
and owner trajectory remain evaluator-only. This is an integration step, not
a body or realistic human-motion qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/realistic_people"))
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/sensors"))
from fan_view_route import FanViewRoute  # noqa: E402
from online_owner_actor import OnlineOwnerActor  # noqa: E402
from measure_real_scan_visual_odometry import feature_packet, visual_transform  # noqa: E402
from real_capture_dynamic_scene import (GaussianCaptureScene, ScannedSittingPerson,
                                        TexturedCaptureScene)  # noqa: E402
from rgbd_range_projection import dual_view_fan, projected_depth_fan  # noqa: E402
from mounted_owner_rig import MountedOwnerRig  # noqa: E402
from mounted_low_scan import MOUNT_FORWARD_M, sample_front_low_scan  # noqa: E402
from native_shaped_pose import NativeShapedPose, SOURCES as NATIVE_SOURCES  # noqa: E402


SCENES = {
    "serving": {
        "trace": ROOT / "assets/real_indoor_scenes/scansceneverse/routes/serving_south_2m/trace.json",
        "name": "The Serving Room", "start_xz": (0., -2.), "heading_deg": 180.,
        "owner_xz": (0., -4.1), "owner_lateral": (1., 0.),
        "person_facing_deg": 0.,
    },
    "living": {
        "trace": ROOT / "assets/real_indoor_scenes/scansceneverse/routes/living_east_1m/trace.json",
        "name": "Living room", "start_xz": (-13., -7.), "heading_deg": 90.,
        "owner_xz": (-10.8, -7.), "owner_lateral": (0., 1.),
        "person_facing_deg": 270.,
    },
}
PLACEMENT_SCREEN = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/screen.json"
EXTRA_PLACEMENT_SCREENS = sorted(PLACEMENT_SCREEN.parent.glob("*_owner_screen/screen.json"))
NEW_CAPTURE_PLACEMENT_SCREENS = sorted((ROOT /
    "assets/real_indoor_scenes/scansceneverse/new_capture_owner_20261014").glob(
        "*/owner_placement_screen/screen.json"))
FRESH_HOLDOUT_PLACEMENT_SCREENS = sorted((ROOT /
    "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014").glob(
        "*/owner_placement_screen/screen.json"))
HABITAT_PLACEMENT_SCREENS = sorted((ROOT /
    "assets/real_indoor_scenes/habitat_gs/owner_placement_screen/v3").glob(
        "scene*/screen.json"))
HABITAT_TEST_PLACEMENT_SCREENS = []
for room in sorted((ROOT / "assets/real_indoor_scenes/habitat_test_scenes").iterdir()):
    if not room.is_dir():
        continue
    screens = list((room / "owner_placement_screen").glob("**/screen.json"))
    if screens:
        # Keep the deepest retained revision for a room; older rejected
        # appearance screens stay on disk but never enter actor episodes.
        HABITAT_TEST_PLACEMENT_SCREENS.append(max(
            screens, key=lambda path: (len(path.relative_to(room).parts), str(path))))
for placement_screen in (PLACEMENT_SCREEN, *EXTRA_PLACEMENT_SCREENS,
                         *NEW_CAPTURE_PLACEMENT_SCREENS,
                         *FRESH_HOLDOUT_PLACEMENT_SCREENS,
                         *HABITAT_PLACEMENT_SCREENS,
                         *HABITAT_TEST_PLACEMENT_SCREENS):
    if not placement_screen.is_file():
        continue
    placement_data = json.loads(placement_screen.read_text())
    placements = placement_data.get("scenes", [placement_data])
    for placement in placements:
        if placement["status"] != "provisional_supported_visible":
            continue
        key = "capture_" + Path(placement["source_trace"]).parent.name
        SCENES[key] = {
            "trace": ROOT / placement["source_trace"],
            "name": placement["name"],
            "start_xz": tuple(placement["start_xz"]),
            "heading_deg": placement["start_heading_deg"],
            "owner_xz": tuple(placement["owner_xz"]),
            "owner_lateral": tuple(placement["owner_lateral_xz"]),
            "person_facing_deg": placement["owner_facing_deg"],
            "click_xy": tuple(placement["click_xy_at_640x360"]),
            "click_source": "evaluator-selected click from visible scanned-person pixels",
            "episode_tag": "_placement_screen_v1",
            "placement_screen": str(placement_screen),
            "source_family": ("Habitat-GS" if placement["scene_id"].startswith(
                "habitat_gs/") else "Habitat test scan" if placement["scene_id"].startswith(
                "habitat_test_scenes/") else "ScanSceneverse"),
        }
MODEL_ROOT = ROOT / ".cache/ape-high-level/models"
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
DT_S = .2


def owner_pose(t: float, config: dict) -> tuple[float, float]:
    # A small lateral movement forces fresh bearing and appearance updates.
    shift = .22 * math.sin(.55 * t)
    return tuple(np.asarray(config["owner_xz"]) +
                 shift * np.asarray(config["owner_lateral"]))


def angle_error(a: float, b: float) -> float:
    return (a - b + 180.) % 360. - 180.


def odometry_delta(previous: dict, current: dict, k: np.ndarray,
                   matcher: cv2.BFMatcher) -> tuple[np.ndarray | None, dict]:
    forward, a = visual_transform(previous, current, k, matcher)
    reverse, b = visual_transform(current, previous, k, matcher)
    evidence = {"forward": a, "reverse": b, "accepted": False}
    if forward is None or reverse is None:
        return None, evidence
    closure = reverse @ forward
    evidence["reciprocal_translation_m"] = float(np.linalg.norm(closure[:3, 3]))
    evidence["reciprocal_rotation_deg"] = float(math.degrees(math.acos(np.clip(
        (np.trace(closure[:3, :3]) - 1) / 2, -1, 1))))
    evidence["accepted"] = bool(a["inliers"] >= 30 and b["inliers"] >= 30 and
        a["inlier_ratio"] >= .25 and b["inlier_ratio"] >= .25 and
        a["median_reprojection_px"] <= 2.5 and b["median_reprojection_px"] <= 2.5 and
        evidence["reciprocal_translation_m"] <= .08 and
        evidence["reciprocal_rotation_deg"] <= 5.)
    return (np.linalg.inv(forward) if evidence["accepted"] else None), evidence


def masked_feature_packet(rgb: np.ndarray, depth: np.ndarray, orb: cv2.ORB,
                          social: dict) -> tuple[dict, int]:
    """Exclude the actor-selected moving person from static-scene odometry."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.full(gray.shape, 255, np.uint8)
    chosen = social.get("chosen_proposal")
    proposals = social.get("proposals", [])
    masked_area = 0
    if chosen is not None and chosen < len(proposals):
        x0, y0, x1, y1 = proposals[chosen]["box_xyxy"]
        x0, y0 = max(0, int(x0) - 12), max(0, int(y0) - 12)
        x1, y1 = min(gray.shape[1], int(x1) + 12), min(gray.shape[0], int(y1) + 12)
        mask[y0:y1, x0:x1] = 0
        masked_area = (x1 - x0) * (y1 - y0)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return {"keypoints": keypoints, "descriptors": descriptors,
            "depth": depth}, masked_area


def near_range_box_association(social: dict, ranges: np.ndarray,
                               angles_deg: np.ndarray, *, width: int,
                               height: int, focal: float) -> dict:
    """Project short horizontal range returns into the chosen person box.

    A box overlap is only an unresolved person-surface candidate, not a
    semantic range label. All rays remain obstacles for route planning.
    """
    chosen = social.get("chosen_proposal")
    proposals = social.get("proposals", [])
    if chosen is not None and chosen < len(proposals):
        x0, y0, x1, y1 = proposals[chosen]["box_xyxy"]
        box_source = "full_person_detector"
    elif social.get("track", {}).get("status") == "tracked":
        x0, y0, x1, y1 = social["track"]["box_xyxy"]
        box_source = "fresh_rgbd_flow"
    else:
        return {"status": "person_box_unavailable", "near_hits": []}
    hits = []
    for level, plane_height in enumerate((.1, .5)):
        for index, offset in enumerate(angles_deg):
            distance = float(ranges[level, index])
            if not (np.isfinite(distance) and distance < 1.25 and
                    abs(float(offset)) <= 20.):
                continue
            radians = math.radians(float(offset))
            u = width / 2 - focal * math.tan(radians)
            v = height / 2 - focal * (plane_height - .42) / max(
                .1, distance * math.cos(radians))
            hits.append({"height_m": plane_height, "angle_deg": float(offset),
                         "range_m": distance, "pixel_uv": [u, v],
                         "inside_person_box": bool(x0 <= u <= x1 and y0 <= v <= y1)})
    return {"status": "projected", "box_source": box_source,
            "person_box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
            "near_hits": hits,
            "any_in_person_box": any(h["inside_person_box"] for h in hits)}


def add_person_mask_association(association: dict, rgb: np.ndarray,
                                processor: object, model: object,
                                device: str) -> dict:
    """Prompt SAM 2 from the actor's box; classify only visible ray pixels."""
    import torch

    if association["status"] != "projected" or not association["near_hits"]:
        return association
    started = time.perf_counter()
    inputs = processor(images=Image.fromarray(rgb),
                       input_boxes=[[association["person_box_xyxy"]]],
                       return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    masks = processor.post_process_masks(
        output.pred_masks.cpu(), inputs["original_sizes"])[0][0]
    best = int(output.iou_scores[0, 0].argmax())
    mask = masks[best].numpy() > 0
    score = float(output.iou_scores[0, 0, best])
    hits_in_mask = []
    all_projected = True
    for hit in association["near_hits"]:
        u, v = (int(round(float(x))) for x in hit["pixel_uv"])
        if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
            inside = bool(mask[v, u])
            hit["inside_person_mask"] = inside
            hits_in_mask.append(inside)
        else:
            hit["inside_person_mask"] = None
            all_projected = False
    association.update(mask_prompt="actor_selected_person_box",
                       mask_predicted_iou=score,
                       mask_area_px=int(mask.sum()),
                       mask_inference_wall_s=time.perf_counter() - started,
                       all_hits_projected=all_projected,
                       any_in_person_mask=any(hits_in_mask),
                       mask_usable=bool(score >= .90 and all_projected))
    return association


def run(scene_id: str, pose_mode: str, *, steps: int, device: str,
        native_source: str = "varied150-forward-seed144",
        person_style: str = "proxy",
        range_mode: str = "two_plane_range",
        gaze_route_arbitration: bool = False,
        pose_crosscheck: bool = False,
        person_masked_odometry: bool = False,
        simulated_inference_delay_s: float = 0.,
        fast_revalidate_during_inference: bool = False,
        spatial_range_association: bool = False,
        sam2_range_association: bool = False,
        far_conversation_viewpoint: bool = False,
        depth_separated_overlap: bool = False,
        receding_owner_target: bool = False,
        mount_screen: Path | None = None,
        camera_free_prefix: bool = True,
        mounted_low_scan: bool = False,
        near_body_monitor=None,
        owner_option_step=None,
        save_raw_frames: bool = True,
        output_dir: Path | None = None) -> dict:
    if person_masked_odometry and pose_mode not in ("rgbd_yaw", "rgbd_native_fallback"):
        raise ValueError("person mask requires visual RGB-D odometry")
    if pose_mode not in ("command", "rgbd_yaw", "native_shaped",
                         "rgbd_native_fallback"):
        raise ValueError("unknown pose mode")
    if simulated_inference_delay_s < 0:
        raise ValueError("simulated inference delay must be nonnegative")
    if fast_revalidate_during_inference and not simulated_inference_delay_s:
        raise ValueError("fast revalidation requires a simulated inference delay")
    if spatial_range_association and sam2_range_association:
        raise ValueError("choose either box or SAM 2 range association")
    if range_mode not in ("two_plane_range", "rgbd_projection",
                          "dual_rgbd_projection", "mounted_body_rgbd"):
        raise ValueError("unknown range mode")
    if range_mode in ("dual_rgbd_projection", "mounted_body_rgbd") and \
            fast_revalidate_during_inference:
        raise ValueError("dual-view fast revalidation is not implemented")
    if range_mode == "mounted_body_rgbd" and (person_style != "scanned" or
            spatial_range_association or sam2_range_association):
        raise ValueError("mounted RGB-D uses screened scanned owner and body-frame camera free rays")
    if range_mode != "mounted_body_rgbd" and (mount_screen is not None or
                                               not camera_free_prefix or
                                               mounted_low_scan):
        raise ValueError("mounted screen and camera free-prefix flag require mounted RGB-D")
    cv2.setRNGSeed(0)
    config = SCENES[scene_id]
    trace_path = config["trace"]
    source = json.loads(trace_path.read_text())
    human_path = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
    scanned_person = ScannedSittingPerson(human_path) if person_style == "scanned" else None
    if source.get("source_type") == "gaussian_splat_mesh":
        scene = GaussianCaptureScene(
            ROOT / source["source_mesh"], ROOT / source["source_splat"],
            floor_y=source["floor_y_1pct_m"],
            scanned_person=scanned_person,
            person_facing_deg=config["person_facing_deg"])
    else:
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
                                     floor_y=source["floor_y_1pct_m"],
                                     scale=source.get("scale_to_metres", 1.),
                                     source_z_up=source.get("source_z_up", False),
                                     scanned_person=scanned_person,
                                     person_facing_deg=config["person_facing_deg"])
    rig = (MountedOwnerRig(scene_id, screen_path=mount_screen)
           if range_mode == "mounted_body_rgbd" else None)
    suffix = ("_arbitrated" if gaze_route_arbitration else "") + \
             ("_crosschecked" if pose_crosscheck else "") + \
             ("_personmasked" if person_masked_odometry else "") + \
             ("_delay850ms" if math.isclose(simulated_inference_delay_s, .85) else
              f"_delay{round(1000 * simulated_inference_delay_s)}ms" if simulated_inference_delay_s else "") + \
             ("_fastrevalidate" if fast_revalidate_during_inference else "") + \
             ("_rangeprojected" if spatial_range_association else "") + \
             ("_rangesam2" if sam2_range_association else "") + \
             ("_farconverse" if far_conversation_viewpoint else "") + \
             ("_depthlayers" if depth_separated_overlap else "") + \
             ("_recedingowner" if receding_owner_target else "") + \
             ("_" + native_source if pose_mode in
              ("native_shaped", "rgbd_native_fallback") else "") + \
             ("_scanned" if person_style == "scanned" else "")
    suffix += ("_rgbdview" if range_mode == "rgbd_projection" else
               "_dualrgbdview" if range_mode == "dual_rgbd_projection" else
               "_mountedbodyrgbd" if rig is not None else "")
    if mounted_low_scan:
        suffix += "_lowscan"
    output = (output_dir if output_dir is not None else trace_path.parent /
              f"dynamic_owner_{pose_mode}{suffix}{config.get('episode_tag', '')}_episode")
    output.mkdir(parents=True, exist_ok=True)
    start_xz = np.asarray(config["start_xz"], dtype=float)
    actual = start_xz.copy()
    actual_heading = float(config["heading_deg"])
    actor = np.zeros(2, dtype=float)
    actor_heading = float(config["heading_deg"])
    command_pose = np.zeros(2, dtype=float)
    command_heading = float(config["heading_deg"])
    vision = OnlineOwnerActor(MODEL_ROOT / "grounding-dino-tiny",
                              MODEL_ROOT / "dinov2-small",
                              click_xy=rig.click_xy if rig is not None else
                                       config.get("click_xy", (320, 160)),
                              device=device,
                              depth_separated_overlap=depth_separated_overlap)
    planner = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1))
    orb = (cv2.ORB_create(nfeatures=2500) if
           pose_mode in ("rgbd_yaw", "rgbd_native_fallback") else None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING) if orb is not None else None
    native_pose = (NativeShapedPose(native_source) if
                   pose_mode in ("native_shaped", "rgbd_native_fallback") else None)
    pending_native_packet = None
    fallback_distance_m = 0.
    fallback_frames = 0
    height, width = 360, 640
    focal = height / (2 * math.tan(math.radians(
        rig.vfov_deg if rig is not None else 75.) / 2))
    hfov = 2 * math.degrees(math.atan(width / (2 * focal)))
    k = np.array([[focal, 0., (width - 1) / 2],
                  [0., focal, (height - 1) / 2], [0., 0., 1.]], dtype=np.float64)
    previous_packet = None
    previous_action = {"yaw_delta_deg": 0., "forward_step_m": 0., "accepted": True}
    previous_turn_only = False
    segment_processor = None
    segment_model = None
    frame_rows = []
    sim_time_s = 0.
    writer = imageio.get_writer(str(output / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    down_writer = (imageio.get_writer(str(output / "downward_view.mp4"), fps=5,
        codec="libx264", quality=7, macro_block_size=8)
        if range_mode in ("dual_rgbd_projection", "mounted_body_rgbd") else None)
    monitor_writer = (imageio.get_writer(str(output / "inference_monitor.mp4"),
        fps=5, codec="libx264", quality=7, macro_block_size=8)
        if fast_revalidate_during_inference else None)
    try:
        for frame_index in range(steps):
            t = sim_time_s
            actual_at_capture = actual.copy()
            person_xz = owner_pose(t, config)
            monitor_mark = None
            if near_body_monitor is not None:
                monitor_mark = near_body_monitor.place_stationary_person(
                    np.asarray(person_xz, float))
            eye = np.array([actual[0], scene.floor_y + .42, actual[1]])
            direction = np.array([math.sin(math.radians(actual_heading)), 0.,
                                  math.cos(math.radians(actual_heading))])
            down_rgb = down_depth = None
            if rig is not None:
                mounted_packet = rig.observe(
                    scene, actual, actual_heading, person_xz, ANGLES)
                rgb, depth = mounted_packet["rgb"], mounted_packet["depth"]
                down_rgb, down_depth = (mounted_packet["lower_rgb"],
                                        mounted_packet["lower_depth"])
            else:
                rgb, depth = scene.observe(eye, direction, owner_xz=person_xz)
            low_scan_packet = (sample_front_low_scan(
                scene, actual, actual_heading, person_xz, ANGLES,
                scene_id=scene_id, frame_index=frame_index)
                if mounted_low_scan else None)
            if down_writer is not None and rig is None:
                tilt = math.radians(45.)
                down_direction = direction * math.cos(tilt) + \
                                 np.array([0., -math.sin(tilt), 0.])
                down_rgb, down_depth = scene.observe(
                    eye, down_direction, owner_xz=person_xz)
            if monitor_writer is not None:
                monitor_frame = rgb.copy()
                cv2.putText(monitor_frame, f"capture {t:.2f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                monitor_writer.append_data(monitor_frame)
            ranges = (mounted_packet["ranges"] if rig is not None else
                      scene.range_fan(eye, actual_heading, owner_xz=person_xz,
                                     angles_deg=ANGLES) if range_mode == "two_plane_range"
                      else dual_view_fan(depth, down_depth, ANGLES)
                      if range_mode == "dual_rgbd_projection" else
                      projected_depth_fan(depth, ANGLES))
            owner_start = time.perf_counter()
            social = vision.observe(rgb, depth, ranges, ANGLES, time_s=t,
                                    fov_deg=hfov,
                                    ego_yaw_delta_deg=-previous_action["yaw_delta_deg"])
            if rig is not None:
                social = rig.calibrate_owner(social)
            inference_wall_s = time.perf_counter() - owner_start
            decision_time_s = t + simulated_inference_delay_s
            person_at_action_xz = owner_pose(decision_time_s, config)
            # This packet is evaluator evidence for the stale-observation
            # comparison. It is not given to the actor as a fast monitor.
            range_at_action = scene.range_fan(
                eye, actual_heading, owner_xz=person_at_action_xz,
                angles_deg=ANGLES) if simulated_inference_delay_s else ranges
            decision_social = social
            decision_ranges = ranges
            decision_rgb = rgb
            fast_updates = []
            if fast_revalidate_during_inference:
                seed = vision.seed_fast_flow(rgb, depth, social,
                    time_s=t, fov_deg=hfov)
                if seed["status"] != "seeded":
                    decision_social = {"intent": "hold_reobserve",
                        "reason": "fast_revalidation_could_not_seed",
                        "seed": seed, "time_s": decision_time_s}
                else:
                    offsets = []
                    tick = .2
                    while tick < simulated_inference_delay_s - 1e-6:
                        offsets.append(tick)
                        tick += .2
                    offsets.append(simulated_inference_delay_s)
                    for tick_index, offset in enumerate(offsets):
                        tick_time = t + offset
                        tick_owner = owner_pose(tick_time, config)
                        tick_rgb, tick_depth = scene.observe(
                            eye, direction, owner_xz=tick_owner)
                        tick_ranges = (scene.range_fan(
                            eye, actual_heading, owner_xz=tick_owner,
                            angles_deg=ANGLES) if range_mode == "two_plane_range"
                            else projected_depth_fan(tick_depth, ANGLES))
                        started_fast = time.perf_counter()
                        fast = vision.observe_fast(
                            tick_rgb, tick_depth, tick_ranges, ANGLES,
                            time_s=tick_time, fov_deg=hfov)
                        fast_wall_s = time.perf_counter() - started_fast
                        fast_rgb_name = f"fast_rgb_{frame_index:03d}_{tick_index:02d}.png"
                        fast_depth_name = f"fast_depth_{frame_index:03d}_{tick_index:02d}.png"
                        Image.fromarray(tick_rgb).save(output / fast_rgb_name)
                        Image.fromarray(np.where(np.isfinite(tick_depth),
                            np.clip(tick_depth * 1000, 0, 65535), 0).astype(
                                np.uint16)).save(output / fast_depth_name)
                        monitor_frame = tick_rgb.copy()
                        cv2.putText(monitor_frame,
                            f"monitor {tick_time:.2f}s {fast['intent']}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
                        monitor_writer.append_data(monitor_frame)
                        fast_updates.append({"time_s": tick_time,
                            "actor": fast, "wall_s": fast_wall_s,
                            "rgb": fast_rgb_name, "depth": fast_depth_name,
                            "nearest_forward_range_m": float(np.nanmin(
                                tick_ranges[:, np.abs(ANGLES) <= 20]))})
                        decision_social = fast
                        decision_ranges = tick_ranges
                        decision_rgb = tick_rgb
                        if fast["intent"] in ("hold_reobserve", "hold_ask_show_again"):
                            break
            pose_evidence = None
            pose_healthy = True
            pose_failure_reason = None
            masked_area = 0
            native_packet = pending_native_packet
            pending_native_packet = None
            if pose_mode == "native_shaped" and native_packet is not None:
                pose_evidence = native_packet
                pose_healthy = bool(native_packet["accepted"])
                if pose_healthy:
                    actor += np.asarray(native_packet["measured_delta_world_m"])
                else:
                    pose_failure_reason = "native_pose_unavailable"
            if orb is not None:
                if person_masked_odometry:
                    packet, masked_area = masked_feature_packet(
                        rgb, np.nan_to_num(depth, nan=0.), orb, social)
                else:
                    packet = feature_packet(rgb, np.nan_to_num(depth, nan=0.), orb)
                if previous_packet is not None and previous_turn_only:
                    pose_evidence = {"source": "ideal_yaw_feedback", "accepted": True}
                elif previous_packet is not None:
                    delta, pose_evidence = odometry_delta(previous_packet, packet, k, matcher)
                    pose_evidence["source"] = "rgbd_pnp"
                    visual_world_delta = None
                    if delta is not None:
                        yaw = math.radians(actor_heading)
                        right = np.array([-math.cos(yaw), math.sin(yaw)])
                        forward = np.array([math.sin(yaw), math.cos(yaw)])
                        left_delta, forward_delta = (rig.odometry_horizontal(delta)
                            if rig is not None else (float(delta[0, 3]),
                                                     float(delta[2, 3])))
                        visual_world_delta = left_delta * right + forward_delta * forward
                    if pose_mode == "rgbd_native_fallback":
                        visual_evidence = pose_evidence
                        pose_evidence = {"source": "rgbd_native_fallback",
                                         "visual": visual_evidence,
                                         "native": native_packet}
                        if visual_world_delta is not None:
                            actor += visual_world_delta
                            fallback_distance_m = 0.
                            fallback_frames = 0
                            if native_packet is not None and native_packet["accepted"]:
                                disagreement = float(np.linalg.norm(
                                    visual_world_delta - np.asarray(
                                        native_packet["measured_delta_world_m"])))
                                pose_evidence["increment_disagreement_m"] = disagreement
                                if disagreement > .10:
                                    pose_healthy = False
                                    pose_failure_reason = "visual_native_increment_disagreement"
                            pose_evidence["accepted"] = pose_healthy
                            pose_evidence["used"] = "visual"
                        elif (native_packet is not None and native_packet["accepted"] and
                              fallback_frames < 4 and
                              fallback_distance_m + native_packet["commanded_step_m"] <= .16 + 1e-9):
                            actor += np.asarray(native_packet["measured_delta_world_m"])
                            fallback_distance_m += native_packet["commanded_step_m"]
                            fallback_frames += 1
                            pose_evidence["accepted"] = True
                            pose_evidence["used"] = "bounded_native_prediction"
                        else:
                            pose_healthy = False
                            pose_failure_reason = "visual_native_fallback_exhausted"
                            pose_evidence["accepted"] = False
                            pose_evidence["used"] = "none"
                    elif visual_world_delta is None:
                        pose_healthy = False
                        pose_failure_reason = "visual_pose_unavailable"
                    else:
                        actor += visual_world_delta
                previous_packet = packet
            command_visual_disagreement = float(np.linalg.norm(actor - command_pose))
            if pose_crosscheck and pose_mode in ("rgbd_yaw", "rgbd_native_fallback") and \
                    command_visual_disagreement > .10:
                pose_healthy = False
                pose_failure_reason = "visual_command_pose_disagreement"
            range_association = (near_range_box_association(
                decision_social, decision_ranges, ANGLES,
                width=width, height=height, focal=focal)
                if (spatial_range_association or sam2_range_association) and
                   decision_social["intent"] == "hold_near_person" else None)
            if sam2_range_association and range_association is not None and \
                    range_association["status"] == "projected" and \
                    range_association["near_hits"]:
                if segment_model is None:
                    from transformers import Sam2Model, Sam2Processor
                    segment_path = MODEL_ROOT / "sam2-hiera-tiny"
                    segment_processor = Sam2Processor.from_pretrained(segment_path)
                    segment_model = Sam2Model.from_pretrained(
                        segment_path).to(device).eval()
                range_association = add_person_mask_association(
                    range_association, decision_rgb, segment_processor,
                    segment_model, device)
            side_obstacle_bypass = bool(range_association is not None and
                range_association["status"] == "projected" and
                range_association["near_hits"] and
                (range_association.get("mask_usable", False) and
                 not range_association["any_in_person_mask"] if sam2_range_association
                 else not range_association["any_in_person_box"]) and
                decision_social.get("local_depth_m", 0.) > 1.35)
            decision = {"intent": "hold_reobserve", "reason": "uninitialized"}
            if not pose_healthy:
                decision = {"intent": "hold_request_help", "reason":
                            pose_failure_reason or "pose_unavailable"}
            elif decision_social["intent"] == "look_to_owner" and not (
                    gaze_route_arbitration and
                    abs(float(decision_social["owner_bearing_deg"])) <= 20. and
                    float(decision_social["local_depth_m"]) > 1.25):
                decision = {"intent": "turn_to_owner", "yaw_delta_deg":
                            -float(decision_social["yaw_delta_deg"]), "reason": decision_social["reason"]}
            elif (decision_social["intent"] in ("approach_bounded", "look_to_owner")
                  or side_obstacle_bypass):
                distance = float(decision_social["local_depth_m"]) / max(.1, math.cos(
                    math.radians(float(decision_social["owner_bearing_deg"]))))
                bearing_world = math.radians(actor_heading - float(decision_social["owner_bearing_deg"]))
                target = actor + distance * np.array([math.sin(bearing_world),
                                                      math.cos(bearing_world)])
                owner_target_distance = float(np.linalg.norm(target - actor))
                intermediate_target = bool(receding_owner_target and
                                           owner_target_distance > 2.2)
                route_target = (actor + 1.8 * (target - actor) /
                                owner_target_distance if intermediate_target
                                else target)
                low_scan_map = None
                if low_scan_packet is not None:
                    mount_forward = MOUNT_FORWARD_M * np.array([
                        math.sin(math.radians(actor_heading)),
                        math.cos(math.radians(actor_heading))])
                    low_scan_map = planner.observe_offset_low_scan(
                        low_scan_packet["ranges_m"], ANGLES,
                        actor + mount_forward, actor_heading, decision_time_s)
                decision = planner.plan(decision_ranges, ANGLES, actor, actor_heading,
                                        route_target, decision_time_s,
                                        free_only_prefix_m=(
                                            mounted_packet["free_only_prefix_m"]
                                            if rig is not None and camera_free_prefix
                                            else None))
                if low_scan_map is not None:
                    decision["low_scan_mapping"] = low_scan_map
                decision["social_intent"] = decision_social["intent"]
                if receding_owner_target:
                    decision["intermediate_owner_target"] = intermediate_target
                    decision["observed_owner_target_distance_m"] = owner_target_distance
                    if intermediate_target and decision["intent"] == "hold_viewpoint":
                        decision["intent"] = "hold_reobserve"
                        decision["reason"] = "intermediate_owner_reobserve"
                decision["gaze_deferred_for_route"] = decision_social["intent"] == "look_to_owner"
                decision["side_obstacle_bypass_attempt"] = side_obstacle_bypass
                if decision["intent"] == "hold_viewpoint" and not (
                        1.20 <= float(decision_social["local_depth_m"]) <= 1.35 and
                        abs(float(decision_social["owner_bearing_deg"])) <= 20.):
                    decision = {"intent": "hold_request_help",
                                "reason": "visual_owner_standoff_disagrees_with_plan",
                                "planner_target_distance_m": decision["target_distance_m"]}
            elif decision_social["intent"] == "hold_near_person":
                distance = float(decision_social["local_depth_m"])
                if (1.20 <= distance <= 1.35 and
                        abs(float(decision_social["owner_bearing_deg"])) <= 20.):
                    decision = {"intent": "hold_viewpoint", "reason": "fresh_owner_standoff"}
                elif (far_conversation_viewpoint and
                      1.45 <= distance <= 1.65 and
                      abs(float(decision_social["owner_bearing_deg"])) <= 20. and
                      float(decision_social["nearest_forward_range_m"]) >= .9):
                    decision = {"intent": "hold_viewpoint_far",
                                "reason": "far_conversation_viewpoint_near_range_stop",
                                "owner_depth_m": distance,
                                "nearest_forward_range_m":
                                    decision_social["nearest_forward_range_m"]}
                else:
                    decision = {"intent": "hold_request_help",
                                "reason": "near_range_before_owner_standoff",
                                "owner_depth_m": distance,
                                "nearest_forward_range_m":
                                    decision_social["nearest_forward_range_m"]}
            else:
                decision = {"intent": decision_social["intent"], "reason": decision_social["reason"]}
            monitor_health = (near_body_monitor.wait_health(monitor_mark)
                if near_body_monitor is not None else None)
            owner_option = None
            if owner_option_step is not None and decision["intent"] != \
                    "hold_request_help":
                owner_option = owner_option_step({
                    "social": decision_social,
                    "pose_healthy": pose_healthy,
                    "known_low_beams": int((~np.isnan(ranges[0])).sum()),
                    "known_mid_beams": int((~np.isnan(ranges[1])).sum()),
                    "source_healthy": bool(monitor_health and
                                           monitor_health["authorized"]),
                    "proposed_intent": decision["intent"],
                    "frame": frame_index})
                if owner_option == "HOLD":
                    decision = {"intent": "hold_request_help",
                                "reason": "learned_owner_route_hold"}
                elif owner_option == "OWNER_TEACH":
                    owner_depth = float(decision_social["local_depth_m"])
                    far = owner_depth >= 1.45
                    decision = {"intent": ("hold_viewpoint_far" if far else
                                           "hold_viewpoint"),
                        "reason": "learner_owner_viewpoint",
                        "owner_depth_m": owner_depth,
                        "nearest_forward_range_m": decision_social[
                            "nearest_forward_range_m"]}
                elif owner_option == "OWNER_CONTINUE" and decision[
                        "intent"] in ("hold_viewpoint", "hold_viewpoint_far"):
                    decision = {"intent": "hold_reobserve",
                                "reason": "learner_owner_route_continuation"}
                elif owner_option != "OWNER_CONTINUE":
                    raise ValueError(f"Unsupported owner option: {owner_option}")
            command_yaw = float(decision.get("yaw_delta_deg", 0.))
            command_step = min(.04, float(decision.get("forward_step_m", 0.)))
            if (monitor_health is not None and not monitor_health["authorized"]
                    and (command_step or decision["intent"] in (
                        "hold_viewpoint", "hold_viewpoint_far"))):
                decision = {"intent": "hold_request_help",
                    "reason": "independent_near_body_source_unhealthy",
                    "source_reason": monitor_health["reason"]}
                command_step = 0.
                command_yaw = 0.
            accepted = True
            if command_step:
                candidate = actual + command_step * np.array([
                    math.sin(math.radians(actual_heading)),
                    math.cos(math.radians(actual_heading))])
                eye_at_candidate = np.array([candidate[0], scene.floor_y + .42, candidate[1]])
                safety = scene.range_fan(eye_at_candidate, actual_heading,
                                         owner_xz=person_at_action_xz, angles_deg=ANGLES)
                accepted = bool(scene.floor_supported(*candidate) and
                                np.nanmin(safety) >= .17)
                if accepted:
                    actual = candidate
                    command_pose += command_step * np.array([
                        math.sin(math.radians(command_heading)),
                        math.cos(math.radians(command_heading))])
                    if native_pose is not None:
                        pending_native_packet = native_pose.advance(
                            command_step * np.array([
                                math.sin(math.radians(actual_heading)),
                                math.cos(math.radians(actual_heading))]))
                    if pose_mode == "command":
                        actor += command_step * np.array([
                            math.sin(math.radians(actor_heading)),
                            math.cos(math.radians(actor_heading))])
                else:
                    planner.observe_rejected_step(actor, actor_heading, command_step, t)
                    decision = {"intent": "hold_request_help", "reason": "motion_rejected"}
            actual_heading += command_yaw
            if near_body_monitor is not None and (command_yaw or
                    (command_step and accepted)):
                near_body_monitor.update_pose(actual, actual_heading)
            actor_heading += command_yaw  # ideal yaw sensor surrogate in RGB-D branch
            command_heading += command_yaw
            previous_turn_only = bool(command_yaw and not command_step)
            previous_action = {"yaw_delta_deg": command_yaw,
                               "forward_step_m": command_step, "accepted": accepted}
            image = rgb.copy()
            proposals = social.get("proposals", [])
            selected = social.get("chosen_proposal")
            if selected is not None and selected < len(proposals):
                box = np.rint(proposals[selected]["box_xyxy"]).astype(int)
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (70, 255, 70), 2)
            cv2.putText(image, f'{frame_index:02d} {decision["intent"]} | {social["intent"]}',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            writer.append_data(image)
            if down_writer is not None:
                down_writer.append_data(down_rgb)
                Image.fromarray(down_rgb).save(output / f"down_rgb_{frame_index:03d}.png")
                Image.fromarray(np.where(np.isfinite(down_depth),
                    np.clip(down_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                        output / f"down_depth_{frame_index:03d}.png")
            if save_raw_frames:
                Image.fromarray(rgb).save(output / f"rgb_{frame_index:03d}.png")
                Image.fromarray(np.where(np.isfinite(depth),
                             np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                                 output / f"depth_{frame_index:03d}.png")
            frame_rows.append({"frame": frame_index, "capture_time_s": t,
                "decision_time_s": decision_time_s,
                "observation_age_at_action_s": simulated_inference_delay_s,
                "owner_actor": social,
                "decision_owner_actor": decision_social,
                "fast_revalidation": fast_updates,
                "range_person_box_association": range_association,
                "planner_decision": decision,
                "learner_owner_option": owner_option,
                "independent_near_body_health": monitor_health,
                "inference_wall_s": inference_wall_s,
                "evaluator_owner_xz_at_action_m": list(person_at_action_xz),
                "evaluator_nearest_forward_range_at_action_m": float(np.nanmin(
                    range_at_action[:, np.abs(ANGLES) <= 20])),
                "range_low_m": [float(v) if np.isfinite(v) else None for v in ranges[0]],
                "range_mid_m": [float(v) if np.isfinite(v) else None for v in ranges[1]],
                "camera_free_prefix_low_m": (
                    mounted_packet["free_only_prefix_m"][0].tolist()
                    if rig is not None else None),
                "camera_free_prefix_mid_m": (
                    mounted_packet["free_only_prefix_m"][1].tolist()
                    if rig is not None else None),
                "camera_free_prefix_loss": (
                    mounted_packet["free_prefix_loss"].tolist()
                    if rig is not None else None),
                "low_scan_ranges_m": ([2.1 if np.isinf(v) else
                                        float(v) if np.isfinite(v) else None
                                        for v in low_scan_packet["ranges_m"]]
                                       if low_scan_packet is not None else None),
                "low_scan_valid_front_beams": (low_scan_packet["valid_front_beams"]
                                               if low_scan_packet is not None else None),
                "low_scan_finite_hits_within_horizon": (
                    low_scan_packet["finite_hits_within_horizon"]
                    if low_scan_packet is not None else None),
                "range_angles_deg": ANGLES.tolist(),
                "range_mode": range_mode,
                "known_range_beam_count": int(np.isfinite(ranges).sum() +
                                               np.isinf(ranges).sum()),
                "unknown_range_beam_count": int(np.isnan(ranges).sum()),
                "known_low_beam_count": int((~np.isnan(ranges[0])).sum()),
                "known_mid_beam_count": int((~np.isnan(ranges[1])).sum()),
                "pose_mode": pose_mode, "pose_healthy": pose_healthy,
                "pose_evidence": pose_evidence,
                "native_pose_packet_at_capture": native_packet,
                "native_fallback_unverified_distance_m": fallback_distance_m,
                "native_fallback_unverified_frames": fallback_frames,
                "person_masked_odometry": person_masked_odometry,
                "masked_person_area_px": masked_area,
                "actor_command_visual_disagreement_at_capture_m":
                    command_visual_disagreement,
                "actor_command_odometry_xy_m": command_pose.tolist(),
                "actor_pose_xy_m": actor.tolist(), "actor_heading_deg": actor_heading,
                "commanded_step_m": command_step, "commanded_yaw_deg": command_yaw,
                "command_accepted": accepted,
                "evaluator_actual_xz_m": actual.tolist(),
                "evaluator_actual_xz_at_capture_m": actual_at_capture.tolist(),
                "evaluator_actual_heading_deg": actual_heading,
                "evaluator_owner_xz_m": list(person_xz),
                "evaluator_pose_error_at_capture_m": float(np.linalg.norm(
                    actor - (actual_at_capture - start_xz))),
                "evaluator_pose_error_m": float(np.linalg.norm(actor - (actual - start_xz)))})
            sim_time_s = decision_time_s + DT_S
            if decision["intent"] in ("hold_viewpoint", "hold_viewpoint_far", "hold_request_help",
                                      "hold_ask_show_again"):
                break
    finally:
        writer.close()
        if down_writer is not None:
            down_writer.close()
        if monitor_writer is not None:
            monitor_writer.close()
    final = frame_rows[-1]
    report = {"schema": "ape.real-capture-dynamic-owner-loop.v1",
              "scene": config.get("source_family", "ScanSceneverse") + "/" + config["name"],
              "source_capture": str(trace_path), "pose_mode": pose_mode,
              "native_pose_source": (native_pose.source if native_pose is not None else None),
              "native_pose_trace_sha256": (native_pose.source_sha256 if native_pose is not None else None),
              "native_pose_qualification": (
                  "Historical rejected-body joint/contact odometry error-shape transfer with illustrative impairments; no actual sensor packet, selected-body qualification or real IMU calibration"
                  if native_pose is not None else None),
              "native_fallback_limits": ({"maximum_unverified_translation_m": .16,
                  "maximum_unverified_frames": 4,
                  "maximum_visual_native_increment_disagreement_m": .10,
                  "require_native_contact_sample": True}
                  if pose_mode == "rgbd_native_fallback" else None),
              "range_mode": range_mode,
              "raw_owner_frames_saved": save_raw_frames,
              "camera_free_prefix_enabled": bool(rig is not None and camera_free_prefix),
              "mounted_low_scan_enabled": mounted_low_scan,
              "mounted_low_scan_contract": ({"mount_body_forward_m": MOUNT_FORWARD_M,
                  "height_m": .1, "front_arc_deg": [-60, 60],
                  "scan_rate_hz": 10, "mean_scan_age_s": .05,
                  "illustrative_range_error_m": .03,
                  "illustrative_dropout": .05,
                  "source_spec": "https://www.slamtec.com/en/c1/spec",
                  "qualification": "product-inspired simulated scan, no body self-occlusion or physical packet"}
                  if mounted_low_scan else None),
              "person": ("CC-BY photogrammetry scan of a seated person rigidly moved laterally"
                         if person_style == "scanned" else
                         "crouched primitive geometry, scripted lateral motion"),
              "person_source": ("https://sketchfab.com/3d-models/sitting-person-0a9933b67fff41a3a29094676dbede9f"
                                if person_style == "scanned" else None),
              "item": ("absent in scanned-person branch; not taught or found in this turn"
                       if person_style == "scanned" else
                       "simple red cylinder held by person; not taught or found in this turn"),
              "body": "kinematic 17 cm circular proxy; no selected articulated design",
              "sensor_sources": [(
                  "real-capture Gaussian-splat RGB" if source.get("source_type") ==
                  "gaussian_splat_mesh" else "textured real-capture RGB"),
                  "mesh-raycast depth from separately published collision geometry" if
                  source.get("source_type") == "gaussian_splat_mesh" else
                  "mesh-raycast RGB-aligned depth",
                                 ("two-height independent mesh-raycast range fan" if
                                  range_mode == "two_plane_range" else
                                  "nominal mounted upper/lower RGB-D projected into body-frame low/mid free rays" if
                                  rig is not None else
                                  "conservative low/mid free evidence projected from upright and 45-degree-down RGB-D frames" if
                                  range_mode == "dual_rgbd_projection" else
                                  "conservative low/mid free evidence projected from the same upright RGB-D frame"),
                                 *(["illustrative offset 10Hz front-arc low planar scan for route map only"]
                                   if mounted_low_scan else []),
                                 *(["historical 50 cm joint/contact odometry error-shape packet"]
                                   if native_pose is not None else []),
                                 "ideal yaw feedback surrogate"],
              "person_teaching": ("evaluator-selected visible pixel in mounted upper image" if
                                  rig is not None else config.get("click_source",
                                  "human click at image pixel (320,160) in first frame")),
              "person_click_xy": list(rig.click_xy if rig is not None else
                                      config.get("click_xy", (320, 160))),
              "mounted_camera_screen": (str(rig.screen_path.relative_to(ROOT))
                                        if rig is not None else None),
              "scenario_placement_screen": config.get("placement_screen"),
              "gaze_route_arbitration": gaze_route_arbitration,
              "pose_crosscheck": pose_crosscheck,
              "person_masked_odometry": person_masked_odometry,
              "person_mask_rule": "exclude actor-selected person box plus 12 px from ORB features" if person_masked_odometry else None,
              "pose_crosscheck_rule": "hold if gated visual RGB-D and accepted-command odometry differ by more than 10 cm" if pose_crosscheck else None,
              "gaze_rule": "Defer body gaze turn while a depth-confirmed owner remains within 20 degrees and outside 1.25 m; keep re-observing after each route step" if gaze_route_arbitration else "owner gaze turn takes priority",
              "virtual_frame_period_s": DT_S,
              "simulated_inference_delay_s": simulated_inference_delay_s,
              "inference_time_advanced_in_simulator": simulated_inference_delay_s > 0,
              "post_inference_range_visible_to_actor": fast_revalidate_during_inference,
              "fast_revalidate_during_inference": fast_revalidate_during_inference,
              "spatial_range_association": spatial_range_association,
              "sam2_range_association": sam2_range_association,
              "far_conversation_viewpoint_policy": far_conversation_viewpoint,
              "receding_owner_target": receding_owner_target,
              "depth_separated_person_overlap": depth_separated_overlap,
              "range_association_rule": ("project <1.25 m range rays into actor-box-prompted SAM 2 person mask; only all-outside rays with usable mask may enter ordinary route planning" if sam2_range_association else
                  "project <1.25 m two-height range rays into fresh selected person box; only all-outside rays may enter ordinary route planning" if spatial_range_association else None),
              "inference_monitor_video": str(output / "inference_monitor.mp4") if monitor_writer is not None else None,
              "outcome": final["planner_decision"]["intent"],
              "frames": len(frame_rows),
              "accepted_travel_m": sum(row["commanded_step_m"] for row in frame_rows
                                       if row["command_accepted"]),
              "max_pose_error_m": max(row["evaluator_pose_error_m"] for row in frame_rows),
              "max_pose_error_at_capture_m": max(
                  row["evaluator_pose_error_at_capture_m"] for row in frame_rows),
              "max_actor_pose_disagreement_m": max(
                  row["actor_command_visual_disagreement_at_capture_m"]
                  for row in frame_rows),
              "final_observed_owner_depth_m":
                  final["decision_owner_actor"].get("local_depth_m"),
              "final_evaluator_owner_center_distance_m": float(np.linalg.norm(
                  np.asarray(final["evaluator_actual_xz_m"]) -
                  np.asarray(final["evaluator_owner_xz_m"]))),
              "observed_standoff_band_met": bool(
                  final["planner_decision"]["intent"] == "hold_viewpoint" and
                  1.20 <= final["decision_owner_actor"].get("local_depth_m", -1) <= 1.35),
              "observed_far_conversation_band_met": bool(
                  final["planner_decision"]["intent"] == "hold_viewpoint_far" and
                  1.45 <= final["decision_owner_actor"].get("local_depth_m", -1) <= 1.65),
              "trace": frame_rows}
    (output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "trace"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=tuple(SCENES), default="serving")
    p.add_argument("--person-style", choices=("proxy", "scanned"), default="proxy")
    p.add_argument("--range-mode", choices=("two_plane_range", "rgbd_projection",
                                            "dual_rgbd_projection", "mounted_body_rgbd"),
                   default="two_plane_range")
    p.add_argument("--pose-mode", choices=("command", "rgbd_yaw",
                                           "native_shaped", "rgbd_native_fallback"),
                   default="command")
    p.add_argument("--native-source", choices=NATIVE_SOURCES,
                   default="varied150-forward-seed144")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--gaze-route-arbitration", action="store_true")
    p.add_argument("--pose-crosscheck", action="store_true")
    p.add_argument("--person-mask-odometry", action="store_true")
    p.add_argument("--simulated-inference-delay-s", type=float, default=0.)
    p.add_argument("--fast-revalidate-during-inference", action="store_true")
    p.add_argument("--spatial-range-association", action="store_true")
    p.add_argument("--sam2-range-association", action="store_true")
    p.add_argument("--far-conversation-viewpoint", action="store_true")
    p.add_argument("--depth-separated-overlap", action="store_true")
    p.add_argument("--receding-owner-target", action="store_true")
    p.add_argument("--mount-screen", type=Path)
    p.add_argument("--no-camera-free-prefix", action="store_true")
    p.add_argument("--mounted-low-scan", action="store_true")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.scene, args.pose_mode, steps=args.steps, device=args.device,
                         native_source=args.native_source,
                         person_style=args.person_style, range_mode=args.range_mode,
                         gaze_route_arbitration=args.gaze_route_arbitration,
                         pose_crosscheck=args.pose_crosscheck,
                         person_masked_odometry=args.person_mask_odometry,
                         simulated_inference_delay_s=args.simulated_inference_delay_s,
                         fast_revalidate_during_inference=args.fast_revalidate_during_inference,
                         spatial_range_association=args.spatial_range_association,
                         sam2_range_association=args.sam2_range_association,
                         far_conversation_viewpoint=args.far_conversation_viewpoint,
                         depth_separated_overlap=args.depth_separated_overlap,
                         receding_owner_target=args.receding_owner_target,
                         mount_screen=args.mount_screen,
                         camera_free_prefix=not args.no_camera_free_prefix,
                         mounted_low_scan=args.mounted_low_scan,
                         output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
