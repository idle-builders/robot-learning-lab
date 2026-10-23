"""Continue a captured-room owner approach into one honest teach/find attempt.

The owner approach is run afresh first. A supplied teaching click and scripted
confirmation bind a textured HOPE grocery object in the same captured room.
The owner then removes or relocates that object while out of view. Search uses
only fresh rendered RGB proposals and an appearance exemplar; object geometry,
placement and visibility masks stay evaluator-side. This is a provisional
kinematic camera/body-turn episode, not articulated skill qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from real_capture_dynamic_scene import (ScannedSittingPerson,
                                        TexturedCaptureScene,
                                        TexturedFloorObject,
                                        scene_from_capture_source)
from run_real_capture_owner_loop import SCENES, owner_pose, run as run_owner


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
from active_scan import Vision, _iou  # noqa: E402


SCREEN = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen/screen.json"
RUNS = SCREEN.parent / "runs"
TARGET_NAME = "Parmesan"
DISTRACTOR_NAME = "CreamCheese"
SEARCH_YAW_OFFSETS_DEG = (0., 45., -45.)
SAME_OBJECT_RADIUS_M = .10
VERIFY_SCORE_HYSTERESIS = .05


def inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) else None


def box_local_depth(box: list[float] | None, depth: np.ndarray,
                    vfov_deg: float = 75.,
                    max_depth_m: float = 2.) -> dict:
    """Estimate a fresh camera-relative point or abstain on a mixed patch."""
    if box is None:
        return {"status": "no_box"}
    height, width = depth.shape
    u = (box[0] + box[2]) / 2
    v = (box[1] + box[3]) / 2
    x, y = int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))
    patch = depth[max(0, y - 2):min(height, y + 3),
                  max(0, x - 2):min(width, x + 3)]
    valid = patch[np.isfinite(patch) & (patch > .08) &
                  (patch < max_depth_m)]
    if len(valid) < 12:
        return {"status": "insufficient_depth", "valid_pixels": int(len(valid))}
    distance = float(np.median(valid))
    mad = float(np.median(np.abs(valid - distance)))
    if mad > .05:
        return {"status": "mixed_depth", "valid_pixels": int(len(valid)),
                "median_absolute_deviation_m": mad}
    focal = height / (2 * math.tan(math.radians(vfov_deg) / 2))
    right = float((u - width / 2) / focal * distance)
    return {"status": "located", "valid_pixels": int(len(valid)),
            "median_absolute_deviation_m": mad,
            "camera_right_m": right, "camera_forward_m": distance,
            "camera_up_m": float((height / 2 - v) / focal * distance),
            "bearing_deg": math.degrees(math.atan2(right, distance))}


def candidate_map_xy(box: list[float], depth: np.ndarray,
                     actor_xy: np.ndarray, actor_heading_deg: float,
                     vfov_deg: float = 75.,
                     max_depth_m: float = 2.) -> tuple[dict, list[float] | None]:
    """Locate a proposal from the actor's depth/yaw packet, without scene truth."""
    local = box_local_depth(box, depth, vfov_deg, max_depth_m)
    if local["status"] != "located":
        return local, None
    yaw = math.radians(actor_heading_deg)
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([-math.cos(yaw), math.sin(yaw)])
    point = (actor_xy + local["camera_forward_m"] * forward +
             local["camera_right_m"] * right)
    return local, point.tolist()


def same_observed_object(first: dict, second: dict) -> bool:
    """Suppress duplicated labels/views only when image or depth evidence agrees."""
    a, b = first.get("actor_map_xy_m"), second.get("actor_map_xy_m")
    if first["view_frame"] == second["view_frame"] and \
            _iou(first["box"], second["box"]) >= .85 and \
            (a is None or b is None):
        return True
    return bool(a is not None and b is not None and
                np.linalg.norm(np.asarray(a) - np.asarray(b)) <= SAME_OBJECT_RADIUS_M)


def rank_distinct_candidates(candidates: list[tuple[int, dict]],
                             score_key: str = "exemplar_similarity") -> list[tuple[int, dict]]:
    ranked = sorted(candidates, key=lambda pair: pair[1][score_key],
                    reverse=True)
    distinct = []
    for pair in ranked:
        if not any(same_observed_object(pair[1], prior[1]) for prior in distinct):
            distinct.append(pair)
    return distinct


def colour_histogram(image: np.ndarray, box: list[float]) -> np.ndarray | None:
    x0, y0, x1, y1 = (round(value) for value in box)
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    result = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    cv2.normalize(result, result)
    return result


def colour_similarity(exemplar: np.ndarray, image: np.ndarray,
                      box: list[float]) -> float:
    sample = colour_histogram(image, box)
    return (1. - cv2.compareHist(exemplar, sample, cv2.HISTCMP_BHATTACHARYYA)
            if sample is not None else -1.)


def camera_photometry(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return rgb
    if mode == "cool_dim":
        normalized = np.power(rgb.astype(np.float32) / 255., 1.2)
        return np.rint(np.clip(normalized * np.array([.82, .95, 1.12]),
                               0., 1.) * 255).astype(np.uint8)
    raise ValueError(f"Unknown search photometry: {mode}")


def multicue_choice(candidates: list[tuple[int, dict]],
                    dino_acceptance: float,
                    colour_acceptance: float
                    ) -> tuple[tuple[int, dict] | None, str | None,
                               float | None, bool]:
    """Choose by independent calibrated cues, never by evaluator identity."""
    choices = {}
    for cue, key, minimum in (("dino", "exemplar_similarity", dino_acceptance),
                              ("colour", "colour_similarity", colour_acceptance)):
        ranked = rank_distinct_candidates(candidates, key)
        if not ranked:
            continue
        best = ranked[0]
        margin = (best[1][key] - ranked[1][1][key]
                  if len(ranked) > 1 else None)
        if best[1][key] >= minimum and (margin is None or margin >= .08):
            choices[cue] = (best, margin)
    if "dino" in choices and "colour" in choices and not same_observed_object(
            choices["dino"][0][1], choices["colour"][0][1]):
        return None, None, None, True
    for cue in ("dino", "colour"):
        if cue in choices:
            best, margin = choices[cue]
            return best, cue, margin, False
    return None, None, None, False


def camera(scene: TexturedCaptureScene, xy: np.ndarray, yaw_deg: float,
           owner_xz: tuple[float, float] | None,
           width: int | None = None, height: int | None = None) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
    width = width or getattr(scene, "WIDTH", 640)
    height = height or getattr(scene, "HEIGHT", 360)
    eye = (scene.eye_at(float(xy[0]), float(xy[1]))
           if hasattr(scene, "eye_at") else
           np.array([xy[0], scene.floor_y + .42, xy[1]]))
    if eye is None:
        raise ValueError(f"No local floor under camera at {xy}")
    yaw = math.radians(yaw_deg)
    forward = np.array([math.sin(yaw), 0., math.cos(yaw)])
    rgb, depth, ids = scene.observe(eye, forward, owner_xz=owner_xz,
                                    width=width, height=height,
                                    return_prop_ids=True)
    angles = np.arange(-90., 91., 5., dtype=np.float32)
    ranges = scene.range_fan(eye, yaw_deg, owner_xz=owner_xz,
                             angles_deg=angles)
    return rgb, depth, ids, ranges


def save_frame(folder: Path, writer: object, index: int, rgb: np.ndarray,
               depth: np.ndarray, caption: str, proposals: list[dict]) -> None:
    Image.fromarray(rgb).save(folder / f"rgb_{index:03d}.png")
    Image.fromarray(np.where(np.isfinite(depth),
        np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            folder / f"depth_{index:03d}.png")
    display = rgb.copy()
    for proposal in proposals:
        box = np.rint(proposal["box"]).astype(int)
        cv2.rectangle(display, tuple(box[:2]), tuple(box[2:]),
                      (64, 240, 64), 2)
    cv2.putText(display, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 2)
    writer.append_data(display)


def run_search(scene: TexturedCaptureScene, vision: Vision,
               actual_xy: np.ndarray, actual_yaw: float, actor_xy: np.ndarray,
               actor_yaw: float, exemplar: np.ndarray, acceptance: float,
               present: bool, out: Path, start_time_s: float,
               search_xz: tuple[float, float],
               search_distractor_xz: tuple[float, float],
               owner_start_xz: tuple[float, float],
               target_name: str = TARGET_NAME,
               distractor_name: str = DISTRACTOR_NAME,
               object_scale: float = .012,
               camera_width: int = 640,
               camera_height: int = 360,
               identity_mode: str = "dino",
               colour_exemplar: np.ndarray | None = None,
               colour_acceptance: float | None = None,
               search_photometry: str = "none") -> dict:
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    placements = {distractor_name: (
        TexturedFloorObject(props / f"{distractor_name}.obj",
                            scale=object_scale), search_distractor_xz, 0.)}
    if present:
        placements[target_name] = (
            TexturedFloorObject(props / f"{target_name}.obj",
                                scale=object_scale), search_xz, 0.)
    scene.set_props(placements)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    writer = imageio.get_writer(str(out / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index, offset in enumerate(SEARCH_YAW_OFFSETS_DEG):
            heading = actual_yaw + offset
            rgb, depth, ids, ranges = camera(scene, actual_xy, heading, None,
                                            camera_width, camera_height)
            rgb = camera_photometry(rgb, search_photometry)
            started = time.perf_counter()
            proposals = vision.detect(Image.fromarray(rgb))
            if proposals:
                crops = [Image.fromarray(rgb).crop(tuple(round(v) for v in p["box"]))
                         for p in proposals]
                features = vision.feature(crops)
                for proposal, feature in zip(proposals, features):
                    proposal["exemplar_similarity"] = float(feature @ exemplar)
                    if identity_mode == "multi_cue":
                        proposal["colour_similarity"] = colour_similarity(
                            colour_exemplar, rgb, proposal["box"])
                    proposal["view_frame"] = index
                    proposal["local_depth"], proposal["actor_map_xy_m"] = candidate_map_xy(
                        proposal["box"], depth, actor_xy, actor_yaw + offset,
                        getattr(scene, "VFOV_DEG", 75.),
                        getattr(scene, "MAX_OBJECT_DEPTH_M", 2.))
                    proposal["border_crop"] = bool(proposal["box"][0] < 8 or
                        proposal["box"][2] > rgb.shape[1] - 8)
                    if not proposal["border_crop"]:
                        candidates.append((index, proposal))
            infer_wall_s = time.perf_counter() - started
            save_frame(out, writer, index, rgb, depth,
                       f"find: look {offset:+.0f} deg", proposals)
            truth = bbox(ids == target_name)
            for proposal in proposals:
                proposal["evaluator_target_iou"] = _iou(proposal["box"], truth)
            rows.append({"frame": index, "capture_time_s": start_time_s + .2 * index,
                "camera_heading_deg": heading,
                "actor_heading_deg": actor_yaw + offset,
                "actor_pose_relative_to_owner_start_xy_m": actor_xy.tolist(),
                "camera_height_m": .42, "depth_file": f"depth_{index:03d}.png",
                "rgb_file": f"rgb_{index:03d}.png",
                "range_angles_deg": np.arange(-90., 91., 5.).tolist(),
                "range_low_m": [float(x) if np.isfinite(x) else None for x in ranges[0]],
                "range_mid_m": [float(x) if np.isfinite(x) else None for x in ranges[1]],
                "inference_wall_s": infer_wall_s, "proposals": proposals,
                "evaluator_target_visible_px": int((ids == target_name).sum()),
                "evaluator_target_box_xyxy": truth})
            ranked = rank_distinct_candidates(candidates)
            if identity_mode == "dino" and ranked and \
                    ranked[0][1]["exemplar_similarity"] >= acceptance:
                break
        ranked = rank_distinct_candidates(candidates)
        best = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        margin = (best[1]["exemplar_similarity"] -
                  runner_up[1]["exemplar_similarity"]
                  if best is not None and runner_up is not None else None)
        selected_cue = "dino"
        disagreement = False
        if identity_mode == "multi_cue":
            choice, selected_cue, margin, disagreement = multicue_choice(
                candidates, acceptance, colour_acceptance)
            best = choice
        verification = None
        action = "report_unidentified"
        eligible = bool(best is not None and best[1]["actor_map_xy_m"] is not None and
            (identity_mode == "multi_cue" or
             (best[1]["exemplar_similarity"] >= acceptance and
              (margin is None or margin >= .08))))
        if eligible:
            index, candidate = best
            center = (candidate["box"][0] + candidate["box"][2]) / 2
            focal = camera_height / (2 * math.tan(math.radians(
                getattr(scene, "VFOV_DEG", 75.)) / 2))
            correction = math.degrees(math.atan((camera_width / 2 - center) / focal))
            heading = rows[index]["camera_heading_deg"] + correction
            rgb, depth, ids, ranges = camera(scene, actual_xy, heading, None,
                                            camera_width, camera_height)
            rgb = camera_photometry(rgb, search_photometry)
            started = time.perf_counter()
            found = vision.detect(Image.fromarray(rgb))
            if found:
                features = vision.feature([Image.fromarray(rgb).crop(
                    tuple(round(v) for v in p["box"])) for p in found])
                for p, f in zip(found, features):
                    p["exemplar_similarity"] = float(f @ exemplar)
                    if identity_mode == "multi_cue":
                        p["colour_similarity"] = colour_similarity(
                            colour_exemplar, rgb, p["box"])
                    p["local_depth"], p["actor_map_xy_m"] = candidate_map_xy(
                        p["box"], depth, actor_xy,
                        rows[index]["actor_heading_deg"] + correction,
                        getattr(scene, "VFOV_DEG", 75.),
                        getattr(scene, "MAX_OBJECT_DEPTH_M", 2.))
            verify_wall_s = time.perf_counter() - started
            save_frame(out, writer, len(rows), rgb, depth, "find: verify",
                       found)
            score_key = ("colour_similarity" if selected_cue == "colour"
                         else "exemplar_similarity")
            winner = max(found, key=lambda p: p[score_key],
                         default=None)
            identity_gap = (float(np.linalg.norm(
                np.asarray(winner["actor_map_xy_m"]) -
                np.asarray(candidate["actor_map_xy_m"])))
                if winner and winner["actor_map_xy_m"] is not None else None)
            verification_floor = (colour_acceptance - VERIFY_SCORE_HYSTERESIS
                if selected_cue == "colour" else
                max(.5, acceptance - VERIFY_SCORE_HYSTERESIS))
            verification = {"frame": len(rows), "camera_heading_deg": heading,
                "capture_time_s": start_time_s + .2 * len(rows),
                "rgb_file": f"rgb_{len(rows):03d}.png",
                "depth_file": f"depth_{len(rows):03d}.png",
                "inference_wall_s": verify_wall_s, "proposals": found,
                "selected": winner,
                "same_object_distance_m": identity_gap,
                "selected_cue": selected_cue,
                "required_similarity": verification_floor,
                "range_angles_deg": np.arange(-90., 91., 5.).tolist(),
                "range_low_m": [float(x) if np.isfinite(x) else None for x in ranges[0]],
                "range_mid_m": [float(x) if np.isfinite(x) else None for x in ranges[1]],
                "evaluator_target_box_xyxy": bbox(ids == target_name),
                "evaluator_selected_target_iou": _iou(winner["box"],
                    bbox(ids == target_name)) if winner else 0.}
            if (winner and identity_gap is not None and
                    identity_gap <= SAME_OBJECT_RADIUS_M and
                    winner[score_key] >= verification_floor):
                action = "candidate_to_owner_for_confirmation"
                location = winner["local_depth"].copy()
                if location["status"] == "located":
                    actor_heading = math.radians(actor_yaw +
                        (heading - actual_yaw))
                    forward_xy = np.array([math.sin(actor_heading),
                                           math.cos(actor_heading)])
                    right_xy = np.array([-math.cos(actor_heading),
                                         math.sin(actor_heading)])
                    predicted = (np.asarray(owner_start_xz) + actor_xy +
                        location["camera_forward_m"] * forward_xy +
                        location["camera_right_m"] * right_xy)
                    location["actor_map_target_xz_m"] = predicted.tolist()
                    location["evaluator_target_position_error_m"] = float(
                        np.linalg.norm(predicted - search_xz))
                verification["candidate_localization"] = location
    finally:
        writer.close()
    result = {"schema": "ape.real-capture-teach-find-search.v1",
        "target_name": target_name, "distractor_name": distractor_name,
        "target_present": present,
        "actor_target_position_available": False,
        "actor_inputs": ("fresh RGB generic proposals, DINOv2 and taught HSV color exemplar; depth/yaw for candidate localization"
                         if identity_mode == "multi_cue" else
                         "fresh RGB proposals and DINOv2 exemplar; saved RGB-D/range/ideal yaw for later sensor analysis"),
        "identity_mode": identity_mode,
        "metric_candidate_max_depth_m": getattr(scene, "MAX_OBJECT_DEPTH_M", 2.),
        "search_photometry": search_photometry,
        "appearance_acceptance": acceptance,
        "colour_acceptance": colour_acceptance,
        "selected_cue": selected_cue,
        "cue_disagreement": disagreement,
        "verification_score_hysteresis": VERIFY_SCORE_HYSTERESIS,
        "same_object_radius_m": SAME_OBJECT_RADIUS_M,
        "action": action,
        "best_similarity": best[1]["exemplar_similarity"] if best else None,
        "best_colour_similarity": best[1].get("colour_similarity") if best else None,
        "best_proposal_evaluator_target_iou": best[1]["evaluator_target_iou"] if best else 0.,
        "margin": margin,
        "distinct_candidate_count": len(ranked),
        "raw_candidate_count": len(candidates),
        "verification": verification,
        "frames": len(rows) + (1 if verification else 0),
        "healthy_views": len(rows),
        "target_visible_in_any_view": any(r["evaluator_target_visible_px"] > 0 for r in rows),
        "wrong_confident_candidate": bool(action == "candidate_to_owner_for_confirmation"
            and (not present or not verification or
                 verification["evaluator_selected_target_iou"] < .25)),
        "observations": rows,
        "video": str((out / "first_person.mp4").relative_to(ROOT))}
    (out / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "observations"}


def run_integrated(scene_key: str, output: Path,
                   simulated_inference_delay_s: float = 0.,
                   placement_screen: Path | None = None,
                   delayed_conversation_screen: Path | None = None,
                   rerun_far_controller: bool = False,
                   depth_separated_overlap: bool = False,
                   pose_mode: str = "rgbd_yaw",
                   native_source: str = "varied150-forward-seed144",
                   target_name: str = TARGET_NAME,
                   distractor_name: str = DISTRACTOR_NAME,
                   object_scale: float = .012,
                   camera_width: int = 640,
                   camera_height: int = 360,
                   identity_mode: str = "dino",
                   search_photometry: str = "none",
                   near_body_monitor=None,
                   owner_option_step=None,
                   owner_raw_frames: bool = True) -> dict:
    if identity_mode not in ("dino", "multi_cue") or search_photometry not in (
            "none", "cool_dim"):
        raise ValueError("Unknown item identity or search photometry mode")
    output = output.resolve()
    if rerun_far_controller and delayed_conversation_screen is None:
        raise ValueError("Far controller requires its frozen placement screen")
    if delayed_conversation_screen is not None and not math.isclose(
            simulated_inference_delay_s, .85):
        raise ValueError("Frozen delayed conversation screen uses 0.85 s age")
    placement_source = (delayed_conversation_screen or placement_screen or SCREEN).resolve()
    placement_screen_data = json.loads(placement_source.read_text())
    placement = next((x for x in placement_screen_data["scenes"]
                      if x.get("scene_key") == scene_key), None)
    if placement is None or placement["status"] != "provisional_supported_visible":
        raise ValueError(f"No supported frozen teach/find placement for {scene_key}")
    if target_name == distractor_name or not all(
            (ROOT / ".cache/ape-high-level/hope_meshes_eval" / f"{name}.obj").is_file()
            for name in (target_name, distractor_name)):
        raise ValueError("Object pair must name two distinct available HOPE meshes")
    if placement.get("target_name", target_name) != target_name or \
            placement.get("distractor_name", distractor_name) != distractor_name:
        raise ValueError("Object pair does not match frozen placement screen")
    output.mkdir(parents=True, exist_ok=True)
    if delayed_conversation_screen is None or rerun_far_controller:
        approach_dir = output / "owner_approach"
        approach = run_owner(scene_key, pose_mode, steps=60,
            device=inference_device(),
            native_source=native_source,
            person_style="scanned", gaze_route_arbitration=True,
            pose_crosscheck=pose_mode in ("rgbd_yaw", "rgbd_native_fallback"),
            person_masked_odometry=pose_mode in ("rgbd_yaw", "rgbd_native_fallback"),
            spatial_range_association=True, output_dir=approach_dir,
            near_body_monitor=near_body_monitor,
            owner_option_step=owner_option_step,
            save_raw_frames=owner_raw_frames,
            simulated_inference_delay_s=simulated_inference_delay_s,
            far_conversation_viewpoint=rerun_far_controller,
            depth_separated_overlap=depth_separated_overlap)
    else:
        approach_dir = (ROOT / placement["source_delayed_owner_trace"]).parent
        approach = {"outcome": json.loads((approach_dir / "trace.json").read_text())["outcome"]}
    trace = json.loads((approach_dir / "trace.json").read_text())
    last = trace["trace"][-1]
    stop_reason = last["planner_decision"].get("reason")
    compatible_far_stop = bool(
        (approach["outcome"] == "hold_viewpoint_far" and
         stop_reason in ("far_conversation_viewpoint_near_range_stop",
                         "learner_owner_viewpoint")) or
        (not rerun_far_controller and approach["outcome"] == "hold_request_help" and
         stop_reason == "near_range_before_owner_standoff"))
    far_view_eligible = bool(delayed_conversation_screen is not None and
        compatible_far_stop and placement["eligible_far_viewpoint"] and
        last["pose_healthy"] and
        1.45 <= float(last["owner_actor"]["local_depth_m"]) <= 1.65 and
        abs(float(last["owner_actor"]["owner_bearing_deg"])) <= 20. and
        float(last["planner_decision"]["nearest_forward_range_m"]) >= .9 and
        last["commanded_step_m"] == 0.)
    if approach["outcome"] != "hold_viewpoint" and not far_view_eligible:
        summary = {"schema": "ape.real-capture-owner-teach-find.v1",
            "scene": "ScanSceneverse/" + SCENES[scene_key]["name"],
            "status": "owner_viewpoint_not_reached",
            "simulated_inference_delay_s": simulated_inference_delay_s,
            "owner_approach_outcome": approach["outcome"],
            "owner_stop_reason": trace["trace"][-1]["planner_decision"].get("reason"),
            "owner_approach_video": str((approach_dir / "first_person.mp4").relative_to(ROOT)),
            "owner_approach_trace": str((approach_dir / "trace.json").relative_to(ROOT)),
            "teaching": None, "find_conditions": []}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return {"owner": approach["outcome"], "status": summary["status"],
                "stop_reason": summary["owner_stop_reason"]}
    cfg = SCENES[scene_key]
    source = json.loads(cfg["trace"].read_text())
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    scene = scene_from_capture_source(ROOT, source, scanned_person=human,
        person_facing_deg=cfg["person_facing_deg"])
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    teach_xz = tuple(placement["teach_target_xz"])
    teach_distractor_xz = tuple(placement["teach_distractor_xz"])
    search_xz = tuple(placement["find_target_xz"])
    search_distractor_xz = tuple(placement["find_distractor_xz"])
    if [camera_width, camera_height] != placement_screen_data.get(
            "camera_resolution_px", [640, 360]):
        raise ValueError("Camera resolution does not match frozen placement screen")
    teach_click = placement["click_xy_at_640x360" if (camera_width, camera_height)
                             == (640, 360) else "click_xy_at_camera_resolution"]
    target_prop = TexturedFloorObject(props / f"{target_name}.obj", scale=object_scale)
    distractor_prop = TexturedFloorObject(props / f"{distractor_name}.obj", scale=object_scale)
    teach_props = {target_name: (target_prop, teach_xz, 0.),
                   distractor_name: (distractor_prop, teach_distractor_xz, 0.)}
    scene.set_props(teach_props)
    for xz in (teach_xz, teach_distractor_xz, search_xz,
               search_distractor_xz):
        if not scene.floor_supported(*xz):
            raise RuntimeError(f"Object placement has no reconstructed floor: {xz}")
    actual_xy = np.asarray(last["evaluator_actual_xz_m"], dtype=float)
    endpoint_offset = float(np.linalg.norm(actual_xy - placement["owner_endpoint_xz"]))
    actual_yaw = float(last["evaluator_actual_heading_deg"])
    actor_xy = np.asarray(last["actor_pose_xy_m"], dtype=float)
    actor_yaw = float(last["actor_heading_deg"])
    teach_rgb, teach_depth, teach_ids, teach_ranges = camera(scene, actual_xy,
        actual_yaw, owner_pose(float(last["decision_time_s"] + .2), cfg),
        camera_width, camera_height)
    revalidated_geometry = None
    if endpoint_offset > .02:
        target_pixels = np.where(teach_ids == target_name)
        scene.set_props({target_name: (target_prop, search_xz, 0.),
                         distractor_name: (distractor_prop, search_distractor_xz, 0.)})
        search_counts = []
        for offset in SEARCH_YAW_OFFSETS_DEG:
            _, _, ids, _ = camera(scene, actual_xy, actual_yaw + offset, None,
                                  camera_width, camera_height)
            search_counts.append({"yaw_offset_deg": offset,
                "target_visible_px": int((ids == target_name).sum()),
                "distractor_visible_px": int((ids == distractor_name).sum())})
        scene.set_props(teach_props)
        revalidated_geometry = {"owner_endpoint_offset_m": endpoint_offset,
            "frozen_object_positions_unchanged": True,
            "teach_target_visible_px": int(len(target_pixels[0])),
            "teach_distractor_visible_px": int((teach_ids == distractor_name).sum()),
            "search_views": search_counts}
        supported = (len(target_pixels[0]) >= 400 and
            revalidated_geometry["teach_distractor_visible_px"] >= 150 and
            search_counts[0]["target_visible_px"] < 50 and
            search_counts[0]["distractor_visible_px"] >= 150 and
            max(x["target_visible_px"] for x in search_counts[1:]) >= 300)
        revalidated_geometry["supported"] = supported
        if not supported:
            summary = {"schema": "ape.real-capture-owner-teach-find.v1",
                "scene": "ScanSceneverse/" + cfg["name"],
                "status": "owner_viewpoint_outside_frozen_item_placement",
                "simulated_inference_delay_s": simulated_inference_delay_s,
                "owner_approach_outcome": approach["outcome"],
                "terminal_view_geometry_revalidation": revalidated_geometry,
                "owner_approach_video": str((approach_dir / "first_person.mp4").relative_to(ROOT)),
                "owner_approach_trace": str((approach_dir / "trace.json").relative_to(ROOT)),
                "teaching": None, "find_conditions": []}
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return {"owner": approach["outcome"], "status": summary["status"],
                    "terminal_view_geometry_revalidation": revalidated_geometry}
        teach_click = [int(np.median(target_pixels[1])),
                       int(np.median(target_pixels[0]))]
    teach_dir = output / "teach"
    teach_dir.mkdir(exist_ok=True)
    Image.fromarray(teach_rgb).save(teach_dir / "rgb_000.png")
    Image.fromarray(np.where(np.isfinite(teach_depth),
        np.clip(teach_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            teach_dir / "depth_000.png")
    vision = Vision("generic-proposal", device=inference_device())
    started = time.perf_counter()
    proposals = vision.detect(Image.fromarray(teach_rgb))
    selection, selection_info = vision.select_click(
        Image.fromarray(teach_rgb), teach_click, proposals)
    if selection is None:
        raise RuntimeError(f"Teaching click selected no visible proposal: {selection_info}")
    exemplar = vision.feature([Image.fromarray(teach_rgb).crop(tuple(selection))])[0]
    negatives = [p for p in proposals if _iou(p["box"], selection) < .25]
    negative_scores = ([float(f @ exemplar) for f in vision.feature([
        Image.fromarray(teach_rgb).crop(tuple(round(v) for v in p["box"]))
        for p in negatives])] if negatives else [])
    acceptance = max(.5, max(negative_scores, default=0.) + .15)
    colour_exemplar = (colour_histogram(teach_rgb, selection)
                       if identity_mode == "multi_cue" else None)
    negative_colour_scores = ([colour_similarity(colour_exemplar, teach_rgb,
        p["box"]) for p in negatives] if colour_exemplar is not None else [])
    colour_acceptance = (max(.5, max(negative_colour_scores, default=0.) + .10)
                         if colour_exemplar is not None else None)
    teaching_wall_s = time.perf_counter() - started
    teach_mask = teach_ids == target_name
    teaching = {"schema": "ape.real-capture-teaching.v1",
        "owner_approach_trace": str((approach_dir / "trace.json").relative_to(ROOT)),
        "capture_time_s": float(last["decision_time_s"] + .2),
        "source_rgb": str((teach_dir / "rgb_000.png").relative_to(ROOT)),
        "source_depth": str((teach_dir / "depth_000.png").relative_to(ROOT)),
        "scenario_click_xy": teach_click,
        "click_source": ("evaluator-selected visible prop pixel in actual stopped frame after frozen-position geometry revalidation"
                         if revalidated_geometry else
                         "evaluator-selected visible prop pixel in frozen placement screen"),
        "owner_confirmation": "scripted confirmed crop; no live owner input",
        "name_input": target_name,
        "proposals": proposals, "selected_box_xyxy": selection,
        "selection_info": selection_info,
        "distractor_similarities": negative_scores,
        "acceptance_similarity": acceptance,
        "identity_mode": identity_mode,
        "negative_colour_similarities": negative_colour_scores,
        "colour_acceptance_similarity": colour_acceptance,
        "inference_wall_s": teaching_wall_s,
        "rgb_sha256": hashlib.sha256(teach_rgb.tobytes()).hexdigest(),
        "range_low_m": [float(x) if np.isfinite(x) else None for x in teach_ranges[0]],
        "range_mid_m": [float(x) if np.isfinite(x) else None for x in teach_ranges[1]],
        "evaluator_target_visible_px": int(teach_mask.sum()),
        "evaluator_target_box_xyxy": bbox(teach_mask),
        "evaluator_selected_target_iou": _iou(selection, bbox(teach_mask))}
    # The evaluator supplies only the scripted owner's yes/no reply to the
    # highlighted crop. A rejected crop must not become the search exemplar.
    if teaching["evaluator_selected_target_iou"] < .25:
        teaching["owner_confirmation"] = "scripted owner rejected highlighted crop"
        teaching["owner_action"] = "ask_show_again"
        (teach_dir / "trace.json").write_text(json.dumps(teaching, indent=2) + "\n")
        summary = {"schema": "ape.real-capture-owner-teach-find.v1",
            "status": "owner_rejected_teaching_crop",
            "simulated_inference_delay_s": simulated_inference_delay_s,
            "scene": "ScanSceneverse/" + cfg["name"],
            "source_capture": str(cfg["trace"].relative_to(ROOT)),
            "scenario_placement_screen": str(placement_source.relative_to(ROOT)),
            "owner_approach_outcome": approach["outcome"],
            "owner_approach_video": str((approach_dir / "first_person.mp4").relative_to(ROOT)),
            "owner_approach_trace": str((approach_dir / "trace.json").relative_to(ROOT)),
            "search_photometry": search_photometry,
            "teaching": teaching, "find_conditions": [],
            "next_action": "ask_owner_to_show_item_again",
            "limits": "Scripted truthful owner; no search or identity claim follows a rejected teaching crop."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return {"owner": approach["outcome"], "status": summary["status"],
                "next_action": summary["next_action"]}
    (teach_dir / "trace.json").write_text(json.dumps(teaching, indent=2) + "\n")
    results = []
    for present in (True, False):
        result = run_search(scene, vision, actual_xy, actual_yaw, actor_xy,
            actor_yaw, exemplar, acceptance, present,
            output / ("find_present" if present else "find_absent"),
            start_time_s=float(last["decision_time_s"] + .4),
            search_xz=search_xz,
            search_distractor_xz=search_distractor_xz,
            owner_start_xz=tuple(cfg["start_xz"]),
            target_name=target_name, distractor_name=distractor_name,
            object_scale=object_scale, camera_width=camera_width,
            camera_height=camera_height, identity_mode=identity_mode,
            colour_exemplar=colour_exemplar,
            colour_acceptance=colour_acceptance,
            search_photometry=search_photometry)
        results.append(result)
    summary = {"schema": "ape.real-capture-owner-teach-find.v1",
        "status": ("teach_find_completed_far_viewpoint" if far_view_eligible
                   else "teach_find_completed"),
        "simulated_inference_delay_s": simulated_inference_delay_s,
        "depth_separated_person_overlap": depth_separated_overlap,
        "scene": "ScanSceneverse/" + cfg["name"],
        "source_capture": str(cfg["trace"].relative_to(ROOT)),
        "scenario_placement_screen": str(placement_source.relative_to(ROOT)),
        "terminal_view_geometry_revalidation": revalidated_geometry,
        "reused_delayed_owner_stop": far_view_eligible and not rerun_far_controller,
        "far_viewpoint_rule": ("torso depth 1.45-1.65 m, bearing <=20 deg, nearest forward range >=0.9 m, healthy pose, no forward step; retain near-body stop" if far_view_eligible else None),
        "person_source": "https://sketchfab.com/3d-models/sitting-person-0a9933b67fff41a3a29094676dbede9f",
        "objects_source": "HOPE textured grocery meshes, rendered into the real-capture room",
        "object_pair": {"target": target_name, "distractor": distractor_name,
                        "scale_to_metres": object_scale},
        "teach_find_camera_resolution_px": [camera_width, camera_height],
        "identity_mode": identity_mode,
        "search_photometry": search_photometry,
        "body": "0.42 m camera on 17 cm kinematic proxy, ideal yaw; no selected articulated design",
        "owner_approach_outcome": approach["outcome"],
        "owner_approach_video": str((approach_dir / "first_person.mp4").relative_to(ROOT)),
        "teaching": teaching,
        "find_conditions": results,
        "limits": ["manually supplied click and scripted confirmation",
                   "object relocated or removed instantaneously between phases",
                   "stationary kinematic camera yaw search; no physical head or grasp",
                   "textured room scale and collision remain provisional",
                   "development room and positions, no held-out generalization"] +
                  (["far conversation viewpoint is a post-stop development counterfactual, not the original owner's chosen success state"]
                   if far_view_eligible and not rerun_far_controller else [])}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    result = {"owner": approach["outcome"],
        "teach_iou": teaching["evaluator_selected_target_iou"],
        "acceptance": acceptance,
        "find": [{k: r[k] for k in ("target_present", "action",
            "best_similarity", "best_proposal_evaluator_target_iou",
            "target_visible_in_any_view", "wrong_confident_candidate")}
                 for r in results]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="capture_serving_south_2m")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--simulated-inference-delay-s", type=float, default=0.)
    parser.add_argument("--placement-screen", type=Path)
    parser.add_argument("--delayed-conversation-screen", type=Path)
    parser.add_argument("--rerun-far-controller", action="store_true")
    parser.add_argument("--depth-separated-overlap", action="store_true")
    parser.add_argument("--pose-mode", choices=("rgbd_yaw", "native_shaped",
                                                "rgbd_native_fallback"),
                        default="rgbd_yaw")
    parser.add_argument("--native-source", default="varied150-forward-seed144")
    parser.add_argument("--target-object", default=TARGET_NAME)
    parser.add_argument("--distractor-object", default=DISTRACTOR_NAME)
    parser.add_argument("--object-scale", type=float, default=.012)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--identity-mode", choices=("dino", "multi_cue"),
                        default="dino")
    parser.add_argument("--search-photometry", choices=("none", "cool_dim"),
                        default="none")
    args = parser.parse_args()
    tag = (f"_delay{round(1000 * args.simulated_inference_delay_s)}ms"
           if args.simulated_inference_delay_s else "")
    output = args.output or RUNS / (args.scene.removeprefix("capture_") + tag)
    print(json.dumps(run_integrated(args.scene, output,
        simulated_inference_delay_s=args.simulated_inference_delay_s,
        placement_screen=args.placement_screen,
        delayed_conversation_screen=args.delayed_conversation_screen,
        rerun_far_controller=args.rerun_far_controller,
        depth_separated_overlap=args.depth_separated_overlap,
        pose_mode=args.pose_mode, native_source=args.native_source,
        target_name=args.target_object, distractor_name=args.distractor_object,
        object_scale=args.object_scale, camera_width=args.camera_width,
        camera_height=args.camera_height, identity_mode=args.identity_mode,
        search_photometry=args.search_photometry), indent=2))


if __name__ == "__main__":
    main()
