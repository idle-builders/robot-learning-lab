"""Execute a conditional post-stop owner recovery turn in the furnished ape room.

The original owner actor is imported unchanged. Ground-truth labels are used
only for interventions, scripted owner replies after an offer, and scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
from experiments.ape_high_level.closed_loop import run as loop  # noqa: E402
from experiments.ape_high_level.native_owner_bridge.replay_owner_packets import (  # noqa: E402
    NativeShapedRoom, SOURCE_ODOM, SOURCE_QPOS)
from experiments.ape_high_level.native_owner_reobserve.compare_post_stop_views import (  # noqa: E402
    snapshot, restore, wrap)

DEFAULT_OUT = ROOT / "artifacts/ape-native-owner-recovery/v1"
CONFIG = "chest_rgbd_range_guard_scan_memory"
SCENARIO = "furnished_moving"
CHEST = "chest_low50_x45"
HEAD = "lower_rgb"
CASES = ("present", "moved_target", "lookalike_replacement")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_view(scene: NativeShapedRoom, vision: loop.Vision,
               exemplar: np.ndarray, camera: str,
               remembered_xy: np.ndarray, folder: Path, stem: str) -> tuple[dict, np.ndarray]:
    """Rank fresh image proposals using only estimated pose and aligned depth."""
    rgb, depth = scene.capture(camera, depth=True)
    if depth is None or not np.isfinite(depth).any():
        return {"camera": camera, "capture_time_s": round(scene.time_s, 4),
                "status": "depth_unavailable", "candidate": None}, np.asarray(rgb)
    rgb_path = folder / f"{stem}-rgb.png"
    depth_path = folder / f"{stem}-depth.npy"
    rgb.save(rgb_path)
    np.save(depth_path, depth)
    proposals = vision.detect(rgb)
    scores = (vision.feature([rgb.crop(item["box"]) for item in proposals]) @ exemplar
              if proposals else np.empty(0))
    ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]),
                    reverse=True)
    pose = scene.camera_pose_estimate(camera)
    options = []
    candidate = None
    for proposal, score in ranked:
        located = loop.locate_world(proposal["box"], pose, depth)
        distance = (float(np.linalg.norm(np.asarray(located[:2]) - remembered_xy))
                    if located is not None else None)
        option = {"box_xyxy_px": [round(float(x), 2) for x in proposal["box"]],
                  "similarity": round(float(score), 4),
                  "located_xyz_m": ([round(float(x), 4) for x in located]
                                    if located is not None else None),
                  "distance_from_remembered_xy_m": (round(distance, 4)
                                                     if distance is not None else None)}
        options.append(option)
        if candidate is None and score >= .35 and distance is not None and distance <= .12:
            candidate = option
    return {
        "camera": camera, "capture_time_s": round(scene.time_s, 4),
        "estimated_pose_at_capture_xyyaw": scene.estimated.round(5).tolist(),
        "head_yaw_at_capture_rad": round(float(scene.head), 5),
        "status": "candidate" if candidate else "no_metric_candidate",
        "ranked_proposals": options, "candidate": candidate,
        "rgb": str(rgb_path.relative_to(ROOT)), "rgb_sha256": sha(rgb_path),
        "depth": str(depth_path.relative_to(ROOT)), "depth_sha256": sha(depth_path),
    }, np.asarray(rgb)


def advance_with_guard(scene: NativeShapedRoom, seed: int, action: str,
                       angle: float | None = None) -> tuple[list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    """Sample the existing torso-range change guard every 100 ms of an action."""
    guard = loop.FastRangeGuard(band_m=.05)
    samples: list[dict] = []
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    serial = 0

    def scan() -> dict:
        nonlocal serial
        serial += 1
        ranges = loop.sensed_range(scene, seed, 60000 + serial)
        result = guard.update(ranges, scene.estimated, scene.time_s)
        return {"time_s": round(scene.time_s, 4),
                "estimated_pose_xyyaw": scene.estimated.round(5).tolist(),
                "range_72_beams_m": ranges, "guard": result}

    samples.append({"phase": "before", **scan()})

    def probe(substep: int, fraction: float) -> dict | None:
        packet = scan()
        packet.update({"phase": "during", "substep": substep,
                       "fraction": round(fraction, 3)})
        samples.append(packet)
        camera = HEAD if action == "head_look" else CHEST
        rgb, _ = scene.capture(camera, depth=False)
        external = scene.capture_external()
        frames.append((np.asarray(rgb).copy(), np.asarray(external).copy()))
        if packet["guard"]["status"] == "near_body_change":
            return {**packet["guard"], "range_72_beams_m": packet["range_72_beams_m"],
                    "capture_time_s": packet["time_s"], "action_fraction": fraction}
        return None

    if action == "chest_wait":
        scene.wait(safety_probe=probe)
    elif action == "head_look" and angle is not None:
        scene.look(angle, HEAD, safety_probe=probe)
    else:
        raise ValueError(action)
    return samples, frames


def annotate_evaluator(scene: NativeShapedRoom, camera: str,
                       candidate: dict | None) -> dict:
    truth = scene.truth_box("TomatoSauce", camera)
    box = candidate["box_xyxy_px"] if candidate else None
    return {"true_target_box_evaluator_only": truth,
            "true_target_pixel_area_evaluator_only":
                (truth[2] - truth[0]) * (truth[3] - truth[1]) if truth else 0,
            "candidate_true_target_iou_evaluator_only":
                round(loop.iou(box, truth), 4) if box is not None else None,
            "minimum_clearance_evaluator_only_m":
                round(float(scene.obstacle_clearance()[0]), 4)}


def restore_case(scene: NativeShapedRoom, base: dict,
                 original_positions: dict, case: str, remembered_xy: np.ndarray) -> None:
    restore(scene, base)
    scene.positions = original_positions.copy()
    for name, xy in original_positions.items():
        scene.model.body_pos[scene.model.body(f"loop_{name}").id] = (*xy, .052)
    if case in ("moved_target", "lookalike_replacement"):
        relocated_xy = (.83, -.45 if remembered_xy[1] > 0 else .45)
        scene.positions["TomatoSauce"] = relocated_xy
        scene.model.body_pos[scene.model.body("loop_TomatoSauce").id] = (*relocated_xy, .052)
    if case == "lookalike_replacement":
        xy = tuple(float(x) for x in remembered_xy)
        scene.positions["Parmesan"] = xy
        scene.model.body_pos[scene.model.body("loop_Parmesan").id] = (*xy, .052)
    scene._forward()


def run_turn(scene: NativeShapedRoom, seed: int, case: str,
             remembered_xy: np.ndarray, vision: loop.Vision,
             exemplar: np.ndarray, folder: Path) -> tuple[dict, list[tuple[str, np.ndarray, np.ndarray]]]:
    folder.mkdir(parents=True, exist_ok=True)
    actions: list[dict] = []
    frames: list[tuple[str, np.ndarray, np.ndarray]] = []
    start_time = scene.time_s
    initial_range = loop.sensed_range(scene, seed, 61000)
    range_healthy = sum(x is not None for x in initial_range) >= 10
    pose_healthy = bool(np.isfinite(scene.estimated).all())
    if not range_healthy or not pose_healthy:
        response = "ask_owner_to_show_again"
        offer = None
    else:
        samples, action_frames = advance_with_guard(scene, seed, "chest_wait")
        frames.extend(("stationary chest recapture", *pair) for pair in action_frames)
        interrupted = scene.safety_stop_observation is not None or scene.motion_interrupted
        observed, final_rgb = actor_view(scene, vision, exemplar, CHEST,
                                         remembered_xy, folder, "chest")
        frames.append(("chest decision", final_rgb, np.asarray(scene.capture_external())))
        actions.append({"action": "chest_wait", "range_samples": samples,
                        "interrupted": interrupted, "observation": observed,
                        "evaluator": annotate_evaluator(scene, CHEST, observed["candidate"])})
        offer = observed["candidate"] if not interrupted else None
        if offer is None and not interrupted:
            delta = remembered_xy - scene.estimated[:2]
            angle = wrap(math.atan2(delta[1], delta[0]) - scene.estimated[2])
            joint = scene.model.joint("head_yaw")
            limits = scene.model.jnt_range[joint.id]
            if np.linalg.norm(delta) > .05 and abs(angle) <= 1.2 and limits[0] <= angle <= limits[1]:
                samples, action_frames = advance_with_guard(scene, seed, "head_look", angle)
                frames.extend(("head look to remembered bearing", *pair)
                              for pair in action_frames)
                interrupted = scene.safety_stop_observation is not None or scene.motion_interrupted
                observed, final_rgb = actor_view(scene, vision, exemplar, HEAD,
                                                 remembered_xy, folder, "head")
                frames.append(("head decision", final_rgb,
                               np.asarray(scene.capture_external())))
                actions.append({"action": "head_look", "requested_head_yaw_rad": round(angle, 5),
                                "range_samples": samples, "interrupted": interrupted,
                                "observation": observed,
                                "evaluator": annotate_evaluator(scene, HEAD, observed["candidate"])})
                offer = observed["candidate"] if not interrupted else None
            else:
                actions.append({"action": "head_look_skipped",
                                "reason": "remembered_bearing_out_of_joint_or_validity_envelope",
                                "computed_head_yaw_rad": round(angle, 5)})
        response = "offer_candidate_for_owner_confirmation" if offer else "ask_owner_to_show_again"
    offered_action = next((action for action in reversed(actions)
                           if action.get("observation", {}).get("candidate") is offer
                           and offer is not None), None)
    evaluator_iou = (offered_action["evaluator"]["candidate_true_target_iou_evaluator_only"]
                     if offered_action else None)
    owner_reply = ("yes" if evaluator_iou is not None and evaluator_iou >= .5 else
                   "no_show_again" if offer else None)
    return ({"seed": seed, "condition": case,
             "stop_time_s": round(start_time, 4),
             "remembered_target_xy_m": remembered_xy.round(5).tolist(),
             "initial_range_72_beams_m": initial_range,
             "pose_healthy": pose_healthy, "range_healthy": range_healthy,
             "actions": actions, "actor_response": response,
             "owner_reply_scripted_after_offer": owner_reply,
             "owner_confirmed_reacquisition": owner_reply == "yes",
             "false_candidate_offer_evaluator_only": bool(offer and owner_reply != "yes"),
             "head_actions": sum(action["action"] == "head_look" for action in actions),
             "elapsed_recovery_s": round(scene.time_s - start_time, 4),
             "path_min_clearance_evaluator_only_m": round(float(scene.path_min_clearance[0]), 4),
             "actual_pose_evaluator_only": scene.actual.round(5).tolist(),
             "estimated_pose": scene.estimated.round(5).tolist()}, frames)


def render_video(source_path: Path, frames: list[tuple[str, np.ndarray, np.ndarray]],
                 output: Path, case: str) -> int:
    source = cv2.VideoCapture(str(source_path))
    if not source.isOpened():
        raise RuntimeError(f"Cannot decode {source_path}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 4.,
                             (1600, 520))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot encode {output}")
    count = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        if frame.shape != (480, 1600, 3):
            raise ValueError(f"Unexpected owner video shape {frame.shape}")
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, "Original bounded owner route", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        writer.write(np.vstack((strip, frame)))
        count += 1
    source.release()
    for label, actor_rgb, external_rgb in frames + frames[-1:] * 6:
        panel = np.zeros((480, 320, 3), dtype=np.uint8)
        cv2.putText(panel, case.replace("_", " "), (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (245, 245, 245), 1)
        cv2.putText(panel, label[:30], (12, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1)
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, "Executed recovery | actor RGB-D camera, evaluator room view",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 2)
        writer.write(np.vstack((strip, np.hstack((
            cv2.cvtColor(actor_rgb, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR), panel)))))
        count += 1
    writer.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[85, 86])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old_out = loop.OUT
    vision = loop.Vision("generic")
    teaching = loop.Room()
    try:
        teaching.teach()
        taught_rgb, _ = teaching.capture("upper_rgb", depth=False)
        box = teaching.truth_box("TomatoSauce", "upper_rgb")
        if box is None:
            raise RuntimeError("Scripted teaching object invisible")
        exemplar = vision.feature([taught_rgb.crop(box)])[0]
        taught_rgb.save(out / "scripted-teach-rgb.png")
    finally:
        teaching.close()
    rows = []
    try:
        for seed in args.seeds:
            folder = out / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            loop.OUT = folder
            scene = NativeShapedRoom()
            try:
                present = loop.run_episode(scene, vision, exemplar, seed=seed,
                                           absent=False, config=CONFIG,
                                           scenario=SCENARIO, observer_video=True)
                episode = folder / f"{SCENARIO}-seed{seed}-0-{CONFIG}-none"
                trace = json.loads((episode / "trace.json").read_text())
                remembered = next((event["fresh_target_xy_m"]
                                   for event in reversed(trace["events"])
                                   if event.get("fresh_target_xy_m") is not None), None)
                if remembered is None:
                    raise RuntimeError(f"No actor target memory for seed {seed}")
                remembered_xy = np.asarray(remembered, dtype=float)
                base = snapshot(scene)
                original_positions = scene.positions.copy()
                for case in CASES:
                    restore_case(scene, base, original_positions, case, remembered_xy)
                    case_folder = folder / case
                    row, frames = run_turn(scene, seed, case, remembered_xy,
                                           vision, exemplar, case_folder)
                    movie = case_folder / "executed-owner-recovery.mp4"
                    row["video_frames"] = render_video(episode / "episode.mp4",
                                                        frames, movie, case)
                    row["video"] = str(movie.relative_to(ROOT))
                    row["video_sha256"] = sha(movie)
                    row["owner_episode_outcome"] = present["outcome"]
                    row["owner_trace"] = str((episode / "trace.json").relative_to(ROOT))
                    row["owner_trace_sha256"] = sha(episode / "trace.json")
                    rows.append(row)
                    print(seed, case, row["actor_response"], row["head_actions"],
                          row["owner_reply_scripted_after_offer"], flush=True)
                absent = loop.run_episode(scene, vision, exemplar, seed=seed,
                                          absent=True, config=CONFIG,
                                          scenario=SCENARIO, observer_video=True)
                absent_episode = folder / f"{SCENARIO}-seed{seed}-1-{CONFIG}-none"
                absent_trace = json.loads((absent_episode / "trace.json").read_text())
                absent_memory = next((event["fresh_target_xy_m"]
                                      for event in reversed(absent_trace["events"])
                                      if event.get("fresh_target_xy_m") is not None), None)
                rows.append({"seed": seed, "condition": "absent",
                             "owner_episode_outcome": absent["outcome"],
                             "owner_trace": str((absent_episode / "trace.json").relative_to(ROOT)),
                             "owner_trace_sha256": sha(absent_episode / "trace.json"),
                             "owner_video": str((absent_episode / "episode.mp4").relative_to(ROOT)),
                             "owner_video_sha256": sha(absent_episode / "episode.mp4"),
                             "actor_memory_available": absent_memory is not None,
                             "actor_response": "ask_owner_to_show_again",
                             "post_stop_actions": 0, "false_candidate_offer_evaluator_only": False})
                print(seed, "absent", absent["outcome"], flush=True)
            finally:
                scene.close()
    finally:
        loop.OUT = old_out
    report = {"schema": "robot-learning-lab.native-owner-recovery.v1",
              "task": "T-eqysqb", "scenario": SCENARIO,
              "scope": "Executed conditional post-stop view policy on an unchanged kinematic furnished owner actor with native-shaped odometry error; scripted owner reply is evaluator-guided only after an actor candidate offer. Synthetic camera/range/IMU and body; no physical sensor qualification or autonomous exact-item identification.",
              "source_sha256": {"owner_actor": sha(Path(loop.__file__)),
                                "recovery": sha(Path(__file__)),
                                "native_adapter": sha(ROOT / "experiments/ape_high_level/native_owner_bridge/replay_owner_packets.py"),
                                "native_qpos": sha(SOURCE_QPOS),
                                "native_odometry": sha(SOURCE_ODOM)},
              "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
