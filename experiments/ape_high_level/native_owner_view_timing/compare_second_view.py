"""Compare second chest wait and head look at equal cumulative recovery time."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from experiments.ape_high_level.native_owner_recovery.run_recovery import (
    ROOT, CONFIG, SCENARIO, CHEST, HEAD, NativeShapedRoom, loop, sha,
    snapshot, restore, wrap, actor_view, advance_with_guard, annotate_evaluator,
    SOURCE_QPOS, SOURCE_ODOM)

DEFAULT_OUT = ROOT / "artifacts/ape-native-owner-view-timing/v1"


def run_second(scene: NativeShapedRoom, seed: int, mode: str,
               remembered: np.ndarray, vision: loop.Vision,
               exemplar: np.ndarray, folder: Path) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]]]:
    camera = CHEST if mode == "second_chest_wait" else HEAD
    angle = None
    if mode == "head_look":
        delta = remembered - scene.estimated[:2]
        angle = wrap(math.atan2(delta[1], delta[0]) - scene.estimated[2])
        bounds = scene.model.jnt_range[scene.model.joint("head_yaw").id]
        if not (abs(angle) <= 1.2 and bounds[0] <= angle <= bounds[1]):
            raise ValueError("Head bearing outside actor validity envelope")
    samples, frames = advance_with_guard(
        scene, seed, "chest_wait" if mode == "second_chest_wait" else "head_look", angle)
    observed, final_rgb = actor_view(scene, vision, exemplar, camera,
                                     remembered, folder, mode)
    frames.append((final_rgb, np.asarray(scene.capture_external())))
    row = {"mode": mode, "requested_head_yaw_rad": round(angle, 5) if angle is not None else None,
           "elapsed_cumulative_recovery_s": round(scene.time_s - (samples[0]["time_s"] - .4), 4),
           "range_samples": samples, "interrupted": scene.safety_stop_observation is not None,
           "actor_observation": observed,
           "evaluator": annotate_evaluator(scene, camera, observed["candidate"]),
           "path_min_clearance_evaluator_only_m": round(float(scene.path_min_clearance[0]), 4)}
    return row, frames


def render(episode: Path, first_frames: list[tuple[np.ndarray, np.ndarray]],
           branch_frames: dict[str, list[tuple[np.ndarray, np.ndarray]]],
           branches: dict[str, dict], output: Path) -> int:
    reader = cv2.VideoCapture(str(episode))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot read {episode}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             4., (1600, 520))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output}")
    count = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, "Original bounded owner route", (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 2)
        writer.write(np.vstack((strip, frame)))
        count += 1
    reader.release()
    for stage, index in ([('first chest recapture', i) for i in range(len(first_frames))] +
                         [('equal-time second action', i) for i in range(
                             max(len(x) for x in branch_frames.values()) + 6)]):
        if stage == 'first chest recapture':
            left = right = first_frames[index][0]
            external = first_frames[index][1]
            left_text = right_text = "first chest"
        else:
            left = branch_frames["second_chest_wait"][min(index, len(branch_frames["second_chest_wait"]) - 1)][0]
            right = branch_frames["head_look"][min(index, len(branch_frames["head_look"]) - 1)][0]
            external = branch_frames["head_look"][min(index, len(branch_frames["head_look"]) - 1)][1]
            left_text = "second chest wait"
            right_text = "head look"
        panel = np.zeros((480, 320, 3), dtype=np.uint8)
        panel[:240] = cv2.resize(cv2.cvtColor(external, cv2.COLOR_RGB2BGR), (320, 240))
        cv2.putText(panel, "Evaluator room", (8, 260), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (240, 240, 240), 1)
        if stage != 'first chest recapture':
            for j, mode in enumerate(("second_chest_wait", "head_look")):
                row = branches[mode]
                candidate = row["actor_observation"]["candidate"] is not None
                cv2.putText(panel, f"{mode}: {candidate}", (8, 305 + 55 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
        strip = np.zeros((40, 1600, 3), dtype=np.uint8)
        cv2.putText(strip, f"{stage} | left {left_text} | right {right_text}",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .63, (245, 245, 245), 2)
        writer.write(np.vstack((strip, np.hstack((
            cv2.cvtColor(left, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(right, cv2.COLOR_RGB2BGR), panel)))))
        count += 1
    writer.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    old_out = loop.OUT
    loop.OUT = out
    vision = loop.Vision("generic")
    teaching = loop.Room()
    try:
        teaching.teach()
        teach_rgb, _ = teaching.capture("upper_rgb", depth=False)
        box = teaching.truth_box("TomatoSauce", "upper_rgb")
        if box is None:
            raise RuntimeError("Teaching crop missing")
        exemplar = vision.feature([teach_rgb.crop(box)])[0]
    finally:
        teaching.close()
    scene = NativeShapedRoom()
    try:
        owner = loop.run_episode(scene, vision, exemplar, seed=args.seed,
                                 absent=False, config=CONFIG,
                                 scenario=SCENARIO, observer_video=True)
        episode = out / f"{SCENARIO}-seed{args.seed}-0-{CONFIG}-none"
        trace = json.loads((episode / "trace.json").read_text())
        memory = next((event["fresh_target_xy_m"]
                       for event in reversed(trace["events"])
                       if event.get("fresh_target_xy_m") is not None), None)
        if memory is None:
            raise RuntimeError("No actor image-derived memory")
        remembered = np.asarray(memory, dtype=float)
        first_samples, first_frames = advance_with_guard(scene, args.seed, "chest_wait")
        first, rgb = actor_view(scene, vision, exemplar, CHEST,
                                remembered, out, "first_chest")
        first_evaluator = annotate_evaluator(scene, CHEST, first["candidate"])
        first_frames.append((rgb, np.asarray(scene.capture_external())))
        if first["candidate"] is not None:
            raise RuntimeError("This seed does not need a second view")
        after_first = snapshot(scene)
        branches = {}
        branch_frames = {}
        for mode in ("second_chest_wait", "head_look"):
            restore(scene, after_first)
            row, frames = run_second(scene, args.seed, mode, remembered,
                                     vision, exemplar, out)
            branches[mode] = row
            branch_frames[mode] = frames
        video = out / f"seed{args.seed}-equal-time-second-view.mp4"
        frame_count = render(episode / "episode.mp4", first_frames,
                             branch_frames, branches, video)
        report = {"schema": "robot-learning-lab.owner-second-view-timing.v1",
                  "task": "T-282ec8", "seed": args.seed,
                  "owner_route_outcome": owner["outcome"],
                  "owner_trace": str((episode / "trace.json").relative_to(ROOT)),
                  "owner_trace_sha256": sha(episode / "trace.json"),
                  "owner_video_sha256": sha(episode / "episode.mp4"),
                  "remembered_xy_m": remembered.round(5).tolist(),
                  "first_chest": {"range_samples": first_samples,
                                  "actor_observation": first,
                                  "evaluator": first_evaluator},
                  "branches": branches,
                  "comparison_video": str(video.relative_to(ROOT)),
                  "comparison_video_sha256": sha(video),
                  "comparison_video_frames": frame_count,
                  "source_sha256": {"owner": sha(Path(loop.__file__)),
                                    "recovery_library": sha(ROOT / "experiments/ape_high_level/native_owner_recovery/run_recovery.py"),
                                    "timing_runner": sha(Path(__file__)),
                                    "native_qpos": sha(SOURCE_QPOS),
                                    "native_odometry": sha(SOURCE_ODOM)},
                  "limitation": "One furnished moving-room seed, synthetic cameras/range, bounded kinematic actions, scripted teaching crop. Ground truth excluded from actor proposal and branch choice; no physical camera selection."}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        for mode, row in branches.items():
            print(mode, row["actor_observation"]["status"],
                  row["evaluator"]["candidate_true_target_iou_evaluator_only"],
                  row["path_min_clearance_evaluator_only_m"], flush=True)
    finally:
        scene.close()
        loop.OUT = old_out


if __name__ == "__main__":
    main()
