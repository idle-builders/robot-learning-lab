"""V378: compare RGB-only camera-motion-compensated bearing to depth tracks.

This replays V374's retained accepted boxes and estimated camera rotations.
It measures continuity only; one candidate per view means it cannot test
crossing or same-looking distractors, depth order, or exact item identity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run import H, ROOT, W, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v374/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v378"
MAX_BEARING_STEP_RAD = .10
FOVY_DEG = 67.


def world_bearing(box: list[float] | None,
                  rotation: np.ndarray) -> np.ndarray | None:
    if box is None:
        return None
    focal_px = H / (2 * math.tan(math.radians(FOVY_DEG) / 2))
    u = (box[0] + box[2]) / 2
    v = (box[1] + box[3]) / 2
    camera_ray = np.asarray(((u - W / 2) / focal_px,
                             -(v - H / 2) / focal_px, -1.))
    direction = rotation @ camera_ray
    return direction / np.linalg.norm(direction)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(INPUT.read_text())
    rows = []
    for original in source["rows"]:
        bearings = []
        for sample in original["samples"]:
            with np.load(ROOT / sample["actor_packet_path"]) as packet:
                rotation = packet["camera_rotation_estimated"].copy()
            bearings.append(world_bearing(
                sample["actor_choice"]["accepted_box"], rotation))
        pairs = []
        for index, (a, b) in enumerate(zip(bearings[:-1], bearings[1:])):
            angle = (None if a is None or b is None else
                     math.acos(float(np.clip(a @ b, -1., 1.))))
            pairs.append({
                "first_frame": index,
                "actor_world_bearing_step_rad": (
                    None if angle is None else round(angle, 5)),
                "actor_bearing_link": (
                    angle is not None and angle <= MAX_BEARING_STEP_RAD),
                "evaluator_both_boxes_true_can": (
                    original["samples"][index][
                        "evaluator_accepted_presented_can"] and
                    original["samples"][index + 1][
                        "evaluator_accepted_presented_can"]),
            })
        rows.append({
            "case": original["case"],
            "actor_rgb_bearing_links": sum(
                pair["actor_bearing_link"] for pair in pairs),
            "actor_depth_3d_links": original["actor_robust_3d_links"],
            "actor_image_iou_links": original[
                "actor_consecutive_box_iou_ge_0p6"],
            "evaluator_wrong_bearing_links": sum(
                pair["actor_bearing_link"] and not
                pair["evaluator_both_boxes_true_can"] for pair in pairs),
            "video_path": original["video_path"],
            "video_sha256": original["video_sha256"],
            "pairs": pairs,
        })
        print(original["case"], "bearing", rows[-1][
            "actor_rgb_bearing_links"], "depth", rows[-1][
            "actor_depth_3d_links"], "IoU", rows[-1][
            "actor_image_iou_links"], flush=True)
    report = {
        "scope": "Same V374 four independent-hand videos and accepted boxes. RGB-only world-bearing continuity uses the actor camera-rotation estimate, including simulated head qpos with no encoder noise/delay, and a 0.10rad adjacent-frame gate; compares with the already-scored robust depth 3D and image-box IoU links. The bearing threshold is a development value chosen after V374 was visible, so this is an ablation, not held-out validation. One visible taught candidate per frame and scripted owner answer cannot assess duplicate/crossing objects or metric proximity; no sensor selection follows.",
        "source_sha256": sha(SOURCE), "input_report_sha256": sha(INPUT),
        "max_bearing_step_rad": MAX_BEARING_STEP_RAD,
        "rows": rows,
    }
    (OUT / "rgb-bearing-track-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
