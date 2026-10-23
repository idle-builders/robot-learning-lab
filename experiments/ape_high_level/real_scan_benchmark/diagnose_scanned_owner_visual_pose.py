"""Replay a dynamic-person mask against the saved failed RGB-D pose segments.

This is a diagnostic on the same captured images, not a closed-loop rerun. The
mask comes only from the owner actor's saved selected person box. Truth is used
after estimation to score pose drift.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from measure_real_scan_visual_odometry import feature_packet
from run_real_capture_owner_loop import masked_feature_packet, odometry_delta


ROOT = Path(__file__).resolve().parents[3]
BATCH = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/owner_batch_rgbd_yaw_arbitrated_crosschecked_v1.json"
OUT = BATCH.parent / "person_mask_visual_pose_diagnostic_v1.json"


def packet(rgb: np.ndarray, depth: np.ndarray, orb: cv2.ORB,
           social: dict, mask_person: bool) -> tuple[dict, int]:
    return (masked_feature_packet(rgb, depth, orb, social) if mask_person else
            (feature_packet(rgb, depth, orb), 0))


def replay(trace: dict, folder: Path, mask_person: bool,
           start_xz: np.ndarray) -> dict:
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    height, width = 360, 640
    focal = height / (2 * math.tan(math.radians(75.) / 2))
    k = np.array([[focal, 0., (width - 1) / 2],
                  [0., focal, (height - 1) / 2], [0., 0., 1.]], float)
    pose = np.zeros(2)
    start = start_xz
    previous_packet = None
    rows = []
    for i, frame in enumerate(trace["trace"]):
        rgb = np.asarray(Image.open(folder / f"rgb_{i:03d}.png").convert("RGB"))
        depth = np.asarray(Image.open(folder / f"depth_{i:03d}.png"), np.float32) / 1000
        current, area = packet(rgb, depth, orb, frame["owner_actor"], mask_person)
        evidence = None
        if previous_packet is not None:
            previous = trace["trace"][i - 1]
            if not (previous["commanded_yaw_deg"] and not previous["commanded_step_m"]):
                delta, evidence = odometry_delta(previous_packet, current, k, matcher)
                if delta is not None:
                    heading = math.radians(previous["actor_heading_deg"])
                    right = np.array([-math.cos(heading), math.sin(heading)])
                    forward = np.array([math.sin(heading), math.cos(heading)])
                    pose += delta[0, 3] * right + delta[2, 3] * forward
        true_capture = (np.asarray(trace["trace"][i - 1]["evaluator_actual_xz_m"])
                        if i else start) - start
        rows.append({"frame": i, "mask_area_px": area,
                     "estimated_xy_m": pose.tolist(),
                     "evaluator_capture_xy_m": true_capture.tolist(),
                     "error_m": float(np.linalg.norm(pose - true_capture)),
                     "pose_update_accepted": None if evidence is None else evidence["accepted"],
                     "pose_update_inliers": None if evidence is None else evidence["forward"].get("inliers")})
        previous_packet = current
    return {"max_error_m": max(r["error_m"] for r in rows),
            "final_error_m": rows[-1]["error_m"],
            "accepted_updates": sum(r["pose_update_accepted"] is True for r in rows),
            "rejected_updates": sum(r["pose_update_accepted"] is False for r in rows),
            "frames": rows}


def main() -> None:
    batch = json.loads(BATCH.read_text())
    placement = json.loads((BATCH.parent / "screen.json").read_text())
    starts = {r["scene_id"]: np.asarray(r["start_xz"], float)
              for r in placement["scenes"] if "start_xz" in r}
    scenes = []
    for row in batch["scenes"]:
        if row["task_status"] != "completed" or row["frames"] < 2:
            continue
        trace = json.loads((ROOT / row["trace"]).read_text())
        folder = (ROOT / row["trace"]).parent
        full = replay(trace, folder, False, starts[row["scene_id"]])
        masked = replay(trace, folder, True, starts[row["scene_id"]])
        scenes.append({"scene_id": row["scene_id"], "name": row["name"],
                       "full_image": full, "owner_masked": masked})
        print(row["name"], round(full["max_error_m"], 3),
              "->", round(masked["max_error_m"], 3), flush=True)
    OUT.write_text(json.dumps({"schema": "ape.person-masked-visual-pose-diagnostic.v1",
        "scope": "offline replay of closed-loop images captured before original stop; no rerun of actions",
        "scene_count": len(scenes), "scenes": scenes}, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
