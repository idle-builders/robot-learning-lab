"""V387: KLT RGB persistence after actor-only moving-item bootstrap.

The starting box is V383's first uniquely moving track, not evaluator truth.
Subsequent images use no head angle, camera pose, range or DINO refresh.
Evaluator boxes score drift and are drawn in the audit video only.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from run import ROOT, iou, sha


SOURCE = Path(__file__)
TRACK_INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v383/multi-candidate-motion-audit.json"
VIDEO_INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v381/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v387"
MIN_FEATURES = 8
BACKTRACK_ERROR_PX = 20.


def frame(sample: dict) -> np.ndarray:
    with np.load(ROOT / sample["actor_packet_path"]) as packet:
        return packet["exposed_rgb"].copy()


def case(source: dict, bootstrap: dict) -> dict:
    first = next(row for row in bootstrap["frames"]
                 if row["actor_selected_box"] is not None)
    first_index = first["frame"]
    box = np.asarray(first["actor_selected_box"], dtype=float)
    branch = OUT / source["case"]
    branch.mkdir(parents=True, exist_ok=True)
    rgb = frame(source["samples"][first_index])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x0, y0, x1, y1 = np.rint(box).astype(int)
    mask[max(0, y0):min(mask.shape[0], y1),
         max(0, x0):min(mask.shape[1], x1)] = 255
    points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=40,
                                     qualityLevel=.005, minDistance=2)
    if points is None or len(points) < MIN_FEATURES:
        raise RuntimeError("Actor bootstrap lacks KLT features")
    rows = []
    video = branch / "rgb-flow-track.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264",
                            quality=7) as writer:
        for index in range(first_index, len(source["samples"])):
            if index > first_index:
                next_rgb = frame(source["samples"][index])
                next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
                forward, status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, next_gray, points, None, winSize=(31, 31),
                    maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS |
                              cv2.TERM_CRITERIA_COUNT, 40, .01))
                backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                    next_gray, gray, forward, None, winSize=(31, 31),
                    maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS |
                              cv2.TERM_CRITERIA_COUNT, 40, .01))
                old = points.reshape(-1, 2)
                new = forward.reshape(-1, 2)
                returned = backward.reshape(-1, 2)
                good = (status.reshape(-1).astype(bool) &
                        reverse_status.reshape(-1).astype(bool) &
                        (np.linalg.norm(returned - old, axis=1) <=
                         BACKTRACK_ERROR_PX))
                if int(good.sum()) < MIN_FEATURES:
                    rows.append({"frame": index, "actor_status":
                                 "insufficient_flow_features",
                                 "actor_feature_count": int(good.sum()),
                                 "actor_box": None,
                                 "evaluator_iou": 0.})
                    break
                shift = np.median(new[good] - old[good], axis=0)
                box += np.asarray((shift[0], shift[1], shift[0], shift[1]))
                points = new[good].reshape(-1, 1, 2).astype(np.float32)
                gray, rgb = next_gray, next_rgb
            truth = source["samples"][index]["evaluator_truth_box"]
            overlap = iou(box.tolist(), truth)
            rows.append({
                "frame": index,
                "actor_status": "tracked",
                "actor_feature_count": int(len(points)),
                "actor_box": np.round(box, 3).tolist(),
                "evaluator_iou": round(float(overlap), 5),
                "evaluator_held_can_match": overlap >= .25,
            })
            view = Image.fromarray(rgb)
            draw = ImageDraw.Draw(view)
            draw.rectangle(tuple(box), outline=(31, 230, 79), width=3)
            if truth is not None:
                draw.rectangle(tuple(truth), outline=(44, 145, 255), width=2)
            draw.rectangle((0, 0, 450, 26), fill=(0, 0, 0))
            draw.text((5, 6), f"RGB flow frame{index} green=actor blue=score "
                      f"IoU{overlap:.2f}", fill=(255, 255, 255))
            writer.append_data(np.asarray(view))
    return {
        "case": source["case"],
        "actor_bootstrap_frame": first_index,
        "actor_initial_feature_count": int(len(
            cv2.goodFeaturesToTrack(cv2.cvtColor(frame(
                source["samples"][first_index]), cv2.COLOR_RGB2GRAY),
                mask=mask, maxCorners=40, qualityLevel=.005,
                minDistance=2))),
        "actor_tracked_frames": sum(row["actor_status"] == "tracked"
                                    for row in rows),
        "evaluator_held_matches": sum(row.get("evaluator_held_can_match",
                                               False) for row in rows),
        "video_path": str(video.relative_to(ROOT)),
        "video_sha256": sha(video),
        "frames": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads(VIDEO_INPUT.read_text())["rows"]
    bootstrap = [row for row in json.loads(TRACK_INPUT.read_text())["rows"]
                 if "v382" in row["source_candidate_report"]]
    rows = [case(source, next(row for row in bootstrap
                              if row["case"] == source["case"]))
            for source in sources]
    for row in rows:
        print(row["case"], "boot", row["actor_bootstrap_frame"],
              "tracked", row["actor_tracked_frames"],
              "matched", row["evaluator_held_matches"], flush=True)
    report = {
        "scope": "Actor-only V383 moving-item box initializes OpenCV pyramidal Lucas-Kanade sparse RGB flow at its first accepted raised-replica frame. The next exposed RGB frames track those points with a permissive forward-backward <=20px check and >=8 features; median flow translates the box. A conventional strict1.5px check left only0-2 points on the first blurred transition and abstained, so20px is a revealed-data development choice, not a qualified confidence bound. No head pose, metric depth or detector refresh after bootstrap. Evaluator held-can boxes score overlap and appear only as blue overlay in the audit video. Same two revealed V381 sequences; this does not test occlusion, blur outside the scripted 20ms exposure, camera/body motion after bootstrap, drift across long gaps or real inference timing.",
        "source_sha256": sha(SOURCE),
        "track_input_sha256": sha(TRACK_INPUT),
        "video_input_sha256": sha(VIDEO_INPUT),
        "rows": rows,
    }
    (OUT / "rgb-flow-track-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
