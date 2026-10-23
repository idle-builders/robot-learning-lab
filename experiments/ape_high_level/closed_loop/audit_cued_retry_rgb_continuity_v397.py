"""Test RGB-only short continuity after a V393 metric moving-item bootstrap.

The saved RGB packets are reprocessed with frozen DINO/hue proposals. A
KLT prediction reanchors to a unique nearby proposal; no depth or head pose
enters those updates. Occlusion or ambiguous proposals cause abstention.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from audit_cued_retry_show_v395 import REPLICA, expected_held
from audit_direct_structural_entry_v271 import V149, V150
from audit_rgb_flow_proposal_refresh_v388 import (
    BACKTRACK_ERROR_PX, MAX_BOX_CENTER_GAP_PX, MIN_FEATURES,
    appearance_boxes, center, features)
from gap_latched_candidate_tracks import GapLatchedCandidateTracks
from run import ROOT, Vision, iou, sha
from run_live_cued_retry_show_v393 import OUT as INPUT_ROOT


SOURCE = Path(__file__)
INPUT = INPUT_ROOT / "cued-retry-show-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v397"
BOOTSTRAP_INDEX = 8


def rgb(sample: dict) -> np.ndarray:
    with np.load(ROOT / sample["actor_packet_path"]) as packet:
        return packet["exposed_rgb"].copy()


def bootstrap_box(samples: list[dict]) -> np.ndarray | None:
    tracker = GapLatchedCandidateTracks()
    selected = None
    for index in range(BOOTSTRAP_INDEX + 1):
        decision = tracker.update(
            index, samples[index]["actor_choice"]["actor_all_candidates"])
        if index == BOOTSTRAP_INDEX:
            selected = decision["selected_candidate"]
    return None if selected is None else np.asarray(selected["box"], dtype=float)


def evaluator_label(mode: str, sample: dict, start_s: float,
                    box: np.ndarray) -> tuple[str, float]:
    ranked = sorted(((iou(box.tolist(), candidate["box"]), candidate)
                     for candidate in sample["actor_choice"][
                         "actor_all_candidates"]), key=lambda row: row[0])
    if not ranked or ranked[-1][0] < .25:
        return "unknown", 0.
    overlap, candidate = ranked[-1]
    point = np.asarray(candidate["world_xyz_m"])
    elapsed = float(sample["capture_time_s"]) - start_s
    held_error = float(np.linalg.norm(point - expected_held(mode, elapsed)))
    replica_error = float(np.linalg.norm(point - REPLICA))
    return ("held_item" if held_error < replica_error else
            "raised_replica"), float(overlap)


def replay(mode: str, samples: list[dict], vision: Vision,
           exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    box = bootstrap_box(samples)
    if box is None:
        return {"bootstrap_status": "no_unique_metric_track",
                "frames": []}
    first_rgb = rgb(samples[BOOTSTRAP_INDEX])
    gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    points = features(gray, box)
    if points is None or len(points) < MIN_FEATURES:
        return {"bootstrap_status": "insufficient_rgb_features",
                "frames": []}
    start_s = float(samples[0]["capture_time_s"])
    rows = []
    for index in range(BOOTSTRAP_INDEX, len(samples)):
        sample = samples[index]
        if index > BOOTSTRAP_INDEX:
            next_rgb = rgb(sample)
            next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
            forward, status, _ = cv2.calcOpticalFlowPyrLK(
                gray, next_gray, points, None, winSize=(31, 31), maxLevel=4)
            backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                next_gray, gray, forward, None, winSize=(31, 31), maxLevel=4)
            old = points.reshape(-1, 2)
            new = forward.reshape(-1, 2)
            error = np.linalg.norm(backward.reshape(-1, 2) - old, axis=1)
            good = (status.reshape(-1).astype(bool) &
                    reverse_status.reshape(-1).astype(bool) &
                    (error <= BACKTRACK_ERROR_PX))
            predicted = box.copy()
            if int(good.sum()) >= MIN_FEATURES:
                delta = np.median(new[good] - old[good], axis=0)
                predicted += np.asarray((delta[0], delta[1],
                                         delta[0], delta[1]))
            candidates = appearance_boxes(vision, exemplar, owner_hist,
                                          next_rgb)
            ranked = sorted((float(np.linalg.norm(center(np.asarray(c)) -
                                                   center(predicted))), c)
                            for c in candidates)
            if (not ranked or ranked[0][0] > MAX_BOX_CENTER_GAP_PX or
                    (len(ranked) > 1 and
                     ranked[1][0] <= MAX_BOX_CENTER_GAP_PX and
                     ranked[1][0] - ranked[0][0] < 10.)):
                rows.append({"index": index, "status": "no_unique_rgb_box",
                             "appearance_candidates": len(candidates)})
                break
            box = np.asarray(ranked[0][1], dtype=float)
            points = features(next_gray, box)
            if points is None or len(points) < MIN_FEATURES:
                rows.append({"index": index,
                             "status": "insufficient_refresh_features",
                             "appearance_candidates": len(candidates)})
                break
            gray = next_gray
        label, overlap = evaluator_label(mode, sample, start_s, box)
        rows.append({"index": index, "status": "tracked",
                     "actor_box": np.round(box, 3).tolist(),
                     "evaluator_association": label,
                     "evaluator_candidate_iou": round(overlap, 5)})
    return {
        "bootstrap_status": "metric_moving_item_at_index8",
        "tracked_frames": sum(row["status"] == "tracked" for row in rows),
        "evaluator_held_frames": sum(row.get("evaluator_association") ==
                                     "held_item" for row in rows),
        "evaluator_replica_frames": sum(row.get("evaluator_association") ==
                                        "raised_replica" for row in rows),
        "frames": rows,
    }


def main() -> None:
    aggregate = json.loads(INPUT.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for entry in aggregate["reports"]:
        mode = entry["mode"]
        report_path = ROOT / entry["report_path"]
        assert sha(report_path) == entry["report_sha256"]
        report = json.loads(report_path.read_text())
        for row in report["owner_show_rows"]:
            result = replay(mode, row["response"]["actor_samples"],
                            vision, exemplar, owner_hist)
            rows.append({"response_mode": mode, "seed": row["seed"],
                         **result})
            print(mode, row["seed"], result["bootstrap_status"],
                  result.get("tracked_frames"),
                  result.get("evaluator_held_frames"),
                  result.get("evaluator_replica_frames"), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "scope": "After a correct actor 3D moving-item bootstrap at V393 index8, these saved exposed RGB images alone drive KLT flow prediction plus fresh frozen DINO/hue appearance-box refresh. No depth, head angle or metric pose is used after bootstrap. Evaluator association compares the RGB box with V393 logged candidate boxes and uses scripted centers to label held versus replica; a missing evaluated candidate is unknown. V388's 20px flow and 60px proposal gates were tuned in a related raised-replica scene, so this is a transfer diagnostic, not held-out calibration or live owner-answer outcome.",
        "source_sha256": sha(SOURCE),
        "input_aggregate_sha256": sha(INPUT),
        "rgb_tracker_source_sha256": sha(Path(appearance_boxes.__code__.co_filename)),
        "rows": rows,
    }
    (OUT / "cued-retry-rgb-continuity-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
