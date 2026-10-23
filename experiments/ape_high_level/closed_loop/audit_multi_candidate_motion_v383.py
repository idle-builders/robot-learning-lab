"""V383: keep all taught-look candidates and prefer a moving item track.

This is an actor-only development replay of V380/V382 proposal packets. It
tests whether an explicit owner request to move an item could distinguish a
hand-held can from stationary identical replicas. The prerecorded hand is
already moving, so this does not test response to a newly issued cue.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from run import ROOT, iou, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v383"
INPUTS = (
    ROOT / "artifacts/ape-closed-loop-cove50-v380/held-item-candidate-audit.json",
    ROOT / "artifacts/ape-closed-loop-cove50-v382/raised-replica-height-audit.json",
)
DEDUP_DISTANCE_M = .04
MAX_TRACK_STEP_M = .12
MIN_MOTION_SPAN_M = .08


def candidates(frame: dict) -> list[dict]:
    ordered = sorted((c for c in frame["actor_appearance_candidates"]
                      if c["world_xyz_m"] is not None),
                     key=lambda c: c["appearance_score"], reverse=True)
    result = []
    for item in ordered:
        if all(math.dist(item["world_xyz_m"], earlier["world_xyz_m"])
               > DEDUP_DISTANCE_M for earlier in result):
            result.append(item)
    return result


def motion_span(track: dict) -> float:
    last = [point["xyz"] for point in track["history"][-3:]]
    if len(last) < 3:
        return 0.
    return max(math.dist(a, b) for a in last for b in last)


def replay(frames: list[dict], truth_frames: list[dict]) -> dict:
    tracks = []
    next_id = 1
    rows = []
    for index, frame in enumerate(frames):
        current = candidates(frame)
        edges = sorted((math.dist(track["history"][-1]["xyz"],
                                  item["world_xyz_m"]), tid, cid)
                       for tid, track in enumerate(tracks)
                       if track["last_index"] == index - 1
                       for cid, item in enumerate(current))
        claimed_tracks, claimed_candidates = set(), set()
        for distance, tid, cid in edges:
            if (distance > MAX_TRACK_STEP_M or tid in claimed_tracks or
                    cid in claimed_candidates):
                continue
            track = tracks[tid]
            track["history"].append({"index": index,
                                     "xyz": current[cid]["world_xyz_m"],
                                     "box": current[cid]["box"]})
            track["last_index"] = index
            claimed_tracks.add(tid)
            claimed_candidates.add(cid)
        for cid, item in enumerate(current):
            if cid in claimed_candidates:
                continue
            tracks.append({"id": next_id, "last_index": index,
                           "history": [{"index": index,
                                        "xyz": item["world_xyz_m"],
                                        "box": item["box"]}]})
            next_id += 1
        moving = [(track, motion_span(track)) for track in tracks
                  if track["last_index"] == index and
                  motion_span(track) >= MIN_MOTION_SPAN_M]
        selected = moving[0][0] if len(moving) == 1 else None
        box = None if selected is None else selected["history"][-1]["box"]
        truth = truth_frames[index]["evaluator_truth_box"]
        rows.append({
            "frame": index,
            "actor_candidate_count": len(current),
            "actor_moving_track_count": len(moving),
            "actor_selected_track_id": None if selected is None else
                selected["id"],
            "actor_selected_box": box,
            "actor_selected_motion_span_m": (
                None if not moving or selected is None else
                round(moving[0][1], 5)),
            "evaluator_selected_held_can": (
                box is not None and truth is not None and
                iou(box, truth) >= .25),
        })
    return {
        "actor_track_count": len(tracks),
        "actor_held_correct_frames": sum(
            row["evaluator_selected_held_can"] for row in rows),
        "actor_wrong_accepted_frames": sum(
            row["actor_selected_box"] is not None and not
            row["evaluator_selected_held_can"] for row in rows),
        "frames": rows,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_path in INPUTS:
        source = json.loads(input_path.read_text())
        for candidate_row in source["rows"]:
            origin_path = (ROOT / candidate_row["source_report_path"]
                           if "source_report_path" in candidate_row else
                           ROOT / "artifacts/ape-closed-loop-cove50-v381/independent-hand-show.json")
            origin = json.loads(origin_path.read_text())
            truth = next(row["samples"] for row in origin["rows"]
                         if row["case"] == candidate_row["case"])
            result = replay(candidate_row["samples"], truth)
            result.update({
                "source_candidate_report": str(input_path.relative_to(ROOT)),
                "source_video_path": candidate_row["source_video_path"],
                "case": candidate_row["case"],
                "baseline_correct_frames": candidate_row[
                    "baseline_held_correct_frames"],
                "baseline_wrong_frames": candidate_row[
                    "baseline_wrong_accepted_frames"],
            })
            rows.append(result)
            print(input_path.parent.name, result["case"],
                  "baseline", result["baseline_correct_frames"],
                  result["baseline_wrong_frames"], "motion",
                  result["actor_held_correct_frames"],
                  result["actor_wrong_accepted_frames"], flush=True)
    report = {
        "scope": "Replay all DINO+hue+finite-depth candidates from V380 floor and single-item controls and V382 raised replica. Deduplicate points within4cm, link each candidate to one previous-frame 3D point within12cm, and select only a unique active track with >=8cm spatial span over its last3 points. Evaluator held-can boxes score after the actor choice. Motion thresholds are development choices on already visible trajectories; owner hand motion is scripted from the start, not a contingent response, and camera pose/head angle have optimistic error. This is an information-action candidate, not exact identity or a sensor decision.",
        "source_sha256": sha(SOURCE),
        "input_report_sha256": {str(path.relative_to(ROOT)): sha(path)
                                for path in INPUTS},
        "dedup_distance_m": DEDUP_DISTANCE_M,
        "max_track_step_m": MAX_TRACK_STEP_M,
        "min_motion_span_m": MIN_MOTION_SPAN_M,
        "rows": rows,
    }
    (OUT / "multi-candidate-motion-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
