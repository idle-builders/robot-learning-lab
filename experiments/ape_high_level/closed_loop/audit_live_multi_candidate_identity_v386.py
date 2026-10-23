"""V386: evaluator-only identity and freshness audit of live V384 tracks.

The actor never receives scene IDs. This post-run audit compares its retained
selected 3D point with the scripted moving hand item and raised replica so a
good owner-help status cannot hide an association to the wrong identical can.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import ITEM_SIDE_OFFSET_M, START_HAND_XYZ_M
from challenge_independent_hand_show_v374 import hand_offset
from challenge_raised_replica_show_v381 import REPLICA_XYZ_M
from run import ROOT, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v384/blocked-owner-help-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v386"


def expected_held(elapsed_s: float) -> np.ndarray:
    point = START_HAND_XYZ_M + ITEM_SIDE_OFFSET_M
    if elapsed_s >= .3:
        point = point + hand_offset("hand_faster", elapsed_s - .3)
    return point


def audit(row: dict) -> dict:
    response = row["response"]
    samples = response["actor_samples"]
    start = float(samples[0]["capture_time_s"])
    frames = []
    for sample in samples:
        point = sample["actor_choice"]["accepted_world_xyz_m"]
        if point is None:
            classification = "none"
            held_error = replica_error = None
        else:
            position = np.asarray(point)
            elapsed = float(sample["capture_time_s"]) - start
            held_error = float(np.linalg.norm(position -
                                               expected_held(elapsed)))
            replica_error = float(np.linalg.norm(position -
                                                  REPLICA_XYZ_M))
            classification = ("moving_held_item" if held_error < replica_error
                              else "static_raised_replica")
        frames.append({
            "capture_time_s": sample["capture_time_s"],
            "actor_selected_track_id": sample["actor_choice"].get(
                "actor_selected_track_id"),
            "actor_candidate_count": sample["actor_choice"].get(
                "actor_candidate_count"),
            "evaluator_selected_class": classification,
            "evaluator_held_center_error_m": (
                None if held_error is None else round(held_error, 5)),
            "evaluator_replica_center_error_m": (
                None if replica_error is None else round(replica_error, 5)),
        })
    return {
        "seed": row["seed"], "mode": row["mode"],
        "task_status": response["status"],
        "actor_track_confirmed": response["actor_track_confirmed"],
        "actor_track_fresh_at_answer": response[
            "actor_track_fresh_at_answer"],
        "evaluator_moving_held_item_frames": sum(
            frame["evaluator_selected_class"] == "moving_held_item"
            for frame in frames),
        "evaluator_static_replica_frames": sum(
            frame["evaluator_selected_class"] == "static_raised_replica"
            for frame in frames),
        "evaluator_final_selected_class": frames[-1][
            "evaluator_selected_class"],
        "frames": frames,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(INPUT.read_text())
    rows = [audit(row) for row in source["owner_show_rows"]]
    for row in rows:
        print(row["seed"], row["mode"], row["task_status"],
              "held", row["evaluator_moving_held_item_frames"],
              "replica", row["evaluator_static_replica_frames"],
              "final", row["evaluator_final_selected_class"], flush=True)
    report = {
        "scope": "Evaluator-only post-run audit of V384 actor-selected 3D points. Compares each with the scripted side-held can/imposter center and static raised replica center, labels whichever is closer, and checks final source at the owner answer. This is not actor identity evidence or a real hand-pose estimator; its only purpose is to detect a wrong association hidden by task status.",
        "source_sha256": sha(SOURCE),
        "input_report_sha256": sha(INPUT),
        "rows": rows,
    }
    (OUT / "live-multi-candidate-identity-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
