"""Score V398's logged internal moving-candidate selections independently."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_cued_retry_show_v395 import REPLICA, expected_held
from gap_latched_candidate_tracks import GapLatchedCandidateTracks
from run import ROOT, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v398/live-pose-health-show-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v399"


def main() -> None:
    audit = json.loads(INPUT.read_text())
    rows = []
    for arm in audit["rows"]:
        path = ROOT / arm["report_path"]
        assert sha(path) == arm["report_sha256"]
        report = json.loads(path.read_text())
        for source in report["owner_show_rows"]:
            tracker = GapLatchedCandidateTracks()
            samples = source["response"]["actor_samples"]
            start_s = float(samples[0]["capture_time_s"])
            frames = []
            for index, sample in enumerate(samples):
                choice = sample["actor_choice"]
                result = tracker.update(index, choice["actor_all_candidates"])
                assert result["selected_track_id"] == choice[
                    "actor_selected_track_id"]
                selected = result["selected_candidate"]
                label = "none"
                if selected is not None:
                    position = np.asarray(selected["world_xyz_m"])
                    elapsed = float(sample["capture_time_s"]) - start_s
                    held_error = float(np.linalg.norm(position -
                                          expected_held("compliant", elapsed)))
                    replica_error = float(np.linalg.norm(position - REPLICA))
                    label = ("held_item" if held_error < replica_error else
                             "raised_replica")
                frames.append({
                    "index": index,
                    "evaluator_selected_class": label,
                    "actor_pose_health": choice.get("actor_pose_health"),
                })
            rows.append({
                "arm": arm["arm"], "seed": source["seed"],
                "selected_held_frames": sum(
                    x["evaluator_selected_class"] == "held_item"
                    for x in frames),
                "selected_replica_frames": sum(
                    x["evaluator_selected_class"] == "raised_replica"
                    for x in frames),
                "selected_after_fault_frames": sum(
                    x["index"] >= audit["fault_first_index"] and
                    x["evaluator_selected_class"] != "none"
                    for x in frames),
                "unreliable_pose_frames": sum(
                    x["actor_pose_health"] ==
                    "unreliable_encoder_time_alignment" for x in frames),
                "frames": frames,
            })
            print(rows[-1]["arm"], rows[-1]["seed"],
                  rows[-1]["selected_held_frames"],
                  rows[-1]["selected_replica_frames"],
                  rows[-1]["selected_after_fault_frames"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "scope": "Evaluator-only reconstruction of V398 internal gap-latched selections from the actual live logged candidates. Asserts reconstructed track IDs equal actor log IDs. Uses scripted held/replica centers only for post-run identity scoring. No actor claim is inferred from an internal track selection; the V398 task status and accepted points remain the outcome evidence.",
        "source_sha256": sha(SOURCE),
        "input_sha256": sha(INPUT),
        "rows": rows,
    }
    (OUT / "live-pose-health-track-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
