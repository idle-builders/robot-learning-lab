"""Counterfactual head-pose sensitivity on V393 live cue/retry packets.

Rotates candidate 3-D points around the captured camera origin before the
unchanged gap-latched tracker sees them. This tests transform sensitivity,
not a measured head encoder, pose-health estimator or new camera image.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from audit_cued_retry_show_v395 import REPLICA, expected_held
from gap_latched_candidate_tracks import GapLatchedCandidateTracks
from run import ROOT, sha
from run_live_cued_retry_show_v393 import OUT as INPUT_ROOT


SOURCE = Path(__file__)
INPUT = INPUT_ROOT / "cued-retry-show-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v396"
ERRORS = (
    ("nominal", 0., "constant"),
    ("fixed_5deg", 5., "constant"),
    ("oscillating_1deg", 1., "oscillating"),
    ("oscillating_3deg", 3., "oscillating"),
    ("oscillating_5deg", 5., "oscillating"),
)


def changed_candidates(sample: dict, index: int, amplitude_deg: float,
                       pattern: str) -> list[dict]:
    if pattern == "constant":
        error = math.radians(amplitude_deg)
    else:
        error = math.radians(amplitude_deg) * math.sin(
            2. * math.pi * index / 4.)
    c, s = math.cos(error), math.sin(error)
    rotation = np.asarray(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    with np.load(ROOT / sample["actor_packet_path"]) as packet:
        origin = packet["camera_origin_estimated_m"].copy()
    candidates = copy.deepcopy(sample["actor_choice"]["actor_all_candidates"])
    for candidate in candidates:
        point = np.asarray(candidate["world_xyz_m"])
        candidate["world_xyz_m"] = (
            origin + rotation @ (point - origin)).tolist()
    return candidates


def replay(mode: str, samples: list[dict], amplitude_deg: float,
           pattern: str) -> dict:
    tracker = GapLatchedCandidateTracks()
    start_s = float(samples[0]["capture_time_s"])
    frames = []
    for index, sample in enumerate(samples):
        candidates = changed_candidates(sample, index, amplitude_deg,
                                        pattern)
        decision = tracker.update(index, candidates)
        selected = decision["selected_candidate"]
        label = "none"
        held_error = replica_error = None
        if selected is not None:
            elapsed = float(sample["capture_time_s"]) - start_s
            point = np.asarray(selected["world_xyz_m"])
            held_error = float(np.linalg.norm(point -
                                              expected_held(mode, elapsed)))
            replica_error = float(np.linalg.norm(point - REPLICA))
            label = ("held_item" if held_error < replica_error else
                     "raised_replica")
        frames.append({
            "index": index,
            "selected": label,
            "held_error_m": (None if held_error is None else
                             round(held_error, 5)),
            "replica_error_m": (None if replica_error is None else
                                round(replica_error, 5)),
            "active_moving_track_count": decision["moving_track_count"],
        })
    return {
        "selected_held_frames": sum(x["selected"] == "held_item"
                                    for x in frames),
        "selected_replica_frames": sum(x["selected"] == "raised_replica"
                                       for x in frames),
        "final_selected": frames[-1]["selected"],
        "frames": frames,
    }


def main() -> None:
    aggregate = json.loads(INPUT.read_text())
    rows = []
    for entry in aggregate["reports"]:
        mode = entry["mode"]
        report_path = ROOT / entry["report_path"]
        assert sha(report_path) == entry["report_sha256"]
        report = json.loads(report_path.read_text())
        for row in report["owner_show_rows"]:
            cases = []
            for name, degrees, pattern in ERRORS:
                result = replay(mode, row["response"]["actor_samples"],
                                degrees, pattern)
                cases.append({"error_case": name,
                              "yaw_error_amplitude_deg": degrees,
                              "pattern": pattern, **result})
            rows.append({"response_mode": mode,
                         "seed": row["seed"], "cases": cases})
            print(mode, row["seed"], [(case["error_case"], case[
                "selected_held_frames"], case["selected_replica_frames"])
                for case in cases], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "scope": "V393 saved actor appearance/depth candidate points replayed through the unchanged gap-latched motion tracker after a constant or oscillating per-frame yaw rotation about the captured camera origin. Oscillation has a 0.4s period. Only post-hoc evaluator labels use scripted held and replica centers. The social stillness, owner answer and base planner are not rerun; this is a localization/association sensitivity audit, not live task outcome or a measured sensor error.",
        "source_sha256": sha(SOURCE),
        "input_aggregate_sha256": sha(INPUT),
        "tracker_source_sha256": sha(Path(
            GapLatchedCandidateTracks.__init__.__code__.co_filename)),
        "rows": rows,
    }
    (OUT / "cued-retry-pose-error-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
