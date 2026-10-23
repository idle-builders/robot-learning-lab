"""V385: sensitivity of the motion cue to delayed/noisy head angle.

Counterfactual replay rotates all 3D candidate points about each retained
camera origin by the same per-frame yaw error. It models an error in the
camera-to-world transform, not new rendering or a measured actuator.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from audit_multi_candidate_motion_v383 import replay
from run import ROOT, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v382/raised-replica-height-audit.json"
TRUTH = ROOT / "artifacts/ape-closed-loop-cove50-v381/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v385"
CASES = (
    ("nominal", 0., "constant"),
    ("fixed_5deg_bias", 5., "constant"),
    ("oscillating_1deg", 1., "oscillating"),
    ("oscillating_3deg", 3., "oscillating"),
    ("oscillating_5deg", 5., "oscillating"),
)


def transform(frames: list[dict], truth: list[dict], amplitude_deg: float,
              pattern: str) -> list[dict]:
    changed = copy.deepcopy(frames)
    for index, frame in enumerate(changed):
        if pattern == "constant":
            error = math.radians(amplitude_deg)
        else:
            error = math.radians(amplitude_deg) * math.sin(
                2 * math.pi * index / 4)
        c, s = math.cos(error), math.sin(error)
        rotation = np.asarray(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
        with np.load(ROOT / truth[index]["actor_packet_path"]) as packet:
            origin = packet["camera_origin_estimated_m"].copy()
        for candidate in frame["actor_appearance_candidates"]:
            point = candidate["world_xyz_m"]
            if point is not None:
                candidate["world_xyz_m"] = (
                    origin + rotation @ (np.asarray(point) - origin)).tolist()
    return changed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_rows = json.loads(INPUT.read_text())["rows"]
    truth_rows = json.loads(TRUTH.read_text())["rows"]
    rows = []
    for original in input_rows:
        truth = next(row["samples"] for row in truth_rows
                     if row["case"] == original["case"])
        cases = []
        for name, degrees, pattern in CASES:
            frames = transform(original["samples"], truth,
                               degrees, pattern)
            result = replay(frames, truth)
            cases.append({
                "case": name, "amplitude_deg": degrees,
                "pattern": pattern,
                "actor_held_correct_frames": result[
                    "actor_held_correct_frames"],
                "actor_wrong_accepted_frames": result[
                    "actor_wrong_accepted_frames"],
                "actor_track_count": result["actor_track_count"],
                "frames": result["frames"],
            })
        rows.append({"source_case": original["case"], "cases": cases})
        print(original["case"], [(case["case"], case[
            "actor_held_correct_frames"], case[
                "actor_wrong_accepted_frames"]) for case in cases],
              flush=True)
    report = {
        "scope": "Deterministic sensitivity replay of V382 all-candidate 3D points under per-frame camera-to-world yaw error. 1/3/5deg oscillations have a 0.4s sampled period; a fixed 5deg bias is the control. V383 motion tracker and thresholds are unchanged. This is not measured head encoder latency, motion blur, physical camera error or a claim about sensor precision. It tests whether ego-pose error can turn a static raised replica into an apparent moving item or force abstention.",
        "source_sha256": sha(SOURCE),
        "v383_source_sha256": sha(Path(replay.__code__.co_filename)),
        "input_report_sha256": sha(INPUT),
        "truth_report_sha256": sha(TRUTH),
        "rows": rows,
    }
    (OUT / "head-angle-track-error.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
