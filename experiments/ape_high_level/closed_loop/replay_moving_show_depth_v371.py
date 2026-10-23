"""V371: replay V370's exact actor packets with hole-tolerant depth location.

The image detector and camera packets are unchanged. Compare the existing
NaN-propagating 5x5 median to a supported finite median and depth spread
check for the accepted box. Truth only measures resulting item error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import (ITEM_SIDE_OFFSET_M,
                                      START_HAND_XYZ_M)
from challenge_moving_distant_show_v370 import owner_offset
from direct_structural_rooms_v270 import selected_visual
from robust_item_depth import locate_box_center
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v371"
V370 = ROOT / "artifacts/ape-closed-loop-cove50-v370/moving-distant-show.json"
FOVY_DEG = 67.


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(V370.read_text())
    rows = []
    for original in prior["rows"]:
        samples = []
        for sample in original["samples"]:
            packet_path = ROOT / sample["actor_packet_path"]
            if sha(packet_path) != sample["actor_packet_sha256"]:
                raise AssertionError("V370 actor packet changed")
            with np.load(packet_path) as packet:
                camera_pose = (
                    packet["camera_origin_estimated_m"].copy(),
                    packet["camera_rotation_estimated"].copy())
                depths = {name: packet[f"{name}_depth_m"].copy()
                          for name in ("ideal", "short", "noisy_long")}
            box = sample["actor_choice"]["accepted_box"]
            estimates = {name: locate_box_center(
                box, camera_pose, depth, fovy_deg=FOVY_DEG)
                for name, depth in depths.items()}
            elapsed_s = sample["capture_time_s"] - 16.2
            true_item = (START_HAND_XYZ_M + ITEM_SIDE_OFFSET_M +
                         owner_offset(original["pattern"], elapsed_s))
            for estimate in estimates.values():
                position = estimate["world_xyz_m"]
                estimate["evaluator_position_error_m"] = (
                    None if position is None or not sample[
                        "evaluator_accepted_presented_can"] else
                    round(float(np.linalg.norm(
                        np.asarray(position) - true_item)), 5))
            samples.append({
                "capture_time_s": sample["capture_time_s"],
                "source_actor_packet_sha256": sample["actor_packet_sha256"],
                "actor_box": box,
                "evaluator_accepted_presented_can": sample[
                    "evaluator_accepted_presented_can"],
                "evaluator_true_item_xyz_m": true_item.round(5).tolist(),
                "estimates": estimates})
        rows.append({"case": original["case"], "pattern": original["pattern"],
                     "seed": original["seed"], "samples": samples,
                     "score": {name: {
                         "located_frames": sum(s["estimates"][name][
                             "world_xyz_m"] is not None for s in samples),
                         "correct_item_located": sum(
                             s["evaluator_accepted_presented_can"] and
                             s["estimates"][name]["world_xyz_m"] is not None
                             for s in samples),
                         "median_correct_item_error_m": (
                             round(float(np.median(errors)), 5) if errors
                             else None),
                         "max_correct_item_error_m": (
                             round(max(errors), 5) if errors else None),
                     } for name in ("ideal", "short", "noisy_long")
                     for errors in [[s["estimates"][name][
                         "evaluator_position_error_m"] for s in samples
                         if s["estimates"][name][
                             "evaluator_position_error_m"] is not None]]}})
    report = {
        "scope": "Read-only exact-packet V370 selected-body replay of the same accepted taught-item boxes at10Hz under 20ms RGB exposure, direct head slew and still/gentle/faster person motion. Compare existing NaN-propagating median world positions in V370 to an actor-only 5x5 finite median requiring >=12 valid pixels and <=5cm median absolute depth spread, for ideal/short/noisy-long depth. True item position scores only afterward. Depth noise/dropout and exposure are procedural; no sensor purchase or motion authorization.",
        "source_sha256": sha(SOURCE),
        "robust_source_sha256": sha(Path(locate_box_center.__code__.co_filename)),
        "v370_report_sha256": sha(V370),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows}
    (OUT / "moving-show-depth-replay.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], row["score"], flush=True)


if __name__ == "__main__":
    main()
