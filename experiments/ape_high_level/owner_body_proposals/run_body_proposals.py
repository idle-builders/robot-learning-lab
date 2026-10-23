"""Full owner turns with a consistent full-person proposal before depth.

The existing full owner turn is unchanged. A narrow sensor-adapter overlay
replaces only the broad frame-health decision immediately after the released
person detector returns. No simulator labels, poses or person IDs enter it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/owner_standoff_response"))
import run_standoff_response as base

OUT = ROOT / "artifacts/ape-owner-body-proposals/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base.OUT = OUT
    base.CASES = (
        ("stationary_frame_control", "stationary", .55, .2),
        ("stationary_local_only", "stationary", .55, .2),
        ("stationary_body", "stationary", .55, .2),
        ("moving_body", "moving", .55, 1.6),
    )
    real_episode = base.joined.episode
    real_detect = base.turn.async_person.detect_person
    real_health = base.turn.depth_change.packet_health
    packet_log = {}
    proposal_log = {}

    def episode(scene, response, *args):
        if response.endswith("_frame_control"):
            return real_episode(scene, response, *args)
        body_mode = response.endswith("_body")
        pending = {}
        lock = threading.Lock()
        decisions = []
        proposals = []

        def detect(rgb):
            result = real_detect(rgb)
            original_detections = list(result["detections"])
            detections = list(original_detections)
            if body_mode:
                qualified = [row for row in detections if row["score"] >= .25]
                other = [row for row in detections if row["score"] < .25]
                qualified.sort(key=lambda row: (
                    row["box_xyxy"][2] - row["box_xyxy"][0]) *
                    (row["box_xyxy"][3] - row["box_xyxy"][1]), reverse=True)
                detections = qualified + other
            result = {**result, "detections": detections}
            with lock:
                pending["detections"] = detections
                proposals.append({
                    "rank_mode": "largest_qualified_box" if body_mode else "detector_score",
                    "detector_boxes": original_detections,
                    "selected_box_xyxy": detections[0]["box_xyxy"] if detections else None,
                })
            return result

        def health(before, after):
            healthy, fractions = real_health(before, after)
            with lock:
                detections = pending.pop("detections", None)
            if detections is None:
                return healthy, fractions
            best = detections[0] if detections else None
            local_fraction = None
            if best is not None and best["score"] >= .25:
                x1, y1, x2, y2 = (int(v) for v in best["box_xyxy"])
                patch = after[max(0, y1):min(after.shape[0], y2),
                              max(0, x1):min(after.shape[1], x2)]
                if patch.size:
                    local_fraction = float(np.mean(
                        np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)))
            local_supported = local_fraction is not None and local_fraction >= .5
            accepted = healthy or local_supported
            decisions.append({
                "central_valid_fraction": fractions[0],
                "person_score": best["score"] if best else None,
                "person_box_local_valid_fraction": local_fraction,
                "whole_frame_healthy": healthy,
                "actor_health_with_person_box": accepted,
                "local_override": not healthy and local_supported,
            })
            return accepted, fractions

        base.turn.async_person.detect_person = detect
        base.turn.depth_change.packet_health = health
        try:
            result = real_episode(scene, response, *args)
        finally:
            base.turn.async_person.detect_person = real_detect
            base.turn.depth_change.packet_health = real_health
        packet_log[response] = decisions
        proposal_log[response] = proposals
        return result

    base.joined.episode = episode
    try:
        base.main()
    finally:
        base.joined.episode = real_episode
        base.turn.async_person.detect_person = real_detect
        base.turn.depth_change.packet_health = real_health
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["task"] = "T-rq5qyt"
    report["source_hashes"][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    report["candidate_depth_sensor_adapter"] = (
        "For the body-proposal condition only, reorder score>=0.25 person boxes by "
        "image area before the existing best-box depth localization. Both local-only "
        "and body conditions accept >=50% finite [0.15,2.8]m depth in that "
        "selected box when whole-frame health fails. Retain 3-D point, two-view "
        "consistency, association, and motion gates. Single-person pilot only.")
    report["candidate_depth_decisions"] = packet_log
    report["person_proposal_decisions"] = proposal_log
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
