"""Paired owner turns with person-box depth health during reacquisition.

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

OUT = ROOT / "artifacts/ape-owner-candidate-depth/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base.OUT = OUT
    base.CASES = (
        ("stationary_control", "stationary", .55, .2),
        ("stationary_candidate", "stationary", .55, .2),
        ("moving_control", "moving", .55, 1.6),
        ("moving_candidate", "moving", .55, 1.6),
    )
    real_episode = base.joined.episode
    real_detect = base.turn.async_person.detect_person
    real_health = base.turn.depth_change.packet_health
    packet_log = {}

    def episode(scene, response, *args):
        if not response.endswith("_candidate"):
            return real_episode(scene, response, *args)
        pending = {}
        lock = threading.Lock()
        decisions = []

        def detect(rgb):
            result = real_detect(rgb)
            with lock:
                pending["detections"] = result["detections"]
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
    report["task"] = "T-qfz7s5"
    report["source_hashes"][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    report["candidate_depth_sensor_adapter"] = (
        "After the current person detector completes, accept a depth frame if the "
        "best >=0.25-score person box has >=50% finite depth in [0.15,2.8]m, "
        "even when the separate whole-frame central patch fails. Keep the "
        "original 3-D point, two-view consistency, association and motion gates.")
    report["candidate_depth_decisions"] = packet_log
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
