"""V358: reveal-based posture tuning, including a closer ignored invitation.

This intentionally tunes on V357 failures. It can suggest a follow-up policy
candidate but is not an independent held-out evaluation or safety signoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import challenge_prepared_posture_v357 as challenge
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v358"
POSTURES_RAD = (-.60, -.55, -.50, -.45, -.40)
VARIANTS = challenge.VARIANTS + (
    {"name": "left_thick", "offset_xyz_m": (0., .12, 0.),
     "arm_extra_radius_m": .02},
)
REACHES = ("near", "near_closer", "far")
NEAR_CLOSER_HAND_XYZ_M = np.asarray((.16, .08, .20))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    challenge.REACHES["near_closer"] = NEAR_CLOSER_HAND_XYZ_M
    rows = [challenge.run_case(
        170, variant, 0., reach, target_shoulder_rad=posture, out=OUT)
        for posture in POSTURES_RAD for variant in VARIANTS
        for reach in REACHES]
    posture_scores = []
    for posture in POSTURES_RAD:
        subset = [r for r in rows if r["target_shoulder_rad"] == posture]
        worst = min(min(r["evaluator_min_person_gap_m"],
                        r["evaluator_min_prop_gap_m"]) for r in subset)
        posture_scores.append({
            "target_shoulder_rad": posture,
            "evaluator_worst_person_or_prop_gap_m": round(worst, 5),
            "evaluator_under_5cm_cases": sum(
                r["evaluator_under_5cm_person"] or
                r["evaluator_under_5cm_prop"] for r in subset),
            "evaluator_worst_case": min(
                subset, key=lambda r: min(r["evaluator_min_person_gap_m"],
                                          r["evaluator_min_prop_gap_m"]))[
                    "case"],
        })
    report = {
        "scope": "Revealed-variant evaluator-only posture tuning after V357 found a shifted-left near-reach miss. Same selected-body invitation qpos and 20ms kinematic arm/person/stool geometry, seed170. Compare five partial shoulder targets across base/shifted/farther/thicker/left+thicker people and near, closer-near, far paths. No actor forecast, force/stop or independent held-out split; do not treat a winning posture as qualified.",
        "source_sha256": sha(SOURCE),
        "challenge_source_sha256": sha(Path(challenge.__file__)),
        "posture_scores": posture_scores,
        "rows": rows,
    }
    path = OUT / "posture-tuning.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for score in posture_scores:
        print(score, flush=True)


if __name__ == "__main__":
    main()
