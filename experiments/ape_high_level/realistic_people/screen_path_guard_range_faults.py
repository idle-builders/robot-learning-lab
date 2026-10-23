"""Saved-packet sensitivity screen for range and pose perturbations.

This is an offline model perturbation, not a sensor specification or a live
closed-loop outcome. Input run audit labels never enter the guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from replay_path_timed_guard import replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazard", required=True, type=Path)
    parser.add_argument("--safe", required=True, type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.trials <= 100:
        raise ValueError("trials must be bounded to 1–100")
    conditions = (("clean", 0., 0., 0., 0.),
                  ("range_5mm_drop5", .005, .05, 0., 0.),
                  ("range_10mm_drop10", .01, .10, 0., 0.),
                  ("pose_1cm_1deg", 0., 0., .01, 1.),
                  ("pose_2cm_2deg", 0., 0., .02, 2.))
    records = []
    for condition, noise, missing, pose_noise, heading_noise in conditions:
        for seed in range(args.trials):
            for role, run in (("hazard", args.hazard),
                              ("safe", args.safe),
                              ("static", args.static)):
                result = replay(run, rng=np.random.default_rng(seed + 1000),
                                noise_std_m=noise,
                                dropout_probability=missing,
                                pose_noise_std_m=pose_noise,
                                heading_noise_std_deg=heading_noise)
                records.append({"role": role, "run": str(run), "seed": seed,
                    "condition": condition,
                    "range_noise_std_m": noise,
                    "beam_dropout_probability": missing,
                    "pose_noise_std_m": pose_noise,
                    "heading_noise_std_deg": heading_noise,
                    "first_fresh_intervention_frame": result[
                        "first_fresh_intervention_frame"],
                    "fresh_guard_actions": result["fresh_guard_actions"]})
    summary = []
    for condition, noise, missing, pose_noise, heading_noise in conditions:
        for role in ("hazard", "safe", "static"):
            group = [r for r in records if r["role"] == role and
                     r["condition"] == condition]
            summary.append({"role": role, "condition": condition,
                "range_noise_std_m": noise,
                "beam_dropout_probability": missing,
                "pose_noise_std_m": pose_noise,
                "heading_noise_std_deg": heading_noise,
                "trials": len(group),
                "intervention_trials": sum(r["first_fresh_intervention_frame"]
                                           is not None for r in group),
                "first_intervention_frames": [r["first_fresh_intervention_frame"]
                                              for r in group]})
    output = {"scope": "counterfactual saved-packet fan fault screen",
              "source_sha256": {name: hashlib.sha256(
                  (Path(__file__).parent / name).read_bytes()).hexdigest()
                  for name in ("fan_view_route.py", "replay_path_timed_guard.py",
                               "screen_path_guard_range_faults.py")},
              "summary": summary, "trials": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for row in summary:
        print(row["role"], row["condition"],
              row["intervention_trials"], "/", row["trials"])


if __name__ == "__main__":
    main()
