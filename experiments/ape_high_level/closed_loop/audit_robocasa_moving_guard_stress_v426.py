"""Independent moving-cylinder and video audit for guard sensor faults."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_moving_prop_v424 import audit_episode
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-moving-guard-stress-v425"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_moving_guard_stress_v425.py"
    trial = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_moving_prop_v423.py"
    cases = []
    for case in report["cases"]:
        audit = audit_episode(case["episode"])
        cases.append({"mode": case["mode"], "outcome": case["episode"]["outcome"],
                      "end_time_s": case["episode"]["end_time_s"],
                      "minimum_signed_prop_distance_m": audit["minimum_signed_prop_distance_m"],
                      "required_5cm_margin_pass": audit["required_5cm_margin_pass"],
                      "video_sha_pass": audit["video_sha_pass"],
                      "video_frames_pass": audit["video_frames_pass"],
                      "distance_replay_max_error_m": audit["distance_replay_max_error_m"]})
    output = {"source_sha256": sha256(Path(__file__)),
              "stress_source_sha_pass": report["source_sha256"] == sha256(source),
              "frozen_trial_source_sha_pass": report["frozen_trial_source_sha256"] == sha256(trial),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "cases": cases,
              "all_video_distance_pass": all(case["video_sha_pass"] and case["video_frames_pass"] and
                                             case["distance_replay_max_error_m"] < .001 for case in cases)}
    (OUT / "audit-v426.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["mode"], case["outcome"],
              round(case["minimum_signed_prop_distance_m"], 4),
              case["required_5cm_margin_pass"], flush=True)
    print("all_video_distance_pass", output["all_video_distance_pass"], flush=True)


if __name__ == "__main__":
    main()
