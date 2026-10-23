"""Diagnostic depth-continuity ablation on saved owner association decisions.

Only RGB choice, its candidate-local depth and capture time enter this gate.
Evaluator character positions/labels are copied afterward for analysis.
The speed/allowance are hypotheses for this clip, not calibrated human limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-radial-speed-mps", type=float, default=1.0)
    parser.add_argument("--depth-allowance-m", type=float, default=0.10)
    parser.add_argument("--min-local-valid-fraction", type=float, default=0.50)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    analysis = json.loads((evidence / "owner-video-analysis.json").read_text())
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    output = {"scope": "offline diagnostic on frozen RGB choices; no evaluator pose in gate",
              "max_radial_speed_mps": args.max_radial_speed_mps,
              "depth_allowance_m": args.depth_allowance_m,
              "min_local_valid_fraction": args.min_local_valid_fraction,
              "views": []}
    for view in analysis["views"]:
        last_depth = None
        last_time = None
        trace = []
        for frame, row in enumerate(view["trace"]):
            time_s = float(motion[frame]["timeline_time_s"])
            choice = row["chosen_proposal"]
            support = (row["depth_health"]["candidate_support"][choice]
                       if choice is not None and row["depth_health"] is not None else None)
            depth = support["torso_patch_depth_m"] if support else None
            valid = support["torso_patch_valid_fraction"] if support else None
            if choice is None:
                reason = "rgb_abstained"
            elif depth is None or valid < args.min_local_valid_fraction:
                reason = "local_depth_unhealthy"
            elif last_depth is not None and abs(depth - last_depth) > (
                    args.depth_allowance_m + args.max_radial_speed_mps * (time_s - last_time)):
                reason = "depth_jump"
            else:
                reason = "accepted"
                last_depth, last_time = depth, time_s
            trace.append({"frame": frame, "capture_time_s": time_s,
                          "rgb_choice": choice, "chosen_patch_depth_m": depth,
                          "chosen_patch_valid_fraction": valid,
                          "decision": reason,
                          "chosen_evaluator_character_index":
                          row["chosen_evaluator_character_index"]})
        output["views"].append({"fov_deg": view["fov_deg"],
            "rgb_choices": sum(row["rgb_choice"] is not None for row in trace),
            "depth_accepted": sum(row["decision"] == "accepted" for row in trace),
            "depth_jump_refusals": sum(row["decision"] == "depth_jump" for row in trace),
            "unhealthy_refusals": sum(row["decision"] == "local_depth_unhealthy" for row in trace),
            "trace": trace})
    target = evidence / "owner-depth-continuity-audit.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in view.items() if k != "trace"}
                      for view in output["views"]], indent=2))


if __name__ == "__main__":
    main()
