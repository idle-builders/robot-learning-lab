"""Same 20-room approach with a 1 cm command-resolution standoff band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_post_confirmation_approach as base


TOLERANCE_M = .01
ORIGINAL_CHOOSE = base.choose_action
ASSETS = ROOT / "assets/real_indoor_scenes/post_confirmation_20_approach_v2_20261015"
PRECOMMIT = ART / "post_confirmation_20_approach_v2_precommit.json"
REPORT = ART / "post_confirmation_20_approach_v2_report.json"


def choose_with_resolution(actor_world, actor_yaw, target_world, ranges):
    distance = float(base.np.linalg.norm(target_world - actor_world))
    if distance <= base.STANDOFF_M + TOLERANCE_M:
        return "hold", 0., "observed_target_standoff"
    return ORIGINAL_CHOOSE(actor_world, actor_yaw, target_world, ranges)


base.choose_action = choose_with_resolution
base.ASSETS = ASSETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args = parser.parse_args()
    cases = json.loads((ART / "integrated_owner_20_conditional_precommit.json").read_text())[
        "cases"]
    frozen = {"schema": "ape.post-confirmation-20-approach-v2-precommit.v1",
        "source_report": relative(base.SOURCE), "source_report_sha256": sha(base.SOURCE),
        "base_policy_source": relative(Path(base.__file__)),
        "base_policy_sha256": sha(Path(base.__file__)),
        "cases": [{"scene_key": c["scene_key"], "capture_group": c["capture_group"],
                   "screen": c["screen"], "screen_sha256": sha(ROOT / c["screen"])}
                  for c in cases],
        "policy_delta": "Hold when the observed target gap is within 1 cm of the 55 cm standoff, avoiding sub-centimetre command chattering; all other turn, range, floor, body and fresh RGB-D rules remain unchanged.",
        "standoff_m": base.STANDOFF_M, "stop_tolerance_m": TOLERANCE_M,
        "score": ["bounded_actions", "observed_range_stop", "final_target_gap",
                  "fresh_visual_reconfirmation", "aligned_video_depth_trace"],
        "limits": "Development-exposed captures, commanded-pose kinematic proxy, ideal yaw/range, static objects and scripted owner."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen v2 approach changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        path = ASSETS / case["scene_key"] / "trace.json"
        if not path.exists():
            row = base.run_case(case)
        else:
            trace = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"], "trace": relative(path),
                "video": trace["video"], "action_result": trace["action_result"],
                "reconfirmation_action": trace["reconfirmation_action"],
                "accepted_forward_steps": sum(a["decision"] == "forward" and a.get(
                    "evaluator_motion_accepted") for a in trace["actions"]),
                "final_actual_target_gap_m": trace["actions"][-1][
                    "evaluator_actual_target_gap_m"]}
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.post-confirmation-20-approach-v2-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
            indent=2) + "\n")
        print(row["scene_key"], row["action_result"], row["accepted_forward_steps"],
              row["reconfirmation_action"], flush=True)


if __name__ == "__main__":
    main()
