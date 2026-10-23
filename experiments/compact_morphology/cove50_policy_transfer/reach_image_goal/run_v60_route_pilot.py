"""Stage bounded off-route checks before the complete development probe."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v60_route_probe as v60,
)


def main() -> None:
    plan = v60.freeze()
    wanted = {"large-20-pitch-low", "large-20-pitch-high",
              "large-30-pitch-low", "large-30-pitch-high",
              "edge-01-roll-low", "edge-01-roll-high",
              "positive-03-pitch-low", "positive-10-center",
              "positive-10-pitch-high"}
    cases = [case for case in plan["cases"] if case["id"] in wanted]
    if len(cases) != len(wanted):
        raise ValueError("Route-pilot identities missing from frozen plan")
    v60.DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(v60.run_case, case, plan): case["id"] for case in cases}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in cases if case["id"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v60-route-pilot.v1",
                "role": "opened_off_authored_route_diagnostic_not_actor_qualification",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": v60.sha(v60.PLAN),
                "complete": len(ordered) == len(cases),
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "thin_contact_count": sum(item["contact_s"] < 1.2 for item in ordered),
                "pilot_gate_met": len(ordered) == len(cases) and all(item["pass"] for item in ordered),
                "cases": ordered,
            }
            (v60.OUT / "route-pilot-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in (
                "id", "pass", "contact_s", "pitch_offset_rad",
                "roll_clearance_rad", "entry_goal_error_mm_scoring_only",
                "negative_camera_sparse_partial_correction",
                "positive_camera_large_sparse_abstention", "edge_goal_bias_applied",
                "reacquisition_trigger_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
