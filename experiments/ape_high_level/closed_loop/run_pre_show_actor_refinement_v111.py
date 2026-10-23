"""V111: refine pre-show actor decisions with continuous point vetoes.

Composes retained V109 map screens and V110 exact actor endpoint gaps.
The result is a planner decision replay, not a new physical episode.
"""

from __future__ import annotations

import json
from pathlib import Path

from pre_show_actor_choice_refinement import MARGIN_M, classify_route
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v111"
V109 = ROOT / "artifacts/ape-closed-loop-cove50-v109/report.json"
V110 = ROOT / "artifacts/ape-closed-loop-cove50-v110/report.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    earlier = json.loads(V109.read_text())
    point = json.loads(V110.read_text())
    rows = []
    for old in earlier["rows"]:
        observed = next(row for row in point["rows"]
                        if row["seed"] == old["seed"] and
                        row["case"] == old["case"])
        routes = {}
        for name, point_samples in observed["routes"].items():
            voxel_samples = old["layouts"][
                "head_rear_pair_right_upper"]["candidate_routes"][
                    name]["pose_samples"]
            routes[name] = classify_route(voxel_samples, point_samples)
        rows.append({
            "seed": old["seed"], "case": old["case"],
            "routes": routes,
            "actor_choice": {
                "status": "no_authorized_meeting_move",
                "request": "defer_show_and_request_clear_space",
                "request_executed_in_this_replay": False,
                "motion_authorized": False,
            },
            "evaluator_only_minimum_gap_by_sequence_m": old[
                "evaluator_only_minimum_gap_by_sequence_m"],
        })
    report = {
        "schema": "robot-learning-lab.ape-pre-show-actor-refinement.v1",
        "source_sha256": sha(SOURCE),
        "classification_source_sha256": sha(Path(__file__).with_name(
            "pre_show_actor_choice_refinement.py")),
        "v109_report_sha256": sha(V109),
        "v110_report_sha256": sha(V110),
        "provisional_margin_m": MARGIN_M,
        "scope": "Actor decision replay joining V109 time-stamped image/range voxel screens with V110 continuous right-upper endpoint distance. Measured close points are a veto; occupied 4cm cells without close endpoints become map disagreement/unknown; absent close endpoints never certify free space. All routes remain unqualified because swept space, pose error, future occupancy and selected-body motion are unresolved. Request is a decision output, not spoken or followed by an owner in this replay.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    counts = {}
    for row in rows:
        for route in row["routes"].values():
            counts[route["status"]] = counts.get(route["status"], 0) + 1
    print(counts)
    for row in rows:
        print(row["seed"], row["case"],
              {name: route["status"] for name, route in row["routes"].items()})
    print(path)


if __name__ == "__main__":
    main()
