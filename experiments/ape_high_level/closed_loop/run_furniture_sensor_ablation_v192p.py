"""V192p: matched no-motion control and forward-range-occluded owner turns.

Keep the V174 camera cue and V153 decision loop frozen while changing only
the scripted prop motion or the visible forward-left range returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
import run_continuous_owner_guard_v95 as v95
import run_standoff_scene_variants_v160p as v160p
from compare_standoff_range_layouts_v165p import front_left_roi
import run_varied_furniture_owner_v191p as v191p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-furniture-ablation"


def occluded_front_scan(room, robot, case):
    clouds, counts = ORIGINAL_SCAN(room, robot, case)
    world = clouds["four"]
    first_count = len(clouds["two"])
    estimated = v95.estimated_shin_points(
        world, room.actual.copy(), room.estimated.copy())
    blocked = (front_left_roi(estimated, room.estimated.copy()) &
               (np.arange(len(world)) < first_count))
    clouds["two"] = clouds["two"][~blocked[:first_count]]
    clouds["four"] = world[~blocked]
    counts["two_environment_points"] = len(clouds["two"])
    counts["injected_forward_pair_roi_removed"] = int(np.count_nonzero(blocked))
    return clouds, counts


ORIGINAL_SCAN = v95.front_scan


def run_condition(name: str, *, stationary: bool,
                  occlude_forward: bool) -> dict:
    original = (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S,
                v95.front_scan)
    try:
        v191p.OUT = OUT / name
        v191p.LAYOUTS = {
            key: {**value, "prop_variant": (
                "stationary_wide" if stationary else value["prop_variant"])}
            for key, value in original[1].items()
            if key in ("near_side_table", "offset_side_table")}
        v191p.RESPONSES_S = (None,) if stationary else (.2, None)
        if occlude_forward:
            v95.front_scan = occluded_front_scan
        v191p.main()
        report_path = v191p.OUT / "report.json"
        report = json.loads(report_path.read_text())
        report.update({
            "schema": "robot-learning-lab.ape-furniture-ablation.v1",
            "condition": name,
            "ablation_source_sha256": sha(SOURCE),
            "stationary_prop_control": stationary,
            "injected_forward_pair_roi_occlusion": occlude_forward,
            "scope": "Complete diagnostic held-pose owner turns with frozen "
                     "V174 camera cue and V153 task decisions. "
                     "Stationary controls keep the low prop at its start "
                     "position through the owner show. Occluded conditions "
                     "remove only front-left ROI returns from the simulated "
                     "forward pair, preserving other range and camera packets. "
                     "The historical pose is not actor-admitted or physically "
                     "reached; evaluator clearance never enters the actor.",
        })
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return {"condition": name,
                "report_path": str(report_path.relative_to(ROOT)),
                "report_sha256": sha(report_path),
                "episodes": len(report["rows"]),
                "alarms": [r["front_alarm"] for r in report["rows"]],
                "minimum_gaps_m": [r["evaluator_minimum_scene_gap_m"]
                                   for r in report["rows"]],
                "task_results": [r["task_result"] for r in report["rows"]]}
    finally:
        v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S, v95.front_scan = original


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v160p.VARIANTS["stationary_wide"] = {
        **v160p.VARIANTS["fast_wide_turn"],
        "approach_start_s": 30., "approach_end_s": 31.}
    try:
        conditions = [
            run_condition("stationary_control", stationary=True,
                          occlude_forward=False),
            run_condition("moving_forward_occluded", stationary=False,
                          occlude_forward=True),
        ]
    finally:
        del v160p.VARIANTS["stationary_wide"]
    report = {"schema": "robot-learning-lab.ape-furniture-ablation-index.v1",
              "source_sha256": sha(SOURCE), "conditions": conditions}
    (OUT / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "index.json")


if __name__ == "__main__":
    main()
