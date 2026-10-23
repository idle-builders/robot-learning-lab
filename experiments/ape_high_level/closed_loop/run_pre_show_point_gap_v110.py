"""V110: distinguish 4cm voxel occupied labels from measured near points.

This replays V109's actor range endpoints against its swept candidate solids.
Evaluator V100/V104 geometry is compared only after the point diagnostic.
No point-cloud pass is treated as observed-free route authorization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import add_person
import direct_structural_rooms_v270 as structural
from pre_show_actor_planner import POSE_OFFSETS
from pre_show_point_gap import endpoints, minimum_swept_point_gap
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_pre_show_reserve_screen_v100 import SEQUENCES, V98


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v110"
V109 = ROOT / "artifacts/ape-closed-loop-cove50-v109"
V104 = ROOT / "artifacts/ape-closed-loop-cove50-v104"
V100 = ROOT / "artifacts/ape-closed-loop-cove50-v100"
V104_PERTURBATIONS = {
    "estimated": "nominal",
    "left_1cm": "left_1cm",
    "right_1cm": "right_1cm",
    "yaw_left_2deg": "yaw_left_2deg",
    "yaw_right_2deg": "yaw_right_2deg",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((V109 / "report.json").read_text())
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v100 = json.loads((V100 / "report.json").read_text())
    v104 = json.loads((V104 / "report.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    rows = []
    try:
        for prior_row in prior["rows"]:
            seed = prior_row["seed"]
            case = prior_row["case"]
            old_case = next(row for row in v98["rows"]
                            if row["condition"] == "moving" and
                            row["case"][0] == seed)
            range_source = next(source for source in prior_row["actor_sources"]
                                if source["kind"] == "current_near_body_range")
            range_path = ROOT / range_source["path"]
            if sha(range_path) != range_source["sha256"]:
                raise ValueError("V109 range source changed")
            actor = json.loads(range_path.read_text())
            points = endpoints(actor["ray_packets"]["right_upper"])
            qpos_source = ROOT / prior_row["source_packet_path"]
            if sha(qpos_source) != prior_row["source_packet_sha256"]:
                raise ValueError("V109 joint packet changed")
            with np.load(qpos_source) as packet:
                qpos = packet["actor_qpos"].copy()
            estimated = np.asarray(prior_row["estimated_pose_xyyaw"])
            room = OfferDriftRoom(case=tuple(old_case["case"]),
                                  pose_source="command", branch=OUT / "scene")
            try:
                room.layout(seed, absent=False, furnished=True)
                robot = RobotGeometry(room.model)
                routes = {}
                for route, commands in SEQUENCES.items():
                    if not commands:
                        continue
                    variants = {}
                    for label, offset in POSE_OFFSETS.items():
                        pose = estimated + np.asarray(offset)
                        gap = minimum_swept_point_gap(
                            robot, qpos, pose, commands, points)
                        voxel = prior_row["layouts"][
                            "head_rear_pair_right_upper"]["candidate_routes"][
                                route]["pose_samples"][label]
                        evaluator = None
                        if seed == 170 and route in ("back2_diag2",
                                                         "back2_diag3"):
                            evaluator = next(
                                row["minimum_scene_gap_m"] for row in
                                v104["rows"] if row["case"] == case and
                                row["sequence"] == route and
                                row["start_perturbation"] ==
                                V104_PERTURBATIONS[label] and
                                row["executed_motion_factor"] == .55)
                        elif label == "estimated":
                            evaluator = next(
                                row["minimum_scene_gap_m"] for row in
                                v100["rows"] if row["seed"] == seed and
                                row["case"] == case and
                                row["sequence"] == route)
                        variants[label] = {
                            **gap,
                            "voxel_coverage_status": voxel["status"],
                            "evaluator_only_scene_gap_m": evaluator,
                        }
                    routes[route] = variants
                rows.append({
                    "seed": seed, "case": case,
                    "actor_range_source": range_source,
                    "actor_joint_source_path": prior_row["source_packet_path"],
                    "actor_joint_source_sha256": prior_row[
                        "source_packet_sha256"],
                    "right_upper_hit_count": len(points),
                    "routes": routes,
                })
            finally:
                room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-point-gap.v1",
        "source_sha256": sha(SOURCE),
        "point_gap_source_sha256": sha(Path(__file__).with_name(
            "pre_show_point_gap.py")),
        "v109_report_sha256": sha(V109 / "report.json"),
        "v100_report_sha256": sha(V100 / "report.json"),
        "v104_report_sha256": sha(V104 / "report.json"),
        "pose_samples_xyyaw": POSE_OFFSETS,
        "scope": "Actor-only continuous distance from current hypothetical right-upper range endpoints to commanded swept historical-body solids at 0/25/50/75/100pct of each step, across V109 candidate routes and five illustrative pose offsets. 5mm-noisy ranges clamp below4cm; one packet with15pct ray dropout. Compared after decision to 4cm voxel labels and V100/V104 evaluator scene gap. Point distance is a near-obstacle veto diagnostic, not free-space proof, person forecasting, a selected-body dynamics test or calibrated uncertainty bound.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    count = {"voxel_occupied_point_above_5cm": 0,
             "voxel_occupied_point_below_5cm": 0}
    for row in rows:
        for variants in row["routes"].values():
            for value in variants.values():
                if value["voxel_coverage_status"] == "occupied":
                    key = ("voxel_occupied_point_above_5cm" if
                           value["minimum_gap_m"] >= .05 else
                           "voxel_occupied_point_below_5cm")
                    count[key] += 1
    print(count)
    for row in rows:
        print(row["seed"], row["case"], {
            route: (variants["estimated"]["minimum_gap_m"],
                    variants["estimated"]["voxel_coverage_status"])
            for route, variants in row["routes"].items()})
    print(path)


if __name__ == "__main__":
    main()
