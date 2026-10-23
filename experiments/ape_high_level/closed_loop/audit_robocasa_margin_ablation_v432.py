"""Independent all-fixture geometry and video audit of V431 margin routes."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, evaluator_scene, sha256
from run_robocasa_layout2_route_v407 import GOAL
from run_robocasa_moving_prop_v423 import obstacle_xy


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-margin-ablation-v431"


def score_route(episode: dict) -> dict:
    model, data, ape_body, ape_geom = evaluator_scene([4.95, -3.34])
    prop_body = model.body("route_obstacle").id
    witness = []
    for row in episode["trace"]:
        start = np.asarray(row["actual_xy_m"], dtype=float)
        requested = row.get("requested_xy_m")
        delta = ((np.asarray(requested) - np.asarray(row["estimated_xy_m"]))
                 * (1. + .30 / 5.89)) if requested is not None else np.zeros(2)
        samples = [(row["time_s"], start)]
        for index, substep in enumerate(row["motion_substeps"]):
            fraction = (index if substep["stopped_before_move"] else index + 1) / 20
            samples.append((substep["time_s"], start + fraction * delta))
        for time_s, pose in samples:
            model.body_pos[prop_body, :2] = obstacle_xy(time_s)
            mujoco.mj_forward(model, data)
            gap, fixture = clearance(model, data, ape_body, ape_geom, pose)
            witness.append({"time_s": time_s, "xy_m": pose.tolist(),
                            "minimum_fixture_gap_capped_m": gap,
                            "nearest_geom": fixture})
    closest = min(witness, key=lambda row: row["minimum_fixture_gap_capped_m"])
    actual_end = np.asarray(episode["end_true_xy_m"])
    return {"outcome": episode["outcome"],
            "end_actual_goal_gap_m": float(np.linalg.norm(actual_end - GOAL)),
            "closest": closest,
            "all_body_5cm_margin_pass": closest["minimum_fixture_gap_capped_m"] >= .05,
            "geometry_samples": len(witness)}


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    cases = []
    for case in report["cases"]:
        geometry = score_route(case["episode"])
        video = case["immediate_audit"]
        cases.append({"margin_m": case["margin_m"],
                      "outcome": case["episode"]["outcome"],
                      "route_geometry": geometry,
                      "video_sha_pass": video["video_sha_pass"],
                      "video_frames_pass": video["video_frames_pass"],
                      "prop_distance_replay_pass": video["distance_replay_max_error_m"] < .001,
                      "post_stop_margin_pass": (case["post_stop"]["margin_held_through_motion"]
                                                if case["post_stop"] is not None else None)})
        print(case["margin_m"], geometry["outcome"],
              round(geometry["end_actual_goal_gap_m"], 3),
              round(geometry["closest"]["minimum_fixture_gap_capped_m"], 3),
              geometry["closest"]["nearest_geom"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_margin_ablation_v431.py")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_prop_checks_pass": all(case["video_sha_pass"] and
                                                case["video_frames_pass"] and
                                                case["prop_distance_replay_pass"]
                                                for case in cases),
              "cases": cases}
    (OUT / "audit-v432.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
