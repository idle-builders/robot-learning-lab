"""Replay saved planar packet integrity and independently score 3-D route gaps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, evaluator_scene
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-spec-planar-route-v202p"
REPORT = OUT / "report.json"
MARGIN_M = .05


def audit_episode(case: dict, profile: dict) -> dict:
    model, data, body_id, geom_id = evaluator_scene(case["obstacle_xy_m"])
    trace = case["trace"]
    packet_checks = []
    predicted_xy = np.array((1.05, -4.20), dtype=float)
    for row in trace:
        path = ROOT / row["packet_path"]
        with np.load(path, allow_pickle=False) as packet:
            ranges = packet["ranges_m"]
            status = packet["status"]
            captured = float(packet["capture_time_s"])
            decided = float(packet["decision_time_s"])
            saved_pose = packet["estimated_pose_xyyaw"]
        counts = {"valid_hit": int(np.count_nonzero(status == 0)),
                  "too_near": int(np.count_nonzero(status == 1)),
                  "no_target": int(np.count_nonzero(status == 2))}
        valid = status == 0
        expected_tick = math.floor((decided + 1e-9) * profile["scan_hz"])
        packet_checks.append({
            "step": row["step"],
            "hash_match": sha(path) == row["packet_sha256"],
            "beam_count_match": len(status) == profile["beams"],
            "status_counts_match": counts == row["status_counts"],
            "invalid_is_not_range": bool(np.all(np.isnan(ranges[~valid]))),
            "valid_is_finite_and_bounded": bool(np.all(
                np.isfinite(ranges[valid]) &
                (ranges[valid] >= profile["minimum_m"]) &
                (ranges[valid] <= 5.02))),
            "capture_tick_match": abs(captured - expected_tick /
                                      profile["scan_hz"]) < 1e-5,
            "age_match": abs(decided - captured - row["capture_age_s"]) < 1e-5,
            "prediction_pose_match": bool(np.allclose(saved_pose[:2],
                                                        predicted_xy,
                                                        atol=1e-5)),
        })
        if row["requested_xy_m"] is not None:
            predicted_xy = (np.asarray(row["estimated_xy_m"]) +
                            np.asarray(row["command_delta_m"]))
    poses = [np.asarray(row["actual_xy_m"]) for row in trace]
    closest = {"gap_m": .1, "step": None, "substep": None,
               "nearest_geom": None}
    for i, pose in enumerate(poses):
        positions = [pose]
        if i + 1 < len(poses):
            positions += [pose + f * (poses[i+1]-pose)
                          for f in (.25, .5, .75)]
        for j, location in enumerate(positions):
            gap, geom = clearance(model, data, body_id, geom_id, location)
            if gap < closest["gap_m"]:
                closest = {"gap_m": round(gap, 6), "step": i,
                           "substep": j, "nearest_geom": geom}
    video_path = ROOT / case["video"]["path"]
    reader = imageio.get_reader(video_path)
    frame_count = reader.count_frames()
    reader.close()
    return {
        "device": case["device"], "obstacle_xy_m": case["obstacle_xy_m"],
        "actor_outcome": case["outcome"], "steps": len(trace),
        "packet_count": len(packet_checks),
        "packet_checks_pass": all(all(v for k, v in check.items()
                                      if k != "step") for check in packet_checks),
        "first_packet_failure": next((check for check in packet_checks
                                      if not all(v for k, v in check.items()
                                                 if k != "step")), None),
        "minimum_signed_clearance": closest,
        "margin_pass": closest["gap_m"] >= MARGIN_M,
        "video_hash_match": sha(video_path) == case["video"]["sha256"],
        "video_frames_match": frame_count == case["video"]["frames"] == len(trace),
        "unknown_return_count": sum(r["status_counts"]["too_near"] +
                                    r["status_counts"]["no_target"]
                                    for r in trace),
    }


def main() -> None:
    report = json.loads(REPORT.read_text())
    rows = [audit_episode(case, report["device_profiles"][case["device"]])
            for case in report["episodes"]]
    summary = {
        "schema": "robot-learning-lab.ape-spec-planar-route-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_spec_planar_route_v202p.py")),
        "matched_scene_fingerprints": all(
            len({case["scene_fingerprint"] for case in report["episodes"]
                 if case["obstacle_xy_m"] == obstacle}) == 1
            for obstacle in (None, [4.4, -2.9])),
        "episodes": rows,
    }
    summary["passed"] = (summary["experiment_source_match"] and
                         summary["matched_scene_fingerprints"] and
                         all(row["packet_checks_pass"] and
                             row["video_hash_match"] and
                             row["video_frames_match"]
                             for row in rows))
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(row["device"], row["obstacle_xy_m"], row["actor_outcome"],
              row["steps"], row["minimum_signed_clearance"]["gap_m"],
              row["packet_checks_pass"], flush=True)
    print("audit passed", summary["passed"])
    if not summary["passed"]:
        raise AssertionError("Saved packet/video audit failed")


if __name__ == "__main__":
    main()
