"""Replay 20 ms moving-prop paths with independent MuJoCo cylinder distance."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import evaluator_scene, sha256
from run_robocasa_moving_prop_v423 import obstacle_xy


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-moving-prop-v423"


def audit_episode(episode: dict) -> dict:
    model, data, ape_body, ape_geom = evaluator_scene([4.95, -3.34])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    witness = []
    for row in episode["trace"]:
        start = np.asarray(row["actual_xy_m"], dtype=float)
        requested = row.get("requested_xy_m")
        if requested is None:
            continue
        delta = (np.asarray(requested) - np.asarray(row["estimated_xy_m"])) * (1. + .30 / 5.89)
        for index, substep in enumerate(row["motion_substeps"]):
            fraction = (index if substep["stopped_before_move"] else index + 1) / 20
            pose = start + fraction * delta
            prop = obstacle_xy(substep["time_s"])
            model.body_pos[ape_body, :2] = pose
            model.body_pos[prop_body, :2] = prop
            mujoco.mj_forward(model, data)
            distance = float(mujoco.mj_geomDistance(model, data, ape_geom, prop_geom, .5, None))
            # MuJoCo returns its largest float for pairs beyond distmax.
            # Only compare exact gap where the two solids are within 50 cm.
            agreement = abs(distance - substep["true_prop_gap_m"]) if distance < .5 else None
            witness.append({"step": row["step"], "time_s": substep["time_s"],
                            "signed_distance_m": distance,
                            "trace_gap_m": substep["true_prop_gap_m"],
                            "distance_agreement_m": agreement,
                            "actor_guard_range_m": substep["actor_guard_range_m"],
                            "stopped_before_move": substep["stopped_before_move"]})
    closest = min(witness, key=lambda row: row["signed_distance_m"])
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    return {"monitor": episode["monitor"], "outcome": episode["outcome"],
            "closest": closest,
            "minimum_signed_prop_distance_m": closest["signed_distance_m"],
            "required_5cm_margin_pass": closest["signed_distance_m"] >= .05,
            "distance_replay_max_error_m": max((row["distance_agreement_m"] for row in witness
                                                if row["distance_agreement_m"] is not None),
                                               default=0.),
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frames_pass": frames == episode["video"]["frames"] == len(episode["trace"]) + 1,
            "motion_substeps": len(witness)}


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_moving_prop_v423.py"
    cases = [audit_episode(episode) for episode in report["episodes"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "source_sha_pass": report["source_sha256"] == sha256(source),
              "scene_pair_pass": len({episode["scene_fingerprint"] for episode in report["episodes"]}) == 1,
              "cases": cases,
              "paired_expected_result_pass": (
                  cases[0]["monitor"] is False and not cases[0]["required_5cm_margin_pass"] and
                  cases[1]["monitor"] is True and cases[1]["required_5cm_margin_pass"]),
              "all_video_replay_pass": all(case["video_sha_pass"] and case["video_frames_pass"] and
                                           case["distance_replay_max_error_m"] < .001 for case in cases)}
    (OUT / "audit-v424.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print("monitor", case["monitor"], case["outcome"],
              "signed distance", round(case["minimum_signed_prop_distance_m"], 4),
              "substeps", case["motion_substeps"], flush=True)
    print("paired_expected_result_pass", output["paired_expected_result_pass"],
          "all_video_replay_pass", output["all_video_replay_pass"], flush=True)


if __name__ == "__main__":
    main()
