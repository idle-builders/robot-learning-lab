"""V312: sensitivity of two-view moving-prop prediction to ego-pose error.

The alternate estimated poses are counterfactual actor beliefs, not truth fed
to the planner. Truth is used only to place the fixed error sweep and score it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_moving_cluster_prediction_v311 import (
    depth_cloud, forecast_gap, groups, MATCH_M, MIN_CLUSTER_POINTS,
)
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import CASES, HeldoutMovingRoom
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v312"
V308 = ROOT / "artifacts/ape-closed-loop-cove50-v308/moving-depth-action-audit.json"
V311 = ROOT / "artifacts/ape-closed-loop-cove50-v311/moving-cluster-prediction-audit.json"
# Same rendered physical scene and first estimate. Only the second estimated
# lateral pose changes: under, perfect, original command, larger overshoot.
ESTIMATED_Y_M = (.02, .0464, .08, .11, .14)
ACTOR_MARGIN_M = .065


def main() -> None:
    truth = {r["case"][0]: r for r in json.loads(V308.read_text())["rows"]}
    previous_run = {r["case"][0]: r for r in json.loads(V311.read_text())["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=OUT)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H,
                             agreement_m=.04)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            old_points, _ = depth_cloud(room, mask, seed, 7.2, 0., 0., 12)
            old = groups(old_points)
            alternatives = {}
            for estimated_y in ESTIMATED_Y_M:
                current_points, qpos = depth_cloud(
                    room, mask, seed, 7.6, .0464, estimated_y, 22)
                current = groups(current_points)
                velocities = []
                track = []
                for cluster in current:
                    center = cluster[:, :2].mean(axis=0)
                    candidates = [(float(np.linalg.norm(
                        center - earlier[:, :2].mean(axis=0))), earlier)
                                  for earlier in old if len(cluster) >=
                                  MIN_CLUSTER_POINTS and len(earlier) >=
                                  MIN_CLUSTER_POINTS]
                    distance, match = (min(candidates, key=lambda pair: pair[0])
                                       if candidates else (float("inf"), None))
                    matched = match is not None and distance <= MATCH_M
                    speed = ((center - match[:, :2].mean(axis=0)) / .4
                             if matched else np.zeros(2))
                    velocities.append(speed)
                    track.append({"points": len(cluster),
                                  "matched": matched,
                                  "velocity_xy_m_s": speed.round(5).tolist()})
                qpos[room.head_addr] = 0.
                pose = np.array((0., estimated_y, 0.))
                predicted = forecast_gap(robot, qpos, pose, current,
                                         velocities, dynamic=True)
                static = forecast_gap(robot, qpos, pose, current,
                                      velocities, dynamic=False)
                if estimated_y == .08 and abs(predicted - previous_run[seed][
                        "actor_predicted_gap_m"]) > 1e-5:
                    raise AssertionError("Frozen V311 original-pose result changed")
                alternatives[str(estimated_y)] = {
                    "pose_error_m_evaluator_only": round(estimated_y - .0464, 5),
                    "track": track,
                    "actor_static_gap_m": round(static, 5),
                    "actor_predicted_gap_m": round(predicted, 5),
                    "actor_static_pass": static >= ACTOR_MARGIN_M,
                    "actor_predicted_pass": predicted >= ACTOR_MARGIN_M}
            rows.append({"case": case,
                         "evaluator_safe_5cm": truth[seed]["evaluator_safe_5cm"],
                         "evaluator_dynamic_clearance_m": truth[seed][
                             "evaluator_min_dynamic_clearance_m"],
                         "alternatives": alternatives})
            print(seed, mode, "safe", truth[seed]["evaluator_safe_5cm"],
                  {key: (value["actor_predicted_pass"],
                         value["actor_predicted_gap_m"],
                         value["track"][0]["velocity_xy_m_s"][1])
                   for key, value in alternatives.items()}, flush=True)
        finally:
            mask.close()
            room.close()
    scores = {}
    for estimate in ESTIMATED_Y_M:
        key = str(estimate)
        scores[key] = {
            "actions": len(rows),
            "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in rows),
            "false_safe": sum(row["alternatives"][key]["actor_predicted_pass"]
                              and not row["evaluator_safe_5cm"] for row in rows),
            "false_block": sum(not row["alternatives"][key]["actor_predicted_pass"]
                               and row["evaluator_safe_5cm"] for row in rows),
            "passed": sum(row["alternatives"][key]["actor_predicted_pass"]
                          for row in rows)}
    report = {"scope": "V312 same six V311 noisy side-depth cluster scenes and fixed true movement, but counterfactual second-frame actor lateral pose estimates 2/4.64/8/11/14cm from a zero first estimate. The actor receives only its chosen estimate, depth/encoders and time; true pose and independent full-body safe labels appear only in evaluator columns. A pose error changes current cluster and estimated ego motion, so apparent obstacle velocity can authorize an unsafe step. This is a diagnostic stress sweep, not a realistic localization-error distribution or a live native route. The 4.64cm estimate is an oracle-pose bound, not a deployable sensor result.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "v308_sha256": sha(V308), "v311_sha256": sha(V311),
              "estimated_y_m": ESTIMATED_Y_M,
              "rows": rows, "scores": scores}
    (OUT / "motion-pose-coupling-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
