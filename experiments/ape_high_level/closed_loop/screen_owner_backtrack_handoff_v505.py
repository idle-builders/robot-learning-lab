"""Action-start latency and slip screen for the V502 owner backtrack request.

Replay the saved *actor-selected* request time and exact scripted owner pose
stream while varying only when and how far a hypothetical retreat executes.
MuJoCo signed distances score the result. This is evaluator sensitivity, not
a qualified gait, IMU estimator or physical stop-envelope measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

import robocasa_owner_person as person
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/skills"))
from cove_velocity_boundary import (  # noqa: E402
    CoveVelocityBoundary, EntryObservation, MotionRequest, PolicyIdentity,
)

SOURCE = ROOT / "artifacts/ape-robocasa-person-monitored-v502/report.json"
OLD_STOP = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
DESIGN = ROOT / "robot/ape-current-design.json"
OUT = ROOT / "artifacts/ape-robocasa-person-monitored-v502/backtrack-handoff-v505.json"
DELAYS_S = (0., .1, .2, .3, .5, 1.)
SPEED_FRACTIONS = (1., .7, .5, 0.)
REQUESTED_SPEED_M_S = .25
MAX_RETREAT_M = .3


def replay(ep, request_time, delay_s, speed_fraction):
    case = ep["scenario"]
    model = joined.scene_with_owner_and_prop(case["from_xy_m"])
    data = mujoco.MjData(model)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    prop_body = model.body("route_obstacle").id
    can_body = model.body("shown_can").id
    prop = motion(case)
    start = np.asarray(ep["route_episode"]["end_true_xy_m"])
    rows = [row for row in ep["trace"]
            if row["stage"] == "recognition_pending_monitor"]
    closest = {"signed_gap_m_capped_at_10cm": .1,
               "time_s": None, "geom": None}
    final_ape = start.copy()
    for row in rows:
        t = float(row["time_s"])
        pose = row["owner_pose_evaluator"]
        person.set_owner_pose(model, pose["torso_xy_m"], pose["hand_xyz_m"],
                              show_can=True)
        model.body_pos[can_body] = pose["can_xyz_m"]
        model.body_pos[prop_body, :2] = prop(t)
        retreat = min(MAX_RETREAT_M,
                      REQUESTED_SPEED_M_S * speed_fraction *
                      max(0., t - request_time - delay_s))
        final_ape = start + [0., -retreat]
        gap, near = clearance(model, data, ape_body, ape_geom, final_ape)
        if gap < closest["signed_gap_m_capped_at_10cm"]:
            closest = {"signed_gap_m_capped_at_10cm": gap,
                       "time_s": round(t, 4), "geom": near}
    return {"start_delay_s": delay_s,
            "executed_speed_m_s": REQUESTED_SPEED_M_S * speed_fraction,
            "retreat_distance_at_end_m": float(start[1] - final_ape[1]),
            "nearest": closest,
            "five_cm_margin_pass": closest["signed_gap_m_capped_at_10cm"] >= .05,
            "samples": len(rows)}


def native_boundary_screen(old):
    identity = PolicyIdentity(old["source_body_sha256"],
                              old["source_checkpoint_sha256"], 21, 20)
    gate = CoveVelocityBoundary(identity)
    entry = EntryObservation(0, "cove-50-direct-frame", identity.body_sha256,
                             True, True, True, True, .231)
    request = MotionRequest("owner-approach-backtrack", "backtrack", -.25, 0., 0.,
                            identity.body_sha256, identity.checkpoint_sha256,
                            deadline_ms=2000)
    return gate.submit(request, entry, now_ms=0)


def main():
    report = json.loads(SOURCE.read_text())
    old = json.loads(OLD_STOP.read_text())
    design = json.loads(DESIGN.read_text())
    cases = []
    for ep in report["episodes"]:
        if not ep["response"].startswith("approach_ignore"):
            continue
        request = next(row for row in ep["trace"]
                       if row.get("actor_action") == "request_backtrack_skill_unverified")
        request_time = float(request["time_s"])
        variants = [replay(ep, request_time, delay, fraction)
                    for delay in DELAYS_S for fraction in SPEED_FRACTIONS]
        cases.append({"response": ep["response"],
                      "request_time_s": request_time,
                      "monitored_episode_video_sha256": ep["video"]["sha256"],
                      "variants": variants})
        print(ep["response"],
              [(v["start_delay_s"], v["nearest"]["signed_gap_m_capped_at_10cm"])
               for v in variants if v["executed_speed_m_s"] == REQUESTED_SPEED_M_S],
              flush=True)
    current_pointer_hash = sha256(DESIGN)
    result = {"source_sha256": sha256(Path(__file__)),
              "v502_report_sha256": sha256(SOURCE),
              "old_native_stop_trace_sha256": sha256(OLD_STOP),
              "current_design_pointer_sha256": current_pointer_hash,
              "current_design_status": design.get("status"),
              "old_native_stop_pointer_matches_current": (
                  old["source_pointer_sha256"] == current_pointer_hash),
              "old_forward_stop_delay_s_range": old["summary"]["observed_stop_delay_s_range"],
              "old_forward_path_to_stop_m_range": old["summary"]["path_to_observed_stop_m_range"],
              "existing_velocity_boundary_reply_to_backtrack": native_boundary_screen(old),
              "scope": "independent MuJoCo geometry sensitivity using V502 owner motion/request times and hypothetical backward execution speed/delay; old native forward-stop evidence retained only as nontransferable reference; no backward native policy or physical safety claim",
              "cases": cases}
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
