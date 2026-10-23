"""V365: diagnose global vs body-relative offer-zone error on V364 packets.

This is a revealed-data coordinate/threshold audit. It proposes an egocentric
interface; it does not rerun the live controller or qualify uncertainty.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run import ROOT, sha
from run_shin_intrusion_posture_v324 import actual_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v365"
V364 = ROOT / "artifacts/ape-closed-loop-cove50-v364/offer-lane-challenge.json"
V360 = ROOT / "artifacts/ape-closed-loop-cove50-v360"
REQUESTED_LOCAL_ITEM_XY_M = np.asarray((.45, 0.))
TOLERANCES_M = (.06, .07, .08, .10)


def body_local(world_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    yaw = float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation_inv = np.asarray(((c, s), (-s, c)))
    return rotation_inv @ (world_xy - pose[:2])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(V364.read_text())
    poses = {}
    for seed in (170, 172):
        branch = (V360 / "compliant_far" / f"seed{seed}-present" /
                  f"seed{seed}-present")
        trace_path = branch / "episode-trace.json"
        invite_path = branch / "handoff-invitation.npz"
        with np.load(invite_path) as packet:
            estimate = packet["estimated_pose_xyyaw"].copy()
        actual = actual_pose(json.loads(trace_path.read_text()))
        poses[seed] = {"estimated": estimate, "actual": actual,
                       "source_invitation_sha256": sha(invite_path),
                       "source_trace_sha256": sha(trace_path)}
    rows = []
    for original in prior["rows"]:
        seed = original["seed"]
        actor_world = original["actor_choice"]["accepted_world_xyz_m"]
        if actor_world is None:
            raise AssertionError("V364 accepted point missing")
        actor_local = body_local(np.asarray(actor_world[:2]),
                                 poses[seed]["estimated"])
        true_local = body_local(np.asarray(
            original["evaluator_true_item_xy_m"]), poses[seed]["actual"])
        actor_error = float(np.linalg.norm(
            actor_local - REQUESTED_LOCAL_ITEM_XY_M))
        true_error = float(np.linalg.norm(
            true_local - REQUESTED_LOCAL_ITEM_XY_M))
        rows.append({"case": original["case"],
                     "source_actor_packet_sha256": original[
                         "actor_packet_sha256"],
                     "actor_item_local_xy_m": actor_local.round(5).tolist(),
                     "evaluator_item_local_xy_m": true_local.round(5).tolist(),
                     "actor_local_zone_error_m": round(actor_error, 5),
                     "evaluator_local_zone_error_m": round(true_error, 5),
                     "actor_localization_error_m": round(float(np.linalg.norm(
                         actor_local - true_local)), 5)})
    scores = []
    for tolerance in TOLERANCES_M:
        false_accept = false_reask = true_accept = true_reask = 0
        for row in rows:
            actor_yes = row["actor_local_zone_error_m"] <= tolerance
            true_yes = row["evaluator_local_zone_error_m"] <= .10
            if actor_yes and true_yes:
                true_accept += 1
            elif actor_yes:
                false_accept += 1
            elif true_yes:
                false_reask += 1
            else:
                true_reask += 1
        scores.append({"actor_tolerance_m": tolerance,
                       "evaluator_zone_tolerance_m": .10,
                       "true_accept": true_accept,
                       "false_accept": false_accept,
                       "false_reask": false_reask,
                       "true_reask": true_reask})
    report = {
        "scope": "Read-only revealed-data comparison of V364 actor accepted-item world estimates converted through estimated Cove-50 base pose to body-relative XY, with true item positions converted through evaluator base pose. Hypothetical requested item point is (0.45,0)m body-relative after the V360 retreat; no actual new owner instruction or live controller run. Sweep actor point tolerance6/7/8/10cm against a 10cm evaluator spatial requested-zone label. A smaller point threshold is a diagnostic tradeoff, not a calibrated uncertainty bound or physical authorization.",
        "source_sha256": sha(SOURCE),
        "v364_report_sha256": sha(V364),
        "poses": {str(k): {**{n: v.tolist() for n, v in p.items()
                           if isinstance(v, np.ndarray)},
                            **{n: v for n, v in p.items()
                               if not isinstance(v, np.ndarray)}}
                  for k, p in poses.items()},
        "requested_local_item_xy_m": REQUESTED_LOCAL_ITEM_XY_M.tolist(),
        "scores": scores, "rows": rows}
    (OUT / "relative-offer-zone-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for score in scores:
        print(score, flush=True)


if __name__ == "__main__":
    main()
