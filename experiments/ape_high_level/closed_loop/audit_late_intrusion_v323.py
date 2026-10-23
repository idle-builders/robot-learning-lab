"""Evaluator-only continuation after V323's guarded forward-step refusal.

The controller does not receive these future clearances. They expose whether
the moving prop still threatens a robot that has correctly stopped routing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_blocked_range_mounts_v319 import add_packet
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_variable_slip_localized_v301 import SLIPS


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)


def main() -> None:
    rows = []
    for seed in SEEDS:
        trace_path = (OUT / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        decision_time = trace["recovery_inputs"][0]["head_inputs"][-1][
            "capture_time_s"]
        actual = np.asarray(next(e["evaluator_actual_pose_xyyaw"]
                                 for e in reversed(trace["controller_events"])
                                 if "evaluator_actual_pose_xyyaw" in e))
        next_slip = SLIPS[len(trace["slips_evaluator_only"]) % len(SLIPS)]
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command",
                                branch=trace_path.parent)
        try:
            room.layout(seed, absent=False, furnished=True)
            policies = {}
            actions = {
                "hold_base": (0., 0., 0.),
                "counterfactual_forward_8cm": (.08, 0., 0.),
                "counterfactual_backward_8cm": (-.08, 0., 0.),
                "counterfactual_away_8cm": (0., -.08, 0.),
                "counterfactual_toward_8cm": (0., .08, 0.),
                "counterfactual_turn_away_45deg": (0., 0., -math.pi / 4),
            }
            for policy, (dx, dy, dyaw) in actions.items():
                closest = (float("inf"), None, "", "")
                for fraction in np.linspace(0., 1., 21):
                    room.time_s = decision_time + .4 * fraction
                    room._update_moving_obstacle()
                    room.actual[:] = actual
                    room.actual[0] += dx * next_slip * fraction
                    room.actual[1] += dy * next_slip * fraction
                    room.actual[2] += dyaw * fraction
                    room._forward()
                    gap, robot_geom, obstacle_geom = (
                        room._current_obstacle_clearance())
                    if gap < closest[0]:
                        closest = (gap, room.time_s,
                                   robot_geom, obstacle_geom)
                policies[policy] = {
                    "minimum_model_clearance_m": round(closest[0], 5),
                    "time_of_minimum_s": round(closest[1], 4),
                    "robot_geom": closest[2],
                    "obstacle_geom": closest[3]}
            head_inputs = trace["recovery_inputs"][0]["head_inputs"]
            voxel_map = DepthVoxelMap()
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            last = None
            for entry in head_inputs:
                with np.load(ROOT / entry["actor_frame_path"]) as frame:
                    last = {key: frame[key].copy() for key in frame.files}
                voxel_map.observe(last["map_depth_m"],
                                  (last["camera_origin_estimated_m"],
                                   last["camera_rotation_estimated"]),
                                  float(last["capture_time_s"]),
                                  fovy_deg=fovy, stride=4,
                                  max_depth_m=.5, self_masked=True)
            packet_path = ROOT / trace["recovery_inputs"][0][
                "actor_gate"]["packet_path"]
            packets = json.loads(packet_path.read_text())["ray_packets"]
            for rays in packets.values():
                add_packet(voxel_map, rays, decision_time)
            robot = RobotGeometry(room.model)
            pose, qpos = (last["estimated_pose_xyyaw"],
                          last["actor_qpos"])
            actor_coverage = {}
            for name, (dx, dy, dyaw) in actions.items():
                if dx == 0. and dy == 0.:
                    continue
                future = pose.copy()
                future[:2] += (dx, dy)
                actor_coverage[name] = robot.swept_exterior_coverage(
                    voxel_map, qpos, pose, future, decision_time,
                    margin_m=.05)
            rows.append({"seed": seed,
                         "actor_gate_status": trace["recovery_inputs"][0][
                             "actor_gate"]["status"],
                         "decision_time_s": decision_time,
                         "next_hidden_slip_evaluator_only": next_slip,
                         "counterfactual_policies": policies,
                         "actor_upper_shell_coverage_at_decision":
                         actor_coverage,
                         "actor_packet_sha256": sha(packet_path),
                         "trace_path": str(trace_path.relative_to(ROOT)),
                         "trace_sha256": sha(trace_path)})
            print(seed, policies, flush=True)
        finally:
            room.renderer.close()
    report = {"scope": "Evaluator-only 20ms sampled, 0.4s continuation from each V323 blocked decision. The scripted 8cm prop keeps approaching at 0.24m/s. Compare holding with forward, backward, away/toward 8cm and away 45deg body-turn counterfactuals under the next hidden slip. The same retained actor ray packets independently show whether the translational alternatives have known-free upper-shell coverage. Future clearance truth never enters the actor and does not stand in for a native stop/evasion controller.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "late-intrusion-continuation-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
