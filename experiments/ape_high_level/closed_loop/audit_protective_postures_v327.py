"""V327 evaluator-only posture screen on V326's hazardous local geometries.

This chooses an *experimental* shoulder-pitch candidate for a later actor
test. Scene geometry and clearance never become controller observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, END_S, FLEX_TIME_S, START_S, actual_pose,
)
from audit_shin_intrusion_geometry_v326 import vary_prop


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v327"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V326 = ROOT / "artifacts/ape-closed-loop-cove50-v326/shin-intrusion-geometry-audit.json"
POLICIES = ("hold", "elbow_flex", "shoulder_pitch_back",
            "shoulder_pitch_forward", "shoulder_roll_in")


def self_clearance(room: LateCorridorRoom, side: str) -> tuple[float, str, str]:
    moving = [room.model.geom(name).id for name in (
        f"{side}_forearm_frame_budget", f"{side}_hand", f"{side}_finger_tip")]
    other = "right" if side == "left" else "left"
    against = [room.model.geom(name).id for name in (
        "torso_frame_compute_budget", "pelvis_frame_budget",
        f"{other}_upper_arm_frame_budget",
        f"{other}_forearm_frame_budget", f"{other}_hand",
        f"{other}_finger_tip")]
    # Head geoms are visual-only in the current model. Include them in this
    # geometric screen rather than claiming native contact would catch them.
    against += [gid for gid in range(room.model.ngeom)
                if room.model.geom_bodyid[gid] in (
                    room.model.body("head").id,
                    room.model.body("head_yaw_body").id)]
    return min((float(mujoco.mj_geomDistance(room.model, room.data, a, b,
                                             10., None)),
                room.model.geom(a).name, room.model.geom(b).name)
               for a in moving for b in against)


def request(side: str, policy: str) -> tuple[str | None, float]:
    if policy == "hold":
        return None, 0.
    if policy == "elbow_flex":
        return f"{side}_elbow_pitch", -.8
    if policy == "shoulder_pitch_back":
        return f"{side}_shoulder_pitch", -.8
    if policy == "shoulder_pitch_forward":
        return f"{side}_shoulder_pitch", .8
    if policy == "shoulder_roll_in":
        return f"{side}_shoulder_roll", -.8 if side == "left" else .8
    raise ValueError(policy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(V326.read_text())
    hazardous = [row for row in geometry["rows"]
                 if row["evaluator_outcomes"]["hold"][
                     "minimum_evaluator_clearance_m"] < .05]
    rows = []
    for prior in hazardous:
        seed = prior["seed"]
        trace_path = (V323 / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        actual = actual_pose(trace)
        side = "right" if prior["mirrored_evaluator_setup"] else "left"
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=OUT)
        try:
            room.layout(seed, absent=False, furnished=True)
            room.actual[:] = actual
            outcomes = {}
            for policy in POLICIES:
                joint, target = request(side, policy)
                address = (int(room.model.joint(joint).qposadr[0])
                           if joint else None)
                scene_min = (float("inf"), "", "")
                self_min = (float("inf"), "", "")
                for time_s in np.linspace(START_S, END_S, 41):
                    vary_prop(room, float(time_s),
                              mirrored=prior["mirrored_evaluator_setup"],
                              x_offset_m=prior["prop_x_offset_m_evaluator_only"],
                              height_m=prior["prop_height_m_evaluator_only"],
                              base_y=actual[1])
                    room.data.qpos[18:27] = 0.
                    if address is not None:
                        room.data.qpos[address] = target * np.clip(
                            (time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
                    room._forward()
                    gap, body, obstacle = room._current_obstacle_clearance()
                    scene_min = min(scene_min, (gap, body, obstacle))
                    self_min = min(self_min, self_clearance(room, side))
                outcomes[policy] = {
                    "joint": joint, "target_rad": target,
                    "minimum_scene_clearance_m": round(scene_min[0], 5),
                    "closest_scene_geoms": [scene_min[1], scene_min[2]],
                    "minimum_self_clearance_m": round(self_min[0], 5),
                    "closest_self_geoms": [self_min[1], self_min[2]],
                    "scene_clear_5cm_and_self_nonoverlap": (
                        scene_min[0] >= .05 and self_min[0] >= 0.)}
            rows.append({"case": prior["case"], "side": side,
                         "evaluator_outcomes": outcomes})
            print(prior["case"], " ".join(
                f"{policy}:{outcomes[policy]['minimum_scene_clearance_m']:.3f}/"
                f"{outcomes[policy]['minimum_self_clearance_m']:.3f}"
                for policy in POLICIES), flush=True)
        finally:
            room.close()
    report = {"scope": "Evaluator-only 20ms geometric screen of five direct-qpos posture surrogates on the 12 V326 scenarios whose hold clearance is below5cm. Self geometry includes forearm/hand/finger against torso/pelvis, other arm and visual-only head. This is not actuator, dynamics, balance or native contact proof. Posture selection based on these revealed cases needs a separate frozen actor test.",
              "source_sha256": sha(SOURCE), "v326_audit_sha256": sha(V326),
              "selected_visual_sha256": sha(selected_visual()),
              "summary": {policy: {
                  "scene_clear_and_self_nonoverlap": sum(
                      row["evaluator_outcomes"][policy][
                          "scene_clear_5cm_and_self_nonoverlap"] for row in rows),
                  "cases": len(rows)} for policy in POLICIES},
              "rows": rows}
    (OUT / "protective-postures-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
