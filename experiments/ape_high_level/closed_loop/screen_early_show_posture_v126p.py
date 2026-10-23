"""V126p: early meeting posture geometry screen in the V98 owner scene.

Evaluator-only direct joint interpolation; no learned joint skill or
actor-visible clearance is inferred from this screen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
import run_pre_show_reserve_screen_v100 as v100


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-posture"
V98 = v100.V98
SEED = 170
CASES_TO_SCREEN = ("clear", "hand_close")
POSTURES = {
    "neutral": (0., 0.),
    "left_n06": (-.6, 0.),
    "left_p06": (.6, 0.),
    "right_n06": (0., -.6),
    "right_p06": (0., .6),
    "both_n06": (-.6, -.6),
    "both_p06": (.6, .6),
    "left_n_right_p": (-.6, .6),
    "left_p_right_n": (.6, -.6),
}
START_S = 16.2
PREP_END_S = 16.6
SHOW_END_S = 17.0
END_S = 18.2
STEP_S = .02


def one(room: OfferDriftRoom, base_qpos: np.ndarray,
        estimate: np.ndarray, case: str, label: str,
        target: tuple[float, float]) -> dict:
    room.layout(SEED, absent=False, furnished=True)
    obstacle = CASES[case]
    if obstacle is not None:
        position, size = obstacle
        room.model.body_pos[room.model.body("loop_box").id] = position
        room.model.geom("loop_box_geom").size[:] = size
    else:
        room.model.body_pos[room.model.body("loop_box").id] = (10., 0., .1)
    room.data.qpos[:] = base_qpos
    room.actual[:] = (base_qpos[0], base_qpos[1],
                      2 * math.atan2(base_qpos[6], base_qpos[3]))
    room.estimated[:] = estimate
    room.head = float(base_qpos[room.head_addr])
    left_addr = room.model.joint("left_shoulder_pitch").qposadr[0]
    right_addr = room.model.joint("right_shoulder_pitch").qposadr[0]
    initial = (float(base_qpos[left_addr]), float(base_qpos[right_addr]))
    minimum = (float("inf"), None, None, None)
    phase_minima = {"prepare": float("inf"), "owner_show": float("inf"),
                    "look_and_hold": float("inf")}
    first_below_5cm = None
    for tick in range(int(round((END_S-START_S)/STEP_S))+1):
        t = START_S + STEP_S*tick
        prep = min(1., max(0., (t-START_S)/(PREP_END_S-START_S)))
        room.data.qpos[left_addr] = initial[0] + prep*(target[0]-initial[0])
        room.data.qpos[right_addr] = initial[1] + prep*(target[1]-initial[1])
        offer = min(1., max(0., (t-PREP_END_S)/(SHOW_END_S-PREP_END_S)))
        hand = START_HAND_XYZ_M + offer*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
        set_person_pose(room.model, hand, carried_item="loop_TomatoSauce")
        if t >= SHOW_END_S-1e-6:
            room.model.body_pos[room.model.body("loop_TomatoSauce").id] = (
                owner.OWNER_SHOW_XYZ_M)
        room.obstacle_xy[3] = (0., v100.prop_y(t))
        room.model.body_pos[room.model.body("loop_stool").id] = (
            0., v100.prop_y(t), STOOL_SIZE[2])
        room.time_s = t
        room._forward()
        gap, robot, obstacle = room._current_obstacle_clearance()
        phase = ("prepare" if t < PREP_END_S-1e-6 else
                 "owner_show" if t < SHOW_END_S-1e-6 else
                 "look_and_hold")
        phase_minima[phase] = min(phase_minima[phase], gap)
        if gap < minimum[0]:
            minimum = (float(gap), t, robot, obstacle)
        if first_below_5cm is None and gap < .05:
            first_below_5cm = t
    return {
        "seed": SEED, "case": case, "posture": label,
        "left_right_shoulder_pitch_rad": list(target),
        "minimum_scene_gap_m": round(minimum[0], 5),
        "minimum_time_s": round(minimum[1], 4),
        "closest_pair": list(minimum[2:]),
        "phase_minima_m": {key: round(value, 5)
                           for key, value in phase_minima.items()},
        "first_below_5cm_s": (round(first_below_5cm, 4)
                              if first_below_5cm is not None else None),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"] if row["condition"] == "moving"
                 and row["case"][0] == SEED)
    source = (V98 / "moving" / f"seed{SEED}-present" /
              f"seed{SEED}-present/side-probe-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
        assert abs(float(packet["capture_time_s"])-7.6) < 1e-5
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rows = []
    try:
        room = OfferDriftRoom(case=tuple(prior["case"]),
                              pose_source="command", branch=OUT)
        try:
            for case in CASES_TO_SCREEN:
                for label, target in POSTURES.items():
                    row = one(room, qpos, estimate, case, label, target)
                    rows.append(row)
                    print(case, label, row["minimum_scene_gap_m"],
                          row["closest_pair"], flush=True)
        finally:
            room.close()
    finally:
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-show-posture-screen.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "source_packet_path": str(source.relative_to(ROOT)),
        "source_packet_sha256": sha(source),
        "selected_visual_sha256": sha(selected_visual()),
        "scope": "Evaluator-only full-body clearance screen from the saved "
        "7.6s early staging pose. The owner waits during a direct 0.4s "
        "shoulder transition, shows over 0.4s, then the ape holds to18.2s "
        "while the fixed moving prop arrives. Checks every20ms in the "
        "original furnished articulated-owner scene and clear/hand-height "
        "side-block cases. No actor perception, image identity, motor speed, "
        "dynamic contact or selected physical body is qualified.",
        "rows": rows,
    }
    (OUT / "screen.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "screen.json")


if __name__ == "__main__":
    main()
