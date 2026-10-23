"""V94: geometry screen of a predeclared rightward response to V93's cue.

This is an evaluator feasibility screen from the saved V92 16.4s state, not
an actor-chosen or motor-controlled escape. It keeps the articulated owner and
low prop moving through the full camera-look horizon and scores all scene
clearances every 20ms for hold, 4cm and 8cm rightward side commands.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import (CASES, OUT as V92,
                                                  OfferDriftRoom)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v94"
START_S = 16.4
ACTION_DURATION_S = .4
END_S = 17.4
COMMANDS_M = (0., .04, .08)
ACTUAL_SLIPS = (.55, .94)


def panel(room, label: str) -> np.ndarray:
    lower, _ = room.capture("lower_rgb", depth=False)
    image = Image.fromarray(np.concatenate(
        (np.asarray(lower), np.asarray(room.capture_external())), axis=1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill="white")
    return np.asarray(image)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((V92 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in report["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            packet = (V92 / f"moving/seed{seed}-present/seed{seed}-present/"
                      "articulated-owner-02.npz")
            with np.load(packet) as data:
                qpos = data["actor_qpos"].copy()
                estimated = data["estimated_pose_xyyaw"].copy()
                capture_time = float(data["capture_time_s"])
            if abs(capture_time - START_S) > 1e-5:
                raise AssertionError("Owner packet is not at response start")
            initial_actual = np.array((qpos[0], qpos[1],
                                       2*math.atan2(qpos[6], qpos[3])))
            for command in COMMANDS_M:
                for slip in (ACTUAL_SLIPS if command else (1.,)):
                    room = OfferDriftRoom(
                        case=tuple(prior["case"]), pose_source="command",
                        branch=OUT / "scene")
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.data.qpos[:] = qpos
                        room.actual[:] = initial_actual
                        room.estimated[:] = estimated
                        room.head = float(qpos[room.head_addr])
                        label = (f"seed{seed}-hold" if command == 0 else
                                 f"seed{seed}-right{int(100*command)}cm-"
                                 f"slip{int(100*slip)}")
                        frames = []
                        first_breach = None
                        minimum = (float("inf"), 0., "", "")
                        video_path = OUT / f"{label}.mp4"
                        with imageio.get_writer(
                                video_path, fps=5, codec="libx264",
                                quality=7) as video:
                            for tick in range(51):
                                t = START_S + .02*tick
                                fraction = min(1., (t-START_S)/ACTION_DURATION_S)
                                room.actual[:] = initial_actual
                                room.actual[1] -= command*slip*fraction
                                room.estimated[:] = estimated
                                room.estimated[1] -= command*fraction
                                room.time_s = t
                                room._update_moving_obstacle()
                                hand_fraction = min(1., max(0.,
                                    (t - 16.2)/.4))
                                hand = (START_HAND_XYZ_M + hand_fraction *
                                        (FAR_HAND_XYZ_M - START_HAND_XYZ_M))
                                set_person_pose(room.model, hand,
                                                carried_item="loop_TomatoSauce")
                                room._forward()
                                gap = room._current_obstacle_clearance()
                                if gap[0] < minimum[0]:
                                    minimum = (float(gap[0]), t, gap[1], gap[2])
                                if first_breach is None and gap[0] < .05:
                                    first_breach = t
                                if tick % 5 == 0:
                                    frames.append({
                                        "time_s": round(t, 4),
                                        "evaluator_gap_m": round(float(gap[0]), 5),
                                        "actor_estimated_y_m": round(
                                            float(room.estimated[1]), 5),
                                        "evaluator_actual_y_m": round(
                                            float(room.actual[1]), 5)})
                                    video.append_data(panel(
                                        room, f"V94 {label} t{t:.1f} "
                                        f"gap{100*gap[0]:.1f}cm"))
                        rows.append({
                            "seed": seed,
                            "command_right_m": command,
                            "evaluator_actual_slip": slip,
                            "minimum_scene_gap_m": round(minimum[0], 5),
                            "minimum_time_s": round(minimum[1], 4),
                            "closest_pair": list(minimum[2:]),
                            "first_below_5cm_s": (
                                round(first_breach, 4) if first_breach else None),
                            "frames": frames,
                            "video_path": str(video_path.relative_to(ROOT)),
                            "video_sha256": sha(video_path),
                        })
                    finally:
                        room.close()
    finally:
        structural.structural_model = old_model
    result = {
        "schema": "robot-learning-lab.ape-owner-phase-exit-screen.v1",
        "source_sha256": sha(SOURCE),
        "v92_report_sha256": sha(V92 / "live-moving-owner-offer-audit.json"),
        "scope": "Evaluator-only 20ms response-feasibility screen from V92's saved owner-motion packet at16.4s. Predeclared 0/4/8cm rightward side commands over0.4s and two actual slip factors; prop and articulated owner continue. Actual geometry is used for scoring only. No actor right-side-corridor free-space proof, continuous monitor dispatch, motor stop response or task resumption is implemented.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["command_right_m"],
              row["evaluator_actual_slip"],
              row["minimum_scene_gap_m"], row["first_below_5cm_s"])
    print(path)


if __name__ == "__main__":
    main()
