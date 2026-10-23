"""V324: actor-side shin motion alarm and a hypothetical protective arm skill.

This continues a V323 blocked pose without feeding scene labels to the alarm.
The arm is repositioned kinematically, so the result is a skill requirement,
not proof that a motor controller can execute the response.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_moving_low_clutter_v302 import STOOL_SIZE


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v324"
PRIOR = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
START_S = 15.6
ALARM_S = 15.8
END_S = 16.4
ELBOW_FLEX_RAD = -.8
FLEX_TIME_S = .4


def actual_pose(trace: dict) -> np.ndarray:
    return np.asarray(next(event["evaluator_actual_pose_xyyaw"]
                           for event in reversed(trace["controller_events"])
                           if "evaluator_actual_pose_xyyaw" in event), dtype=float)


def estimated_pose(trace: dict) -> np.ndarray:
    source = ROOT / trace["recovery_inputs"][0]["head_inputs"][-1]["actor_frame_path"]
    with np.load(source) as frame:
        return frame["estimated_pose_xyyaw"].copy()


def scene(room: LateCorridorRoom, time_s: float, *, moving: bool,
          mirrored: bool, base_y: float) -> None:
    room.time_s = time_s
    room._update_moving_obstacle()
    y = float(room.obstacle_xy[3][1]) if moving else .45
    if mirrored:
        y = 2 * base_y - y
    room.model.body_pos[room.model.body("loop_stool").id] = (
        0., y, STOOL_SIZE[2])
    room._forward()


def side_returns(points: np.ndarray, pose: np.ndarray, side: int) -> np.ndarray:
    """A low side strip relative to sensed robot pose; no prop class is used."""
    delta = points[:, :2] - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    x = c * delta[:, 0] + s * delta[:, 1]
    y = side * (-s * delta[:, 0] + c * delta[:, 1])
    return y[(np.abs(x) < .12) & (y > .15) & (y < .50) &
             (points[:, 2] > .015) & (points[:, 2] < .10)]


def actor_alarm(first: np.ndarray, second: np.ndarray,
                pose: np.ndarray) -> dict:
    sides = {}
    for side, label in ((1, "left"), (-1, "right")):
        a, b = side_returns(first, pose, side), side_returns(second, pose, side)
        closing = (float(np.median(a) - np.median(b))
                   if len(a) >= 2 and len(b) >= 2 else None)
        sides[label] = {"first_returns": len(a), "second_returns": len(b),
                        "median_first_m": round(float(np.median(a)), 5) if len(a) else None,
                        "median_second_m": round(float(np.median(b)), 5) if len(b) else None,
                        "closing_0p2s_m": round(closing, 5) if closing is not None else None,
                        "candidate": closing is not None and closing >= .03}
    candidates = [label for label, item in sides.items() if item["candidate"]]
    selected = (max(candidates, key=lambda label: sides[label]["closing_0p2s_m"])
                if candidates else None)
    return {"status": "approaching_side" if selected else "no_approach",
            "side": selected, "side_returns": sides,
            "threshold_m_per_0p2s": .03}


def make_room(seed: int, trace: dict, branch: Path) -> LateCorridorRoom:
    room = LateCorridorRoom(case=tuple(trace["case"]),
                            pose_source="command", branch=branch)
    room.layout(seed, absent=False, furnished=True)
    room.actual[:] = actual_pose(trace)
    room.estimated[:] = estimated_pose(trace)
    return room


def estimated_shin_points(points: np.ndarray, actual: np.ndarray,
                          estimated: np.ndarray) -> np.ndarray:
    c, s = np.cos(actual[2]), np.sin(actual[2])
    body = (points[:, :2] - actual[:2]) @ np.array(((c, -s), (s, c)))
    ce, se = np.cos(estimated[2]), np.sin(estimated[2])
    output = points.copy()
    output[:, :2] = (body @ np.array(((ce, se), (-se, ce))) +
                     estimated[:2])
    return output


def panel(pixels: np.ndarray, label: str) -> np.ndarray:
    # The external overview makes the small ape difficult to inspect. Keep
    # the same fixed crop for both policies so apparent motion is comparable.
    image = Image.fromarray(pixels).crop((60, 55, 300, 235)).resize((640, 480))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        trace_path = (PRIOR / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        assert trace["recovery_inputs"][0]["head_inputs"][-1]["capture_time_s"] == 16.
        for mirrored in (False, True):
            for moving in (False, True):
                label = (f"seed{seed}-" + ("right" if mirrored else "left") +
                         ("-moving" if moving else "-parked"))
                branch = OUT / label
                branch.mkdir(parents=True, exist_ok=True)
                room = make_room(seed, trace, branch)
                try:
                    robot = RobotGeometry(room.model)
                    actor_frames = []
                    for index, time_s in enumerate((START_S, ALARM_S)):
                        scene(room, time_s, moving=moving,
                              mirrored=mirrored, base_y=room.actual[1])
                        cloud, counts = shin_scan(
                            room, robot, {"case": f"v324-{label}-{index}"})
                        # Sensor generation knows actual geometry; the actor
                        # receives rays transformed by its retained V323 pose.
                        points = estimated_shin_points(
                            cloud["four"], room.actual, room.estimated)
                        source = branch / f"shin-{index}.npz"
                        np.savez_compressed(source, points_xyz_m=points,
                                            estimated_pose_xyyaw=room.estimated,
                                            actor_qpos=room.data.qpos,
                                            capture_time_s=time_s)
                        actor_frames.append({"path": str(source.relative_to(ROOT)),
                                             "sha256": sha(source), "time_s": time_s,
                                             "point_count": len(points),
                                             "scan_counts": counts})
                    with np.load(ROOT / actor_frames[0]["path"]) as frame:
                        first = frame["points_xyz_m"].copy()
                    with np.load(ROOT / actor_frames[1]["path"]) as frame:
                        second = frame["points_xyz_m"].copy()
                    alarm = actor_alarm(first, second, room.estimated)
                    # A paired hold is the evaluator baseline. The actor asks
                    # for side-specific elbow flexion only when rays alarm.
                    outcomes = {}
                    frames = {}
                    for policy in ("hold", "alarm_elbow_flex"):
                        closest = (float("inf"), None, "", "")
                        rendered = []
                        joint = (f"{alarm['side']}_elbow_pitch"
                                 if alarm["side"] and policy == "alarm_elbow_flex" else None)
                        address = (int(room.model.joint(joint).qposadr[0])
                                   if joint else None)
                        for index, time_s in enumerate(np.linspace(START_S, END_S, 41)):
                            scene(room, float(time_s), moving=moving,
                                  mirrored=mirrored, base_y=room.actual[1])
                            room.data.qpos[18:27] = 0.
                            if address is not None:
                                fraction = np.clip((time_s - ALARM_S) / FLEX_TIME_S, 0., 1.)
                                room.data.qpos[address] = ELBOW_FLEX_RAD * fraction
                            room._forward()
                            gap, body, obstacle = room._current_obstacle_clearance()
                            if gap < closest[0]:
                                closest = (gap, float(time_s), body, obstacle)
                            if moving and seed == SEEDS[0] and index % 5 == 0:
                                rendered.append(panel(np.asarray(room.capture_external()),
                                    f"{label} | {policy} | t={time_s:.2f}s | gap={gap:.3f}m"))
                        outcomes[policy] = {
                            "minimum_evaluator_clearance_m": round(closest[0], 5),
                            "time_of_minimum_s": round(closest[1], 4),
                            "closest_robot_geom": closest[2],
                            "closest_obstacle_geom": closest[3],
                            "requested_joint": joint,
                            "requested_joint_target_rad": ELBOW_FLEX_RAD if joint else None,
                            "requested_duration_s": FLEX_TIME_S if joint else None}
                        frames[policy] = rendered
                    video_path = None
                    if moving and seed == SEEDS[0]:
                        video_path = branch / "hold-vs-alarm-elbow-flex.mp4"
                        with imageio.get_writer(video_path, fps=6,
                                                codec="libx264", quality=7) as writer:
                            for left, right in zip(frames["hold"], frames["alarm_elbow_flex"]):
                                writer.append_data(np.concatenate((left, right), axis=1))
                    row = {"seed": seed, "case": label,
                           "moving_evaluator_setup": moving,
                           "mirrored_evaluator_setup": mirrored,
                           "actor_shin_frames": actor_frames,
                           "actor_alarm": alarm,
                           "actor_estimated_pose_xyyaw": room.estimated.tolist(),
                           "evaluator_outcomes": outcomes,
                           "video_path": (str(video_path.relative_to(ROOT))
                                          if video_path else None),
                           "video_sha256": sha(video_path) if video_path else None,
                           "v323_trace_sha256": sha(trace_path)}
                    rows.append(row)
                    print(label, alarm["status"], alarm["side"],
                          outcomes["hold"]["minimum_evaluator_clearance_m"],
                          outcomes["alarm_elbow_flex"]["minimum_evaluator_clearance_m"],
                          flush=True)
                finally:
                    room.close()
    report = {
        "scope": "V324 selected Direct Frame continuation from the V323 blocked pose. Two independent noisy four-shin scans 0.2s apart drive a low side-strip closing alarm. The controller requests corresponding elbow flexion; evaluator-only geometry compares it with holding through t16.4 at 20ms samples. The arm moves by qpos interpolation rather than native actuators or a trained skill. Mirror and parked variants test side selection and false alarms. Dynamic human avoidance, self-collision under motion and skill feasibility remain unqualified.",
        "source_sha256": sha(SOURCE),
        "shin_scanner_sha256": sha(Path(shin_scan.__code__.co_filename)),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows}
    (OUT / "shin-intrusion-posture-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
