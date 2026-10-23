"""V273: replay a source-policy stop-distance envelope in Direct Frame rooms.

The replay is a straight-line kinematic stress test, not a native stop rollout
or a transferable bound for the selected Direct Frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v273"
V272 = ROOT / "artifacts/ape-closed-loop-cove50-v272/direct-structural-live-audit.json"
STOP = ROOT / "artifacts/ape-skill-contract-cove50-20261009/stop-trace.json"
SAMPLES = 31


def pane(room: DirectStructuralRoom, label: str) -> np.ndarray:
    rgb, _ = room.capture("lower_rgb", depth=False)
    external = room.capture_external()
    image = Image.fromarray(np.concatenate((np.asarray(rgb),
                                             np.asarray(external)), axis=1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 29), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return np.asarray(image)


def audit(seed: int, kind: str, distance: float) -> dict:
    source = ROOT / f"artifacts/ape-closed-loop-cove50-v272/seed{seed}"
    nav = json.loads((source / "trace.json").read_text())
    frames = {sha(path): path for path in source.glob("frame-*.npz")}
    room = DirectStructuralRoom(kind=kind, mode="present",
                                side=1 if seed % 2 else -1)
    room.layout(seed, absent=False, furnished=True)
    obstacles = scene_geoms(room)
    robots = body_geoms(room, obstacles)
    rows = []
    try:
        for event in nav["events"]:
            if event["action"] != "guarded_approach_step":
                continue
            direction = np.asarray(event["detail"]["command_delta_xy_m"], dtype=float)
            magnitude = float(np.linalg.norm(direction))
            if magnitude < 1e-6:
                continue
            direction /= magnitude
            path = frames.get(event["actor_input_sha256"])
            if path is None:
                raise FileNotFoundError("Saved actor frame not hash-matched")
            with np.load(path) as saved:
                qpos = saved["actor_qpos"].copy()
            room.data.qpos[:] = qpos
            room.head = float(qpos[room.head_addr])
            start = np.asarray(event["evaluator_actual_pose_xyyaw"], dtype=float)
            clearances = []
            for extra in np.linspace(0., distance, SAMPLES):
                room.actual[:] = start
                room.actual[:2] += extra * direction
                room._forward()
                clearance = full_body_clearance(room, obstacles, robots)
                clearances.append((float(extra), float(clearance[0]),
                                   clearance[1], clearance[2]))
            minimum = min(clearances, key=lambda row: row[1])
            first_below_5cm = next((r[0] for r in clearances if r[1] < .05), None)
            first_contact = next((r[0] for r in clearances if r[1] < 0.), None)
            rows.append({"time_s": event["time_s"],
                         "action_index": event["detail"]["move_index"],
                         "recorded_clearance_m":
                             event["evaluator_model_clearance_m"],
                         "replayed_start_clearance_m": round(clearances[0][1], 5),
                         "minimum_straight_overshoot_clearance_m":
                             round(minimum[1], 5),
                         "limiting_geoms": list(minimum[2:]),
                         "first_5cm_violation_at_extra_m": first_below_5cm,
                         "first_contact_at_extra_m": first_contact,
                         "unit_direction_xy": direction.tolist()})
        last = rows[-1]
        event = next(e for e in reversed(nav["events"])
                     if e["action"] == "guarded_approach_step")
        with np.load(frames[event["actor_input_sha256"]]) as saved:
            qpos = saved["actor_qpos"].copy()
        room.data.qpos[:] = qpos
        room.head = float(qpos[room.head_addr])
        start = np.asarray(event["evaluator_actual_pose_xyyaw"], dtype=float)
        direction = np.asarray(last["unit_direction_xy"])
        video_path = OUT / f"seed{seed}-hypothetical-stop-envelope.mp4"
        with imageio.get_writer(video_path, fps=5, codec="libx264", quality=7) as video:
            for extra in np.linspace(0., distance, 16):
                room.actual[:] = start
                room.actual[:2] += extra * direction
                room._forward()
                gap = full_body_clearance(room, obstacles, robots)[0]
                video.append_data(pane(room,
                    f"hypothetical straight stop travel {extra*100:.1f}cm | model gap {gap*100:.1f}cm"))
        return {"seed": seed, "kind": kind,
                "stop_distance_replayed_m": distance,
                "minimum_all_action_stops_m": min(
                    row["minimum_straight_overshoot_clearance_m"] for row in rows),
                "actions_below_5cm": sum(row["first_5cm_violation_at_extra_m"]
                                          is not None for row in rows),
                "actions_with_contact": sum(row["first_contact_at_extra_m"]
                                            is not None for row in rows),
                "last_action": last, "rows": rows,
                "video_sha256": sha(video_path)}
    finally:
        room.close()


def main() -> None:
    live = json.loads(V272.read_text())
    stop = json.loads(STOP.read_text())
    upper = float(stop["summary"]["path_to_observed_stop_m_range"][1])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit(row["seed"], row["kind"], upper)
            for row in live["rows"]]
    report = {"scope": "V273 evaluator-only diagnostic: at every V272 guarded translation endpoint, carry the Direct Frame articulated pose forward along that action's commanded XY direction by the upper 11.314cm source-body native zero-twist path length. Sample 31 straight-line poses, evaluate all Direct Frame body solids against target/furniture/partition/walls, render terminal hypothetical continuation. This is NOT the actual native stop trajectory or a transferable physical bound: source policy/body differs from selected Direct Frame, actual yaw/slip/contact response and 1.14-1.68s latency are omitted. A failure of this stress screen blocks a safety claim; passing would not prove one.",
              "source_sha256": sha(SOURCE), "v272_sha256": sha(V272),
              "stop_trace_sha256": sha(STOP),
              "selected_visual_sha256": sha(selected_visual()),
              "stop_source_body_sha256": stop["source_body_sha256"],
              "rows": rows}
    (OUT / "direct-stop-envelope-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["kind"], "minimum",
              row["minimum_all_action_stops_m"], "below5",
              row["actions_below_5cm"], "contact",
              row["actions_with_contact"], flush=True)


if __name__ == "__main__":
    main()
