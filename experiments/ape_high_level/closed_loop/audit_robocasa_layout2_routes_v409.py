"""Independently score saved layout-2 routes with a 3D ape cylinder.

The audit rebuilds the seeded RoboCasa room, adds an evaluator-only vertical
50 cm by 34 cm cylinder, and checks signed MuJoCo geom distances at saved
positions and quarter-step interpolations. Floor contact is excluded. The
actor scripts never see this cylinder or these distances.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask


ROOT = Path(__file__).resolve().parents[3]
ROUTE = ROOT / "artifacts/ape-robocasa-layout2-route-v407/report.json"
LOCALIZED = ROOT / "artifacts/ape-robocasa-layout2-localized-route-v408/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout2-localized-route-v408/audit-v409.json"
MARGIN_M = .05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluator_scene(obstacle_xy: list[float] | None) -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1, rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    if obstacle_xy is not None:
        obstacle = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                                 pos=f"{obstacle_xy[0]} {obstacle_xy[1]} .25")
        ET.SubElement(obstacle, "geom", name="route_obstacle_geom", type="cylinder",
                      size=".24 .25", rgba="0.86 0.69 0.15 1")
    ape = ET.SubElement(task.worldbody, "body", name="audit_ape", pos="1.05 -4.2 .25")
    ET.SubElement(ape, "geom", name="audit_ape_geom", type="cylinder",
                  size=".17 .25", rgba="0 0 0 0", contype="0", conaffinity="0")
    model = task.get_model()
    return model, mujoco.MjData(model), model.body("audit_ape").id, model.geom("audit_ape_geom").id


def clearance(model: mujoco.MjModel, data: mujoco.MjData,
              body_id: int, geom_id: int, xy: np.ndarray) -> tuple[float, str | None]:
    model.body_pos[body_id, :2] = xy
    mujoco.mj_forward(model, data)
    # mj_geomDistance returns the platform's largest float when the pair is
    # farther apart than distmax. Treat that as a censored >=10 cm reading.
    best = .1
    label = None
    for other in range(model.ngeom):
        if other == geom_id or model.geom(other).name.startswith("floor_"):
            continue
        if np.linalg.norm(data.geom_xpos[other, :2] - xy) > (
                model.geom_rbound[other] + model.geom_rbound[geom_id] + .1):
            continue
        distance = float(mujoco.mj_geomDistance(model, data, geom_id, other, .1, None))
        if distance < best:
            best, label = distance, model.geom(other).name
    return best, label


def audit_episode(episode: dict, kind: str) -> dict:
    model, data, body_id, geom_id = evaluator_scene(episode.get("obstacle_xy_m"))
    trace = episode["trace"]
    poses = [np.array(row.get("actual_xy_m", row.get("pose_xy_m")), dtype=float)
             for row in trace]
    witness = []
    for i, pose in enumerate(poses):
        positions = [pose]
        if i + 1 < len(poses):
            positions += [pose + f * (poses[i + 1] - pose) for f in (.25, .5, .75)]
        for substep, location in enumerate(positions):
            distance, fixture = clearance(model, data, body_id, geom_id, location)
            witness.append({"step": i, "substep": substep,
                            "clearance_m": distance, "nearest_geom": fixture})
    closest = min(witness, key=lambda row: row["clearance_m"])
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    return {"kind": kind, "source": episode.get("source", "sensor_grid"),
            "range_m": episode["max_range_m"], "slip_m": episode.get("slip_m"),
            "obstacle_xy_m": episode.get("obstacle_xy_m"),
            "actor_outcome": episode["outcome"], "minimum_signed_clearance_m": closest["clearance_m"],
            "closest": closest, "margin_pass": closest["clearance_m"] >= MARGIN_M,
            "video_sha_match": sha256(video) == episode["video"]["sha256"],
            "video_frames_match": frames == episode["video"]["frames"] == len(trace),
            "video_frames": frames}


def main() -> None:
    route = json.loads(ROUTE.read_text())
    localized = json.loads(LOCALIZED.read_text())
    route_source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout2_route_v407.py"
    localized_source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout2_localized_route_v408.py"
    entries = []
    for kind, report in (("route", route), ("localized", localized)):
        for episode in report["episodes"]:
            entry = audit_episode(episode, kind)
            entries.append(entry)
            print(kind, entry["source"], entry["range_m"], entry["slip_m"],
                  "obstacle", entry["obstacle_xy_m"],
                  "clearance", round(entry["minimum_signed_clearance_m"], 3), flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "route_source_sha_match": route["source_sha256"] == sha256(route_source),
              "localized_source_sha_match": localized["source_sha256"] == sha256(localized_source),
              "dependency_source_sha_match": localized["base_scene_source_sha256"] == sha256(route_source),
              "body_proxy": {"height_m": .50, "diameter_m": .34,
                             "required_extra_margin_m": MARGIN_M},
              "episodes": entries}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print("all margin/video pass", all(e["margin_pass"] and e["video_sha_match"] and
                                       e["video_frames_match"] for e in entries), flush=True)


if __name__ == "__main__":
    main()
