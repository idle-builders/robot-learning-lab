"""Independent 3D footprint and video audit for held-out layout 4."""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

from audit_robocasa_layout2_routes_v409 import clearance, sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-transfer-v412"


def evaluator_scene() -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ape = ET.SubElement(task.worldbody, "body", name="audit_ape", pos="1.05 -4.2 .25")
    ET.SubElement(ape, "geom", name="audit_ape_geom", type="cylinder",
                  size=".17 .25", rgba="0 0 0 0", contype="0", conaffinity="0")
    model = task.get_model()
    return model, mujoco.MjData(model), model.body("audit_ape").id, model.geom("audit_ape_geom").id


def audit_episode(episode: dict) -> dict:
    model, data, body_id, geom_id = evaluator_scene()
    poses = [np.array(row["actual_xy_m"], dtype=float) for row in episode["trace"]]
    best = {"distance_m_capped_at_10cm": .1, "geom": None, "step": None}
    for i, pose in enumerate(poses):
        samples = [pose]
        if i + 1 < len(poses):
            samples += [pose + f * (poses[i + 1] - pose) for f in (.25, .5, .75)]
        for sample in samples:
            distance, geom = clearance(model, data, body_id, geom_id, sample)
            if distance < best["distance_m_capped_at_10cm"]:
                best = {"distance_m_capped_at_10cm": distance, "geom": geom, "step": i}
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    return {"source": episode["source"], "max_range_m": episode["max_range_m"],
            "outcome": episode["outcome"], "steps": episode["steps"],
            "final_pose_error_m": episode["end_pose_error_m"],
            "clearance": best,
            "proxy_5cm_margin_pass": best["distance_m_capped_at_10cm"] >= .05,
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frames_pass": frames == episode["video"]["frames"] == len(episode["trace"])}


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout4_transfer_v412.py"
    actor = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout2_localized_route_v408.py"
    cases = [audit_episode(episode) for episode in report["episodes"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "scene_source_sha_pass": report["source_sha256"] == sha256(source),
              "frozen_actor_source_sha_pass": report["frozen_actor_source_sha256"] == sha256(actor),
              "scene_pair_fingerprint_pass": len({x["scene_fingerprint"] for x in report["episodes"]}) == 1,
              "cases": cases,
              "all_proxy_video_pass": all(x["proxy_5cm_margin_pass"] and
                                          x["video_sha_pass"] and
                                          x["video_frames_pass"] for x in cases)}
    (OUT / "audit-v413.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["source"], case["max_range_m"], case["outcome"],
              "pose error", round(case["final_pose_error_m"], 3),
              "clearance", case["clearance"]["distance_m_capped_at_10cm"], flush=True)
    print("all_proxy_video_pass", output["all_proxy_video_pass"], flush=True)


if __name__ == "__main__":
    main()
