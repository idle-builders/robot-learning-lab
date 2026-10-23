"""Independent full-height, post-stop and prop-path audit for layout-4 crossing."""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import audit_robocasa_dynamic_suite_v440 as common_audit
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-crossing-v447"


def layout4_evaluator_scene(obstacle_xy):
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    prop = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                         pos=f"{obstacle_xy[0]} {obstacle_xy[1]} .25")
    ET.SubElement(prop, "geom", name="route_obstacle_geom", type="cylinder",
                  size=".24 .25", rgba="0.86 0.69 0.15 1")
    ape = ET.SubElement(task.worldbody, "body", name="audit_ape",
                         pos="1.05 -4.2 .25")
    ET.SubElement(ape, "geom", name="audit_ape_geom", type="cylinder",
                  size=".17 .25", rgba="0 0 0 0", contype="0", conaffinity="0")
    model = task.get_model()
    return model, mujoco.MjData(model), model.body("audit_ape").id, model.geom("audit_ape_geom").id


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    cases = [common_audit.audit_case(case) for case in report["cases"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_crossing_v447.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_crossing_v447.json")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                    for case in cases),
              "all_prop_paths_nonoverlap": all(case["prop_fixture_nonoverlap_pass"]
                                               for case in cases),
              "cases": cases}
    (OUT / "audit-v448.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["scenario_id"], case["outcome"],
              "gap", round(case["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "phase", case["nearest_ape_any"]["phase"],
              "prop_room", round(case["prop_fixture_nearest"]["gap_m_capped_at_30cm"], 3), flush=True)
    print("all_video_pass", output["all_video_pass"],
          "all_prop_paths_nonoverlap", output["all_prop_paths_nonoverlap"], flush=True)


if __name__ == "__main__":
    main()
