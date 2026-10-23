"""Screen the selected Cove-50 CAD views against interaction targets.

This uses the selected visual MJCF and its two declared cameras. Segmentation
counts only evaluator marker pixels; it is not object recognition, a physical
optical calibration, or a policy rollout. The CAD camera package is audited
separately because a virtual camera can exist without a buildable aperture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
WIDTH, HEIGHT = 640, 480
TARGETS = {
    "shown_at_head_height_35cm": ((0.35, 0, 0.457), 0.03),
    "adult_face_1m": ((1.0, 0, 1.5), 0.095),
    "floor_8cm_40cm": ((0.4, 0, 0.04), 0.04),
    "floor_8cm_25cm": ((0.25, 0, 0.04), 0.04),
    "floor_8cm_12cm": ((0.12, 0, 0.04), 0.04),
    "floor_3cm_80cm": ((0.8, 0, 0.015), 0.015),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output: Path) -> dict:
    design = json.loads(DESIGN.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError(f"Unexpected current design: {design['id']}")
    visual = ROOT / design["evidence"]["visual_body"]
    generator = ROOT / design["generator"]["path"]
    if sha256(visual) != design["evidence"]["visual_body_sha256"]:
        raise ValueError("Selected visual body hash changed")
    if sha256(generator) != design["generator"]["sha256"]:
        raise ValueError("Selected CAD source hash changed")

    xml = ET.parse(visual)
    world = xml.getroot().find("worldbody")
    if world is None:
        raise ValueError("Missing worldbody")
    for name, (_, radius) in TARGETS.items():
        body = ET.SubElement(world, "body", name=f"marker_{name}", pos="10 0 0")
        ET.SubElement(body, "geom", name=f"marker_{name}", type="sphere",
                      size=str(radius), rgba="0.95 0.2 0.1 1", group="0",
                      contype="0", conaffinity="0")
    # Keep the temporary XML beside its relative mesh paths.
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=visual.parent,
                                     delete=False) as temporary:
        temporary.write(ET.tostring(xml.getroot(), encoding="unicode"))
        scene = Path(temporary.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    finally:
        scene.unlink()
    data = mujoco.MjData(model)
    yaw_joint = model.joint("head_yaw").id
    yaw_addr = int(model.jnt_qposadr[yaw_joint])
    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    renderer.enable_segmentation_rendering()
    rows = []
    try:
        for name, (position, _) in TARGETS.items():
            for other in TARGETS:
                model.body_pos[model.body(f"marker_{other}").id] = (
                    position if other == name else (10, 0, 0))
            geom_id = model.geom(f"marker_{name}").id
            for yaw_deg in (-40, 0, 40):
                data.qpos[yaw_addr] = math.radians(yaw_deg)
                mujoco.mj_forward(model, data)
                for camera in ("upper_rgb", "lower_rgb"):
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render()
                    pixels = int(np.count_nonzero(
                        (image[:, :, 0] == geom_id)
                        & (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))
                    rows.append({"target": name, "yaw_deg": yaw_deg,
                                 "camera": camera, "visible_pixels": pixels})
    finally:
        renderer.close()

    cad_source = generator.read_text()
    return {
        "design_id": design["id"],
        "selected_visual_body_sha256": sha256(visual),
        "selected_cad_generator_sha256": sha256(generator),
        "selected_cad_camera_modules_in_source": cad_source.count('"camera_module": box('),
        "selected_mjcf_camera_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)],
        "image_size": [WIDTH, HEIGHT],
        "target_definitions": TARGETS,
        "rows": rows,
        "scope": "Selected visual CAD mesh and declared MJCF cameras at neutral body pose; "
                 "ideal evaluator spheres, no recognition, physical optics, walking motion, "
                 "sensor timing, or proof of second camera package.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        if row["yaw_deg"] == 0:
            print(row)


if __name__ == "__main__":
    main()
