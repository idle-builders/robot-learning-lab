"""Merge historical articulated 50 cm ape replay into a RoboCasa kitchen.

RoboCasa fixtures/textures are upstream resources under ignored .cache. The
ape is still the rejected Direct Frame historical body, and replaying qpos
does not execute a newly learned policy or physical collisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-native-route-guard/v8-robocasa-native"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRACE = ROOT / "artifacts/ape-cove50-effort-envelope/forward-stop-late.npz"
START_XY = np.array([2.75, -1.80])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_xml() -> str:
    arena = KitchenArena(layout_id=1, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    root = ET.fromstring(task.get_xml())
    world = root.find("worldbody")
    historical = ET.parse(BODY).getroot()
    ape = copy.deepcopy(historical.find("worldbody").find("body"))
    for element in ape.iter():
        if "name" in element.attrib:
            element.set("name", "ape_" + element.get("name"))
        if element.tag == "geom":
            color = element.get("rgba")
            if color and float(color.split()[-1]) == 0:
                element.set("rgba", " ".join(color.split()[:3]) + " .87")
    world.append(ape)
    person = ET.SubElement(world, "body", name="approaching_person", pos="10 -1 0")
    ET.SubElement(person, "geom", name="person_capsule", type="capsule",
                  fromto="0 0 .10 0 0 1.30", size=".09", group="0",
                  rgba=".75 .42 .36 1", contype="0", conaffinity="0")
    pet = ET.SubElement(world, "body", name="low_pet", pos="10 -1 0")
    ET.SubElement(pet, "geom", name="pet_box", type="box", pos="0 0 .08",
                  size=".18 .08 .08", group="0", rgba=".66 .51 .30 1",
                  contype="0", conaffinity="0")
    # Fixed, ray-visible clutter beside the route challenges static/dynamic separation.
    stool = ET.SubElement(world, "body", name="side_stool", pos="3.10 -1.25 0")
    ET.SubElement(stool, "geom", name="stool_column", type="cylinder",
                  pos="0 0 .21", size=".105 .21", group="0",
                  rgba=".37 .29 .22 1", contype="0", conaffinity="0")
    ET.SubElement(stool, "geom", name="stool_seat", type="cylinder",
                  pos="0 0 .44", size=".18 .025", group="0",
                  rgba=".45 .33 .22 1", contype="0", conaffinity="0")
    ET.SubElement(world, "camera", name="ape_route_overview",
                  pos="3.2 -4.6 2.1", xyaxes="1 0 0 0 .46 .89", fovy="50")
    return ET.tostring(root, encoding="unicode")


def compile_scene() -> tuple[mujoco.MjModel, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / "scene.xml"
    xml_path.write_text(scene_xml())
    return mujoco.MjModel.from_xml_path(str(xml_path)), xml_path


def put_native_pose(model: mujoco.MjModel, data: mujoco.MjData, saved,
                    world: int, t: float, stop_at: float) -> np.ndarray:
    # The native source's four worlds share motion but not XY starts.
    index = min(len(saved["time_s"])-1, max(0, round(t/.02)-1))
    native_stop = 149
    if t <= stop_at:
        pose = saved["root_pose"][index, world].copy()
        joints = saved["joint_pos"][index, world]
    else:
        trigger = min(native_stop, max(0, round(stop_at/.02)-1))
        tail = min(len(saved["time_s"])-1,
                   native_stop + max(0, round((t-stop_at)/.02)))
        pose = saved["root_pose"][tail, world].copy()
        pose[:3] = (saved["root_pose"][trigger, world, :3] +
                    saved["root_pose"][tail, world, :3] -
                    saved["root_pose"][native_stop, world, :3])
        joints = saved["joint_pos"][tail, world]
    pose[:2] += (START_XY - saved["root_pose"][0, world, :2])
    data.qpos[:] = model.qpos0
    root_adr = model.joint("ape_root").qposadr[0]
    data.qpos[root_adr:root_adr+7] = pose
    for name, angle in zip(saved["leg_names"], joints):
        data.qpos[model.joint("ape_"+str(name)).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    return pose


def main() -> None:
    model, xml_path = compile_scene()
    data = mujoco.MjData(model)
    with np.load(TRACE, allow_pickle=False) as saved:
        pose = put_native_pose(model, data, saved, 0, .02, 3.0)
    report = {"schema": "robot-learning-lab.ape-robocasa-native-merge.v1",
              "task": "T-9v3p3n",
              "scope": "Compiled RoboCasa layout 1/style 1 with articulated historical 50 cm body, no learned rollout or task loop yet",
              "scene_sha256": sha(xml_path), "body_sha256": sha(BODY),
              "native_trace_sha256": sha(TRACE),
              "source_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__,
              "model": {"bodies": model.nbody, "geoms": model.ngeom,
                        "joints": model.njnt, "qpos": model.nq},
              "first_pose_xy_m": pose[:2].tolist(),
              "head_cameras": ["ape_upper_rgb", "ape_lower_rgb"]}
    (OUT / "compile-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["model"]))


if __name__ == "__main__":
    main()
