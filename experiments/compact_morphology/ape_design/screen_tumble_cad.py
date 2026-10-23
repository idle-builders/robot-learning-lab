"""Small exact-solid clearance screen for the Tumble CAD candidate.

This is an engineering readout, not a test suite or swept-volume certificate.
It checks neutral plus a few representative poses against sourced case boxes.
"""

from __future__ import annotations

import argparse
import json

import build123d as bd
import mujoco
import numpy as np
import yaml

import tumble_cad as cad


def separated(a: bd.BoundBox, b: bd.BoundBox) -> bool:
    return any(getattr(a.max, axis) <= getattr(b.min, axis) or
               getattr(b.max, axis) <= getattr(a.min, axis)
               for axis in "XYZ")


def screen_pose(model: mujoco.MjModel, data: mujoco.MjData,
                label: str) -> dict:
    mujoco.mj_forward(model, data)
    cases = cad.envelope.poses(model, data) + [cad.envelope.battery_pose(model, data)]
    case_pairs = cad.envelope.clashes(cases, 1)
    visible = []
    for part in cad.PARTS:
        shape = cad.place(part.shape, cad.body_tf(model, data, part.body))
        visible.append((part, shape, shape.bounding_box(optimal=False)))
    intersections = []
    for component in cases:
        transform = np.eye(4)
        transform[:3, :3] = component["rot"]
        transform[:3, 3] = component["center"]
        case = cad.place(bd.Box(*(component["half"] * 2000)), transform)
        bounds = case.bounding_box(optimal=False)
        for part, shape, part_bounds in visible:
            if separated(bounds, part_bounds):
                continue
            solid = shape.intersect(case)
            volume = sum(s.volume for s in solid) if solid is not None else 0
            if volume > 5:
                intersections.append({"case": component["name"],
                                      "part": part.name,
                                      "volume_mm3": round(float(volume), 1),
                                      "own_link": component["owner"] == part.body})
    shell_intersections = []
    for i, (left, left_shape, left_bounds) in enumerate(visible):
        for right, right_shape, right_bounds in visible[i + 1:]:
            if left.body == right.body or separated(left_bounds, right_bounds):
                continue
            solid = left_shape.intersect(right_shape)
            volume = sum(s.volume for s in solid) if solid is not None else 0
            if volume > 5:
                shell_intersections.append({"left": left.name, "right": right.name,
                                            "volume_mm3": round(float(volume), 1)})
    return {"pose": label, "case_pairs_under_1mm": case_pairs,
            "case_part_intersections": intersections,
            "between_body_part_intersections": shell_intersections}


def home_trial() -> None:
    """Screen the rejected parked-arm idea from the exported immutable parts."""
    cad.envelope.CFG = yaml.safe_load(
        (cad.ROOT / "artifacts/ape-tumble-two-axis-20261001/layout.yaml").read_text())
    report = json.loads((cad.OUT / "report.json").read_text())
    cad.PARTS[:] = [cad.Part(part["name"], part["body"], part["kind"],
                            bd.import_step(cad.OUT / part["step"]), part["role"])
                    for part in report["parts"]]
    model = mujoco.MjModel.from_xml_path(str(cad.MODEL))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    for side in ("left", "right"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{side}_shoulder_pitch")
        data.qpos[model.jnt_qposadr[jid]] = .3
    result = screen_pose(model, data, "parked_arms_pitch_0.3_trial")
    (cad.OUT / "clearance-home-trial.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print("parked pose:", len(result["case_pairs_under_1mm"]), "case pairs;",
          len(result["case_part_intersections"]), "case/part intersections;",
          len(result["between_body_part_intersections"]), "moving-part overlaps")


def main() -> None:
    cad.envelope.CFG = yaml.safe_load(
        (cad.ROOT / "artifacts/ape-tumble-two-axis-20261001/layout.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(cad.MODEL))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    cad.PARTS.clear()
    cad.torso()
    cad.head()
    cad.limbs()
    cad.fit_own_motor_pods(model, data)
    reliefs = cad.apply_pose_case_reliefs(model, data)
    poses = [screen_pose(model, data, "neutral")]

    old = mujoco.MjModel.from_xml_path(str(cad.SOURCE))
    selected = {"forward:199", "squat:499", "acquire-place:874",
                "forward:232", "squat:332", "acquire-place:999"}
    for label, q in cad.envelope.recorded_poses(old):
        if label not in selected:
            continue
        data.qpos[:] = cad.mapped_qpos(old, model, q)
        poses.append(screen_pose(model, data, label))

    data.qpos[:] = model.key_qpos[0]
    data.qpos[2] = .143
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler("y", 80, degrees=True).as_quat()
    data.qpos[3:7] = [w, x, y, z]
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -np.deg2rad(80)),
                             ("knee", 1.4), ("ankle_pitch", -1.4),
                             ("shoulder_pitch", -.9234768)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{side}_{joint}")
            data.qpos[model.jnt_qposadr[jid]] = value
    poses.append(screen_pose(model, data, "illustrative_crawl_endpoint"))
    report = {"status": "screened_selected_poses_only",
              "sampled_case_reliefs": reliefs, "poses": poses,
              "limits": "Exact CAD-vs-box intersections at eight static poses only. No full joint sweep, joint torque, shaft/horn, wire, fastener, print-strength or learned-policy validation."}
    path = cad.OUT / "clearance.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for pose in poses:
        hits = pose["case_part_intersections"]
        print(pose["pose"], len(pose["case_pairs_under_1mm"]),
              len(hits), len(pose["between_body_part_intersections"]),
              sorted({h["case"] for h in hits}))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", action="store_true",
                        help="Screen only the experimental parked-arm pose")
    if parser.parse_args().home:
        home_trial()
    else:
        main()
