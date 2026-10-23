"""Build and screen a source-sized two-axis-shoulder mini-ape candidate.

This checks motor boxes and replayed joint poses, not a fitted shell or a
trained policy. Run from the repository root with .venv/bin/python.
"""

from __future__ import annotations

import copy
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ape_cad"))
import build as envelope  # noqa: E402
from screen_tumble_envelopes import draw_projections  # noqa: E402


def build_candidate(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "ape_tumble_two_axis_shoulder_envelope")
    for side, sign in (("left", 1), ("right", -1)):
        carrier = root.find(f".//body[@name='{side}_shoulder_yaw_body']")
        roll = root.find(f".//body[@name='{side}_shoulder_roll_body']")
        arm = root.find(f".//body[@name='{side}_arm']")
        forearm = root.find(f".//body[@name='{side}_forearm']")
        finger = root.find(f".//body[@name='{side}_finger']")
        assert carrier is not None and roll is not None
        assert arm is not None and forearm is not None and finger is not None
        carrier.set("pos", f"0 {sign * .070:.3f} .125")
        # The yaw carrier becomes a fixed structural bridge; its motor is gone.
        carrier.remove(carrier.find(f"joint[@name='{side}_shoulder_yaw']"))
        carrier.remove(carrier.find("geom"))
        roll.set("pos", "-.035 0 0")
        arm.set("pos", f".035 {sign * .030:.3f} 0")
        forearm.set("pos", "0 0 -.075")
        finger.set("pos", f"0 {sign * .015:.3f} -.055")
        upper = arm.find("geom[@type='capsule']")
        assert upper is not None
        upper.set("fromto", "0 0 0 0 0 -.075")
        upper.set("mass", str(.012320628 * .075 / .095))
        drive = root.find(f"./actuator/position[@joint='{side}_shoulder_yaw']")
        assert drive is not None
        root.find("actuator").remove(drive)
    head = root.find(".//body[@name='head']/geom[@type='sphere']")
    assert head is not None
    head.set("size", ".043")
    head.set("mass", ".060")
    key = root.find("./keyframe/key")
    assert key is not None
    key.set("qpos", " ".join(["0", "0", ".181", "1", "0", "0", "0"] + ["0"] * 20))
    key.set("ctrl", " ".join(["0"] * 20))
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)


def mapped_qpos(old: mujoco.MjModel, new: mujoco.MjModel, values: np.ndarray) -> np.ndarray:
    result = np.array(new.key_qpos[0])
    result[:7] = values[:7]
    for jid in range(new.njnt):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        source_id = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_JOINT, name)
        result[new.jnt_qposadr[jid]] = values[old.jnt_qposadr[source_id]]
    return result


def four_point_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Kinematic contact candidate only; no balance or contact-force solution."""
    frames = []
    for index in range(21):
        t = index / 20
        data.qpos[:] = model.key_qpos[0]
        data.qpos[2] = .181 + (.143 - .181) * t
        x, y, z, w = Rotation.from_euler("y", 80 * t, degrees=True).as_quat()
        data.qpos[3:7] = [w, x, y, z]
        for side in ("left", "right"):
            for name, angle in (("hip_pitch", -np.deg2rad(80)),
                                ("knee", 1.4), ("ankle_pitch", -1.4),
                                ("shoulder_pitch", -np.deg2rad(80))):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_{name}")
                data.qpos[model.jnt_qposadr[jid]] = angle * t
        mujoco.mj_forward(model, data)
        frames.append({"index": index, "motor_box_pairs_under_1mm":
                       envelope.clashes(envelope.poses(model, data) +
                                        [envelope.battery_pose(model, data)], 1)})
    sites = {}
    for name in ("left_sole_center", "right_sole_center",
                 "left_hand_center", "right_hand_center"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        sites[name] = (data.site_xpos[sid] * 1000).round(2).tolist()
    return {"status": "illustrative_kinematic_pose_only",
            "root_pitch_deg": 80, "root_height_mm": 143,
            "bilateral_joint_rad": {"hip_pitch": round(-float(np.deg2rad(80)), 4),
                                    "knee": 1.4, "ankle_pitch": -1.4,
                                    "shoulder_pitch": round(-float(np.deg2rad(80)), 4)},
            "frames": frames, "endpoint_sites_mm": sites,
            "note": "Hand and sole site centers are near floor at the endpoint; the interpolated path does not constrain contacts, balance, shell clearance or torque."}


def main() -> None:
    out = ROOT / "artifacts/ape-tumble-two-axis-20261001"
    out.mkdir(parents=True, exist_ok=True)
    source = HERE.parent / "ape_cad/ape_spaced_v7_candidate.xml"
    candidate = out / "two-axis-shoulder.xml"
    build_candidate(source, candidate)
    layout = copy.deepcopy(envelope.CFG)
    layout["model"] = str(candidate.relative_to(ROOT))
    layout["tails"].update({"shoulder_roll_left": [0, 0, -1],
                            "shoulder_roll_right": [0, 0, -1],
                            "shoulder_pitch": [1, 0, 0]})
    layout_path = out / "layout.yaml"
    layout_path.write_text(yaml.safe_dump(layout, sort_keys=False))
    envelope.CFG = layout
    old_model = mujoco.MjModel.from_xml_path(str(source))
    model = mujoco.MjModel.from_xml_path(str(candidate))
    data = mujoco.MjData(model)
    counts = Counter()
    poses = []
    neutral_parts = None
    for label, old_qpos in envelope.recorded_poses(old_model):
        data.qpos[:] = mapped_qpos(old_model, model, old_qpos)
        mujoco.mj_forward(model, data)
        parts = envelope.poses(model, data) + [envelope.battery_pose(model, data)]
        hits = envelope.clashes(parts, layout["clearance_mm"])
        group = label.split(":")[0]
        counts[group] += bool(hits)
        poses.append({"pose": label, "pairs_under_1mm": hits})
        if label == "neutral":
            neutral_parts = parts
    assert neutral_parts is not None
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    total_mass = float(sum(model.body_mass))
    model_com = sum((model.body_mass[i] * data.xipos[i]
                     for i in range(1, model.nbody)), np.zeros(3)) / total_mass
    case_shift = sum((part["mass"] * (part["center"] - part["shaft"])
                      for part in neutral_parts if part["family"] != "battery"),
                     np.zeros(3)) / total_mass
    battery = next(part for part in neutral_parts if part["family"] == "battery")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    head_center = data.xpos[head_id]
    head_case_gap = min(
        np.linalg.norm(np.maximum(np.abs(part["rot"].T @ (head_center - part["center"]))
                                  - part["half"], 0)) - .043
        for part in neutral_parts if part["family"] == "arms"
    )
    projection = out / "motor-projections.svg"
    bounds = draw_projections(neutral_parts, projection)
    report = {
        "status": "motor_boxes_clear_only" if not sum(counts.values()) else "motor_boxes_collide",
        "model": str(candidate.relative_to(ROOT)),
        "source_model": str(source.relative_to(ROOT)),
        "source_sha256": envelope.sha256(source),
        "candidate_sha256": envelope.sha256(candidate),
        "layout_sha256": envelope.sha256(layout_path),
        "script_sha256": envelope.sha256(Path(__file__)),
        "trace_sha256": {
            stem: envelope.sha256(path)
            for stem in ("forward", "turn-left", "squat", "step-left", "acquire-place")
            if (path := ROOT / f"artifacts/ape-physics-pair-20260930/{stem}-candidate.npz").exists()
        },
        "axes": model.nu,
        "mass_kg": total_mass,
        "battery_example_center_mm": (battery["center"] * 1000).round(2).tolist(),
        "model_com_mm": (model_com * 1000).round(2).tolist(),
        "estimated_fixed_case_com_mm": ((model_com + case_shift) * 1000).round(2).tolist(),
        "head_proxy_diameter_mm": 86,
        "nearest_arm_case_to_head_proxy_gap_mm": round(head_case_gap * 1000, 2),
        "motor_box_bounds_mm": bounds,
        "sampled_poses": len(poses),
        "pose_groups_with_clash": dict(counts),
        "isolated_case_clearance_rad": envelope.single_joint_ranges(model, data),
        "poses": poses,
        "illustrative_crawl": four_point_pose(model, data),
        "limits": "This changes morphology and removes two axes. Old qpos traces were name-mapped, not run by a learned controller. Only motor/battery boxes and a head sphere are checked; structure, shell, bearings, cable paths, realistic mass distribution and dynamic reach remain unproved. Head proxy size and mass are design assumptions.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "axes", "mass_kg", "motor_box_bounds_mm", "sampled_poses", "pose_groups_with_clash", "nearest_arm_case_to_head_proxy_gap_mm")}, indent=2))


if __name__ == "__main__":
    main()
