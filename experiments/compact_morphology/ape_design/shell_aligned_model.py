"""Retarget the existing ape MJCF joint tree to sketch-shell v2 proportions.

This only gives the shell a moving skeleton for packaging screens. The old
actuator/control parameters and provisional masses are deliberately retained;
these files are not promoted to trained or calibrated simulation models.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/ape_cad/ape_spaced_v7_candidate.xml"
OUT = ROOT / "artifacts/ape-sketch-shell-kinematics-20261002"


def body(root: ET.Element, name: str) -> ET.Element:
    result = root.find(f".//body[@name='{name}']")
    if result is None:
        raise ValueError(name)
    return result


def make_model(axes: int) -> Path:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", f"ape_sketch_shell_v2_{axes}_axis_kinematics")
    for side, sign in (("left", 1), ("right", -1)):
        body(root, f"{side}_hip_yaw").set("pos", f"0 {sign * .040:.3f} .025")
        body(root, f"{side}_hip_roll").set("pos", "0 0 -.032")
        body(root, f"{side}_thigh").set("pos", "0 0 -.008")
        body(root, f"{side}_shin").set("pos", "0 0 -.061")
        body(root, f"{side}_foot").set("pos", "0 0 -.073")
        body(root, f"{side}_shoulder_yaw_body").set(
            "pos", f"-.004 {sign * .060:.3f} .077")
        body(root, f"{side}_shoulder_roll_body").set(
            "pos", f"0 {sign * .009:.3f} -.011")
        body(root, f"{side}_arm").set("pos", f".001 {sign * .018:.3f} -.003")
        body(root, f"{side}_forearm").set("pos", f"0 {sign * .009:.3f} -.045")
        body(root, f"{side}_finger").set("pos", f".010 {sign * .012:.3f} -.076")
        thigh = body(root, f"{side}_thigh").find("geom[@type='capsule']")
        shin = body(root, f"{side}_shin").find("geom[@type='capsule']")
        arm = body(root, f"{side}_arm").find("geom[@type='capsule']")
        forearm = body(root, f"{side}_forearm").find("geom[@type='capsule']")
        assert thigh is not None and shin is not None and arm is not None and forearm is not None
        thigh.set("fromto", "0 0 0 0 0 -.061")
        shin.set("fromto", "0 0 0 0 0 -.073")
        arm.set("fromto", "0 0 0 0 0 -.045")
        forearm.set("fromto", "0 0 0 0 0 -.076")
        foot = body(root, f"{side}_foot")
        foot.find(f"geom[@name='{side}_sole']").set("size", ".039 .032 .012")
        foot.find(f"site[@name='{side}_sole_touch']").set("size", ".039 .032 .012")
        if axes == 20:
            carrier = body(root, f"{side}_shoulder_yaw_body")
            carrier.remove(carrier.find(f"joint[@name='{side}_shoulder_yaw']"))
            carrier.remove(carrier.find("geom"))
            drive = root.find(f"./actuator/position[@joint='{side}_shoulder_yaw']")
            assert drive is not None
            root.find("actuator").remove(drive)
    if axes == 21:
        head_body = body(root, "head")
        head_body.remove(head_body.find("joint[@name='head_pitch']"))
        head_body.remove(head_body.find("geom[@name='candidate_head_pitch_motor_mass']"))
        drive = root.find("./actuator/position[@joint='head_pitch']")
        assert drive is not None
        root.find("actuator").remove(drive)
    head = body(root, "head").find("geom[@type='sphere']")
    assert head is not None
    head.set("size", ".045")
    key = root.find("./keyframe/key")
    assert key is not None
    key.set("qpos", " ".join(["0", "0", ".181", "1", "0", "0", "0"] + ["0"] * axes))
    key.set("ctrl", " ".join(["0"] * axes))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"shell-aligned-{axes}.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


def main() -> None:
    summary = {}
    for axes in (21, 22, 20):
        path = make_model(axes)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        mujoco.mj_forward(model, data)
        anchors = {}
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name != "root":
                anchors[name] = [round(float(x) * 1000, 2) for x in data.xanchor[jid]]
        summary[str(axes)] = {
            "model": str(path.relative_to(ROOT)), "axes": model.nu,
            "old_mass_kg_not_CAD_derived": round(float(sum(model.body_mass)), 3),
            "neutral_joint_anchors_mm": anchors,
        }
    (OUT / "anchors.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Retargeted 21-, 22- and 20-axis joint trees -> {OUT}")


if __name__ == "__main__":
    main()
