"""Screen source-sized motor boxes under the aesthetic-first Tumble sketch.

This is an internal packaging study, not the exterior CAD or a learned-policy
replay. Run from the repo root with:
  .venv/bin/python experiments/compact_morphology/ape_design/screen_tumble_envelopes.py
"""

from __future__ import annotations

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ape_cad"))
import build as envelope  # noqa: E402


def illustrative_crawl(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """One geometric four-point candidate, not a trained or balanced crawl."""
    samples = []
    for index in range(21):
        t = index / 20
        data.qpos[:] = model.key_qpos[0]
        data.qpos[2] = .181 + (.168 - .181) * t
        x, y, z, w = Rotation.from_euler("y", 40 * t, degrees=True).as_quat()
        data.qpos[3:7] = [w, x, y, z]
        for side in ("left", "right"):
            for joint, angle in (("hip_pitch", np.deg2rad(-40)),
                                 ("knee", .9), ("ankle_pitch", -.9),
                                 ("shoulder_pitch", np.deg2rad(-40))):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{side}_{joint}")
                data.qpos[model.jnt_qposadr[jid]] = angle * t
        mujoco.mj_forward(model, data)
        hits = envelope.clashes(envelope.poses(model, data) +
                                 [envelope.battery_pose(model, data)], 1)
        samples.append({"index": index, "motor_box_pairs_under_1mm": hits})
    sites = {}
    for name in ("left_sole_center", "right_sole_center",
                 "left_hand_center", "right_hand_center"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        sites[name] = (data.site_xpos[sid] * 1000).round(2).tolist()
    return {
        "status": "illustrative_kinematic_pose_only",
        "root_pitch_deg": 40,
        "root_height_mm": 168,
        "bilateral_joint_rad": {"hip_pitch": round(float(np.deg2rad(-40)), 4),
                                "knee": .9, "ankle_pitch": -.9,
                                "shoulder_pitch": round(float(np.deg2rad(-40)), 4)},
        "samples": samples,
        "endpoint_sites_mm": sites,
        "note": "Hand centers sit 9.8 mm above floor while soles are about 0.3 mm below; a broad palm contact surface, shell clearance and balanced control remain unproved."
    }


def draw_projections(parts: list[dict], destination: Path) -> dict:
    """Dimensioned motor-box projections; overlapping 2D boxes may separate in 3D."""
    colors = {"legs": "#df7c58", "arms": "#547e9a",
              "small": "#e9ad55", "battery": "#7caa87"}
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                             "width": "980", "height": "600",
                             "viewBox": "0 0 980 600"})
    ET.SubElement(svg, "rect", {"width": "980", "height": "600",
                                "fill": "#f5f2eb"})
    heading = ET.SubElement(svg, "text", {"x": "40", "y": "38",
                                         "font-family": "sans-serif",
                                         "font-size": "20", "fill": "#23313a"})
    heading.text = "Tumble · source-sized motor envelopes, no exterior shell"
    for horizontal, title, origin, scale in ((1, "FRONT · side-to-side mm", 50, 1.2),
                                             (0, "SIDE · fore–aft mm", 560, 2.0)):
        min_h = -175 if horizontal == 1 else -80
        label = ET.SubElement(svg, "text", {"x": str(origin), "y": "70",
                                            "font-family": "sans-serif",
                                            "font-size": "14", "font-weight": "bold",
                                            "fill": "#23313a"})
        label.text = title
        for height in range(0, 361, 50):
            y = 530 - height * 1.25
            ET.SubElement(svg, "line", {"x1": str(origin), "x2": str(origin + 420),
                                         "y1": str(y), "y2": str(y),
                                         "stroke": "#d7d2c9", "stroke-width": "1"})
            tick = ET.SubElement(svg, "text", {"x": str(origin - 28),
                                               "y": str(y + 4),
                                               "font-family": "sans-serif",
                                               "font-size": "10", "fill": "#67737a"})
            tick.text = str(height)
        for part in parts:
            ext = np.abs(part["rot"]) @ part["half"]
            center = part["center"]
            low = (center - ext) * 1000
            high = (center + ext) * 1000
            ET.SubElement(svg, "rect", {
                "x": str(origin + (low[horizontal] - min_h) * scale),
                "y": str(530 - high[2] * 1.25),
                "width": str((high[horizontal] - low[horizontal]) * scale),
                "height": str((high[2] - low[2]) * 1.25),
                "fill": colors[part["family"]], "fill-opacity": ".44",
                "stroke": "#23313a", "stroke-width": ".65"})
            if part["name"].endswith("shoulder_yaw") or part["name"].endswith("hip_yaw"):
                ET.SubElement(svg, "circle", {
                    "cx": str(origin + (part["shaft"][horizontal] * 1000 - min_h) * scale),
                    "cy": str(530 - part["shaft"][2] * 1000 * 1.25),
                    "r": "2.7", "fill": "#23313a"})
    for index, (name, color) in enumerate(colors.items()):
        x = 60 + index * 150
        ET.SubElement(svg, "rect", {"x": str(x), "y": "565", "width": "13",
                                    "height": "13", "fill": color})
        legend = ET.SubElement(svg, "text", {"x": str(x + 19), "y": "576",
                                             "font-family": "sans-serif",
                                             "font-size": "12", "fill": "#23313a"})
        legend.text = name
    ET.indent(svg, space="  ")
    destination.write_text(ET.tostring(svg, encoding="unicode") + "\n")
    centers = np.array([p["center"] for p in parts])
    extents = np.array([np.abs(p["rot"]) @ p["half"] for p in parts])
    low = (centers - extents).min(axis=0) * 1000
    high = (centers + extents).max(axis=0) * 1000
    return {"min_mm": low.round(2).tolist(), "max_mm": high.round(2).tolist(),
            "span_mm": (high - low).round(2).tolist()}


def main() -> None:
    out = ROOT / "artifacts/ape-tumble-packaging-20261001"
    out.mkdir(parents=True, exist_ok=True)

    source = HERE.parent / "ape_cad/ape_spaced_v7_candidate.xml"
    tree = ET.parse(source)
    for side, sign in (("left", 1), ("right", -1)):
        yaw = tree.find(f".//body[@name='{side}_shoulder_yaw_body']")
        roll = tree.find(f".//body[@name='{side}_shoulder_roll_body']")
        assert yaw is not None and roll is not None
        yaw.set("pos", f"0 {sign * .080:.3f} .085")
        roll.set("pos", "-.025 0 -.050")
    tree.getroot().set("model", "ape_tumble_aligned_envelope_study")
    model_path = out / "aligned-shoulder-envelope.xml"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode", xml_declaration=True)

    layout = copy.deepcopy(envelope.CFG)
    layout["model"] = str(model_path.relative_to(ROOT))
    layout["tails"]["shoulder_yaw"] = [1, 0, 0]
    layout["tails"]["shoulder_roll_left"] = [0, 0, -1]
    layout["tails"]["shoulder_roll_right"] = [0, 0, -1]
    layout["tails"]["shoulder_pitch"] = [1, 0, 0]
    layout_path = out / "layout.yaml"
    layout_path.write_text(yaml.safe_dump(layout, sort_keys=False))

    envelope.CFG = layout
    envelope.MODEL = model_path
    envelope.OUT = out
    envelope.main()

    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    parts = envelope.poses(model, data) + [envelope.battery_pose(model, data)]
    projection = ROOT / "docs/design/ape-concepts/packaging-projections.svg"
    report["component_box_bounds_mm"] = draw_projections(parts, projection)
    report["illustrative_crawl"] = illustrative_crawl(model, data)
    report["design_sketch"] = "docs/design/ape-concepts/04-tumble-refined.png"
    report["study_limit"] = (
        "Source-sized motor and battery boxes only. The selected shell, structure, "
        "shaft couplings, and wiring are not yet fitted. The crawl pose is "
        "illustrative kinematics only, without force balance or shell clearance. "
        "Sampled poses come from old policies and are kinematic screens, not "
        "learned outcomes on this candidate."
    )
    report["provenance"]["source_model_sha256"] = envelope.sha256(source)
    report["provenance"]["layout_sha256"] = envelope.sha256(layout_path)
    report["provenance"]["screen_script_sha256"] = envelope.sha256(Path(__file__))
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
