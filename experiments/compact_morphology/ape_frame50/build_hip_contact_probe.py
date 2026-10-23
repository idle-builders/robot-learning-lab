"""Add native MuJoCo contact pairs for two CAD-derived hip pinch points.

This is a narrow experimental collision proxy, not a full CAD collision mesh.
Its zero-mass geoms leave the current body mass/action contract unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate


BODY = ROOT / "artifacts/ape-frame50-supported-body/ape-frame50-supported-body.xml"
OUT = ROOT / "artifacts/ape-frame50-supported-body/ape-frame50-hip-contact-probe.xml"
SCREENS = {
    "independent_grid": "hardware-candidate-envelope-arm-relief.json",
    "coordinated_paths": "hardware-coupled-motion-probes-relieved.json",
    "gait_edge": "hardware-learned-walk-edge-probes-relieved.json",
    "learned_walk_world0": "hardware-all-frames-arm-relief.json",
    "contact_walk_world0": "hardware-all-frames-hip-contact.json",
}


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def sim_bounds(shape):
    bounds = shape.bounding_box()
    lo = np.array([bounds.min.X, bounds.min.Y, bounds.min.Z], dtype=float)
    hi = np.array([bounds.max.X, bounds.max.Y, bounds.max.Z], dtype=float)
    center = (lo + hi) / 2
    # Cove CAD is right/rear/up in millimetres; MJCF is forward/left/up.
    return np.array([-center[1], center[0], center[2]]) / 1000, (hi - lo)[[1, 0, 2]] / 2000


def screen_contacts(model: mujoco.MjModel) -> dict:
    """Compare only these two native pairs with existing exact-CAD pose reports."""
    data = mujoco.MjData(model)
    pairs = {
        frozenset((mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"{side}_yaw_motor_housing_tube_proxy"),
                   mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"{side}_{part}_proxy"))): (side, part)
        for side in ("left", "right")
        for part in ("pitch_seat_medial", "pitch_fork_medial")
    }
    base = ROOT / "artifacts/ape-frame50-supported-body/learned-replay"
    results = {}
    for label, filename in SCREENS.items():
        path = base / filename
        cad = json.loads(path.read_text())
        trace = None
        if label.endswith("walk_world0"):
            trace_name = ("forward-hip-contact.npz" if label == "contact_walk_world0"
                          else "forward-arm-relief.npz")
            trace = np.load(base / trace_name)["qpos"]
        counts = {part: {key: 0 for key in ("TP", "FP", "FN", "TN")}
                  for part in ("pitch_seat_medial", "pitch_fork_medial")}
        for row in cad["reports"]:
            data.qpos[:] = trace[row["frame"]] if trace is not None else model.qpos0
            if trace is None:
                for side, detail in row["sides"].items():
                    for name, degrees in detail["joint_deg"].items():
                        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                 f"{side}_{name}")
                        data.qpos[model.jnt_qposadr[joint]] = np.radians(degrees)
            mujoco.mj_forward(model, data)
            native = {pairs[frozenset((c.geom1, c.geom2))] for c in data.contact
                      if c.dist < 0 and frozenset((c.geom1, c.geom2)) in pairs}
            for side, detail in row["sides"].items():
                for part in counts:
                    actual = any(hit[0] == f"hip_roll_{part}" and
                                 hit[1] == "yaw_motor_housing_tube"
                                 for hit in detail["hard_hits"])
                    predicted = (side, part) in native
                    counts[part]["TP" if actual and predicted else
                                 "FP" if predicted else "FN" if actual else "TN"] += 1
        results[label] = {"cad_report": str(path.relative_to(ROOT)),
                          "cad_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "poses": len(cad["reports"]), "by_part": counts}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, default=BODY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--screen", action="store_true",
                        help="compare native contacts with saved exact-CAD pose reports")
    args = parser.parse_args()
    source, out = args.body.resolve(), args.out.resolve()
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tree = ET.parse(source)
    root = tree.getroot()
    pelvis = root.find("worldbody/body[@name='pelvis']")
    assert pelvis is not None
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    hardware, _ = candidate()
    source_bounds = {}
    for side, sign in (("left", 1), ("right", -1)):
        stages = hip.frame(hardware, sign, 0, 0, 0)
        yaw = pelvis.find(f"body[@name='{side}_hip_yaw_body']")
        assert yaw is not None
        roll = yaw.find(f"body[@name='{side}_hip_roll_body']")
        assert roll is not None
        for stage, part, body, kind in (
            ("fixed", "yaw_motor_housing_tube", pelvis, "cylinder"),
            ("roll", "pitch_seat_medial", roll, "box"),
            ("roll", "pitch_fork_medial", roll, "box"),
        ):
            center, half = sim_bounds(stages[stage][part])
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.attrib["name"])
            rotation = data.xmat[body_id].reshape(3, 3)
            position = rotation.T @ (center - data.xpos[body_id])
            name = f"{side}_{part}_proxy"
            size = np.array((half[0], half[2])) if kind == "cylinder" else half
            ET.SubElement(body, "geom", {
                "name": name, "type": kind, "pos": fmt(position), "size": fmt(size),
                "mass": "0", "contype": "0", "conaffinity": "0", "group": "5",
                "rgba": "0.9 0.2 0.15 0.3" if kind == "cylinder" else "0.15 0.8 0.5 0.3",
            })
            source_bounds[name] = {"cad_part": part, "center_m": center.tolist(),
                                   "half_extent_m": half.tolist()}
        for part in ("pitch_seat_medial", "pitch_fork_medial"):
            ET.SubElement(contact, "pair", {
                "name": f"{side}_tube_{part}",
                "geom1": f"{side}_yaw_motor_housing_tube_proxy",
                "geom2": f"{side}_{part}_proxy", "condim": "1",
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(out))
    if not np.isclose(compiled.body_mass.sum(), model.body_mass.sum(), atol=1e-9):
        raise ValueError("contact probes changed body mass")
    report = {"stage": "CAD-bounds native MuJoCo hip contact probe; two part pairs per side only",
              "source_body_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "cad_source_sha256": hashlib.sha256(
                  (ROOT / "experiments/compact_morphology/ape_frame50/pelvis_tube_candidate.py").read_bytes()
              ).hexdigest(),
              "body_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
              "mass_kg": float(compiled.body_mass.sum()), "explicit_pairs": compiled.npair,
              "source_bounds": source_bounds,
              "limits": "bounding primitives approximate two nonmating CAD pairs, not the complete assembly; nominal contact parameters and loaded clearance unverified"}
    if args.screen:
        report["screen"] = screen_contacts(compiled)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("body_sha256", "mass_kg", "explicit_pairs")}))


if __name__ == "__main__":
    main()
