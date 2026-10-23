"""Build and screen additional CAD-derived hard-part contacts on the 50 cm ape.

Local experiment: compare simple native contact proxies with exact CAD on the
existing broad pose grid before accepting them into the training body.
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
from experiments.compact_morphology.ape_frame50.build_hip_contact_probe import (
    fmt, screen_contacts, sim_bounds,
)
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate

SOURCE = ROOT / "artifacts/ape-frame50-supported-body/ape-frame50-hip-contact-probe.xml"
GRID = ROOT / "artifacts/ape-frame50-supported-body/learned-replay/hardware-candidate-envelope-arm-relief.json"
RUN = ROOT / "artifacts/ape-frame50-supported-body/learned-replay"
SCREENS = {
    "grid": (GRID, None),
    "coordinated": (RUN / "hardware-coupled-motion-probes-relieved.json", None),
    "gait_edges": (RUN / "hardware-learned-walk-edge-probes-relieved.json", None),
    "walk": (RUN / "hardware-all-frames-hip-contact.json", RUN / "forward-hip-contact.npz"),
    "motor_adapted": (RUN / "hardware-all-frames-frame50-motor-adapted-seed144.json",
                      RUN / "forward-frame50-motor-adapted-seed144.npz"),
}
PAIRS = (
    ("pitch_motor_medial_stock", "yaw_belt_123_3MGT_15"),
    ("pitch_fork_lateral", "yaw_housing_outboard_rib"),
    ("pitch_fork_lateral", "yaw_housing_arm"),
    ("pitch_fork_medial", "yaw_motor_housing_arm"),
    ("pitch_fork_medial", "yaw_housing_arm"),
)


def build(out: Path) -> mujoco.MjModel:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    pelvis = root.find("worldbody/body[@name='pelvis']")
    assert pelvis is not None
    contact = root.find("contact")
    assert contact is not None
    base = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(base)
    mujoco.mj_forward(base, data)
    hardware, _ = candidate()
    for side, sign in (("left", 1), ("right", -1)):
        stages = hip.frame(hardware, sign, 0, 0, 0)
        roll = pelvis.find(f"body[@name='{side}_hip_yaw_body']/body[@name='{side}_hip_roll_body']")
        assert roll is not None
        for stage, name, body in (("roll", name, roll)
                                  for name in dict.fromkeys(pair[0] for pair in PAIRS)):
            center, half = sim_bounds(stages[stage][name])
            body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, body.attrib["name"])
            rotation = data.xmat[body_id].reshape(3, 3)
            local = rotation.T @ (center - data.xpos[body_id])
            ET.SubElement(body, "geom", {
                "name": f"{side}_{name}_extra_proxy", "type": "box",
                "pos": fmt(local), "size": fmt(half), "mass": "0",
                "contype": "0", "conaffinity": "0", "group": "5",
                "rgba": "0.9 0.2 0.2 0.2",
            })
        for stage, name, body in (("fixed", name, pelvis)
                                  for name in dict.fromkeys(pair[1] for pair in PAIRS)):
            center, half = sim_bounds(stages[stage][name])
            body_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, body.attrib["name"])
            rotation = data.xmat[body_id].reshape(3, 3)
            local = rotation.T @ (center - data.xpos[body_id])
            ET.SubElement(body, "geom", {
                "name": f"{side}_{name}_extra_proxy", "type": "box",
                "pos": fmt(local), "size": fmt(half), "mass": "0",
                "contype": "0", "conaffinity": "0", "group": "5",
                "rgba": "0.9 0.2 0.2 0.2",
            })
        for moving, fixed in PAIRS:
            ET.SubElement(contact, "pair", {
                "name": f"{side}_{moving}_{fixed}",
                "geom1": f"{side}_{moving}_extra_proxy",
                "geom2": f"{side}_{fixed}_extra_proxy", "condim": "1",
            })
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(out))
    if not np.isclose(model.body_mass.sum(), base.body_mass.sum(), atol=1e-9):
        raise ValueError("additional contact proxies changed the body mass")
    return model


def compare(model: mujoco.MjModel, report: Path, trace: Path | None) -> dict:
    data = mujoco.MjData(model)
    rows = json.loads(report.read_text())["reports"]
    qpos = np.load(trace)["qpos"] if trace else None
    counts = {"|".join(pair): {k: 0 for k in ("TP", "FP", "FN", "TN")}
              for pair in PAIRS}
    ids = {}
    for side in ("left", "right"):
        for pair in PAIRS:
            ids[frozenset(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                             f"{side}_{p}_extra_proxy") for p in pair)] = (side, pair)
    for row in rows:
        data.qpos[:] = qpos[row["frame"]] if qpos is not None else model.qpos0
        if qpos is None:
            for side, detail in row["sides"].items():
                for name, degrees in detail["joint_deg"].items():
                    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
                    data.qpos[model.jnt_qposadr[j]] = np.radians(degrees)
        mujoco.mj_forward(model, data)
        native = {ids[frozenset((c.geom1, c.geom2))] for c in data.contact
                  if c.dist < 0 and frozenset((c.geom1, c.geom2)) in ids}
        for side, detail in row["sides"].items():
            for pair in PAIRS:
                actual = any(hit[0] == f"hip_roll_{pair[0]}" and hit[1] == pair[1]
                             for hit in detail["hard_hits"])
                predicted = (side, pair) in native
                counts["|".join(pair)]["TP" if actual and predicted else
                                         "FP" if predicted else
                                         "FN" if actual else "TN"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true",
                        help="regenerate the current CAD-sized contact body without comparing historical CAD reports")
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build(out)
    report = {
        "stage": "experimental CAD-bounds native contacts for five additional hip pairs per side",
        "source_body": str(SOURCE.relative_to(ROOT)),
        "source_body_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "body_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "mass_kg": float(model.body_mass.sum()),
        "explicit_pairs": model.npair,
        "new_pairs": [list(pair) for pair in PAIRS],
        "existing_pair_screens": screen_contacts(model),
        "screens": ({} if args.build_only else
                    {name: {"cad_report": str(paths[0].relative_to(ROOT)),
                            "cad_report_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                            "poses": len(json.loads(paths[0].read_text())["reports"]),
                            "by_pair": compare(model, *paths)}
                     for name, paths in SCREENS.items()}),
        "limits": ("Five additional selected pairs only; box bounds are not exact surfaces. "
                   "Historical pose classification was intentionally skipped for the changed CAD/body. "
                   "Run new matched CAD reports and a learned-policy replay before comparing contacts."
                   if args.build_only else
                   "Five additional selected pairs only; box bounds are not exact surfaces. The rejected lateral-seat/outboard-rib box yielded 12 false positives on the grid and 4 on coordinated paths. No new policy replay or loaded joint qualification follows from pose classification."),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"body_sha256": report["body_sha256"],
                      "contact_pairs": model.npair,
                      "grid": (report["screens"]["grid"]["by_pair"]
                               if not args.build_only else "not screened")}, indent=2))


if __name__ == "__main__":
    main()
