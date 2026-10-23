"""Replace the thin contact plate with the approved Cove boot's bounds.

The box is a physics sensitivity proxy, not the rounded boot mesh or CAD.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-frame50-bare"
SOURCE = ROOT / "experiments/compact_morphology/ape_design/swarm_shell_v2/output/cove-report.json"


def fmt(values):
    return " ".join(f"{float(v):.7g}" for v in values)


def main():
    source_xml = OUT / "ape-frame50-thinsole-baseline.xml"
    report = json.loads(SOURCE.read_text())
    boot = next(p for p in report["parts"] if p["name"] == "R_boot")
    lo, hi = boot["bounds_mm"]
    forward_center = -(lo[1] + hi[1]) / 2000
    lateral_center = (lo[0] + hi[0]) / 2000
    height_center = (lo[2] + hi[2]) / 2000
    half = [(hi[1] - lo[1]) / 2000, (hi[0] - lo[0]) / 2000,
            (hi[2] - lo[2]) / 2000]
    tree = ET.parse(source_xml)
    for side, sign in (("left", 1), ("right", -1)):
        foot = tree.getroot().find(f".//body[@name='{side}_foot']")
        sole = foot.find(f"geom[@name='{side}_sole']")
        ankle_x, ankle_y, ankle_z = (-.002, sign * .063, .043)
        sole.set("pos", fmt([forward_center - ankle_x,
                             sign * lateral_center - ankle_y,
                             height_center - ankle_z]))
        sole.set("size", fmt(half))
    path = OUT / "ape-frame50-fullboot-diagnostic.xml"
    ET.indent(tree.getroot(), space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    outcome = {"stage": "Cove bounding-box contact sensitivity, not exact boot CAD",
               "source_shape": str(SOURCE.relative_to(ROOT)),
               "source_shape_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "thinsole_body_sha256": hashlib.sha256(source_xml.read_bytes()).hexdigest(),
               "fullboot_body_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "cove_R_boot_bounds_mm": boot["bounds_mm"],
               "box_half_size_m": half,
               "mass_kg_unchanged": round(float(model.body_mass.sum()), 4),
               "caveat": "Cove bounds overfill a rounded shell; actual support, contact and inertia require the assembled boot."}
    (OUT / "fullboot-diagnostic.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
