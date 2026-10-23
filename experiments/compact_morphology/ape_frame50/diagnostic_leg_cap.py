"""Make a peak-torque diagnostic body; never use this as a rated-duty model."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ape-frame50-bare"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg-cap-nm", type=float, required=True)
    parser.add_argument("--source-body", type=Path)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    cfg = yaml.safe_load((HERE / "frame_contract.yaml").read_text())
    rated = cfg["actuators"]["leg"]["rated_effort_nm"]
    stall = cfg["actuators"]["leg"]["stall_effort_nm"]
    if not rated < args.leg_cap_nm <= stall:
        parser.error(f"Diagnostic cap must be above rated {rated} and at most stall {stall} Nm")
    source = args.source_body or OUT / "ape-frame50-nominal.xml"
    tree = ET.parse(source)
    names = set()
    for actuator in tree.getroot().find("actuator"):
        joint = actuator.get("joint")
        if joint and any(joint.endswith("_" + axis) for axis in
                         ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")):
            actuator.set("forcerange", f"{-args.leg_cap_nm:g} {args.leg_cap_nm:g}")
            names.add(joint)
    if len(names) != 10:
        raise ValueError(f"Expected ten leg axes, found {len(names)}")
    path = OUT / (args.output_name or f"ape-frame50-legcap{args.leg_cap_nm:g}-diagnostic.xml")
    ET.indent(tree.getroot(), space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    report = {"stage": "peak leg-torque sensitivity ONLY; no continuous-duty claim",
              "source_body_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "diagnostic_body_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_rated_leg_effort_nm": rated,
              "source_stall_leg_effort_nm": stall,
              "diagnostic_leg_cap_nm": args.leg_cap_nm,
              "changed_axes": sorted(names)}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
