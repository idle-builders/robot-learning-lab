"""Make an unselected right-hand contact coupon from the narrow-bridge body."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/low-pad-coupon-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUT / "plan.json")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    source_path = ROOT / plan["source_body"]
    tree = ET.parse(source_path)
    root = tree.getroot()
    updates = plan["geom_updates"]
    before = {}
    for name, attributes in updates.items():
        found = root.findall(f".//geom[@name='{name}']")
        if len(found) != 1:
            raise ValueError(f"Expected one geometry named {name}")
        geom = found[0]
        before[name] = {key: geom.get(key) for key in attributes}
        for key, value in attributes.items():
            geom.set(key, value)
    body_path = out / "candidate-training.xml"
    tree.write(body_path, encoding="unicode")
    original = mujoco.MjModel.from_xml_path(str(source_path))
    candidate = mujoco.MjModel.from_xml_path(str(body_path))
    if (original.nq, original.nu) != (candidate.nq, candidate.nu):
        raise ValueError("The contact coupon changed the 21-axis action or state interface")
    report = {"schema": plan["schema"].replace("-plan.", "."),
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "source_body_sha256": sha(source_path), "candidate_body_sha256": sha(body_path),
              "plan_sha256": sha(plan_path), "generator_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__, "source_geoms": before,
              "candidate_geoms": updates, "nq": candidate.nq, "nu": candidate.nu,
              "limit": "Unselected, right-hand-only collision coupon. It changes contact geometry while preserving the 21-axis interface. Static and native force/retention evidence are required; this is not a selected mechanical design or qualified grip."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
