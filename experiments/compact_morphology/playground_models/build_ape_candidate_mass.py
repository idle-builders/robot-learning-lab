"""Expose sourced battery and head-motor masses in the v5 ape proxy.

The source body is retained. This changes inertia placement, not joint order,
collision masks, motor response, or an unverified printed-shell clearance.
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import mujoco


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ape_sourced_v5_collision_proxy.xml"
OUTPUT = HERE / "ape_sourced_v6_candidate_mass.xml"
NARROW_OUTPUT = HERE / "ape_sourced_v7_candidate_narrow55.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sole-width-mm", type=int, choices=(55, 60), default=60)
    args = parser.parse_args()
    output = NARROW_OUTPUT if args.sole_width_mm == 55 else OUTPUT
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ape_candidate_mass_narrow55_v7" if args.sole_width_mm == 55 else "ape_candidate_mass_v6")
    bodies = {body.get("name"): body for body in root.iter("body")}

    pelvis = bodies["pelvis"]
    pelvis_shell = pelvis.find("geom")
    assert pelvis_shell is not None
    pelvis_shell.set("mass", str(float(pelvis_shell.get("mass")) - 0.107))
    pelvis.insert(2, ET.Element("geom", {
        "name": "candidate_4s_battery_mass",
        "type": "box", "pos": "0 0 0.005", "size": "0.030 0.015 0.01525",
        "mass": "0.107", "contype": "0", "conaffinity": "0",
        "rgba": "0 0 0 0",
    }))

    for name, shell_mass, motor_name in (
        ("head_yaw_body", 0.02986547085201794, "candidate_head_yaw_motor_mass"),
        ("head", 0.100, "candidate_head_pitch_motor_mass"),
    ):
        body = bodies[name]
        shell = body.find("geom")
        assert shell is not None
        assert abs(float(shell.get("mass")) - shell_mass) < 1e-10
        shell.set("mass", str(shell_mass - 0.018))
        body.insert(1, ET.Element("geom", {
            "name": motor_name,
            "type": "box", "size": "0.010 0.017 0.013",
            "mass": "0.018", "contype": "0", "conaffinity": "0",
            "rgba": "0 0 0 0",
        }))

    if args.sole_width_mm == 55:
        for side in ("left", "right"):
            for element in root.iter():
                if element.get("name") in (f"{side}_sole", f"{side}_sole_touch"):
                    size = element.get("size").split()
                    assert size == ["0.045", "0.030", "0.012"]
                    size[1] = "0.0275"
                    element.set("size", " ".join(size))

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    new = mujoco.MjModel.from_xml_path(str(output))
    assert old.nu == new.nu == 22
    assert abs(sum(old.body_mass) - sum(new.body_mass)) < 1e-9
    print(f"{output}: {new.nu} actuators, {sum(new.body_mass):.9f} kg")


if __name__ == "__main__":
    main()
