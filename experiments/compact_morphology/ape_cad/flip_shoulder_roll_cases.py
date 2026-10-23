"""Test an outward motor-case orientation without moving either joint axis.

The box dimensions, motor mass, actuator, limits and control frame are held
fixed. Flipping each roll servo's output-side orientation moves its fixed-side
case center 26 mm outward in Y. This is a packaging hypothesis pending exact
horn, back-case and cable verification; v8 remains the control baseline.
"""

from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ape_spaced_v8_casedynamics.xml"
TARGET = HERE / "ape_spaced_v9_outward_shoulder_roll.xml"


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ape_spaced_v9_outward_shoulder_roll")
    for side, y in (("left", 0.013), ("right", -0.013)):
        geom = root.find(f".//geom[@name='{side}_shoulder_roll_casedynamics']")
        if geom is None:
            raise ValueError(f"missing {side} shoulder roll case")
        xyz = [float(v) for v in geom.get("pos").split()]
        assert abs(xyz[1]) == 0.013
        xyz[1] = y
        geom.set("pos", " ".join(str(v) for v in xyz))
    tree.write(TARGET, encoding="utf-8", xml_declaration=True)
    print(TARGET)


if __name__ == "__main__":
    main()
