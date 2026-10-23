"""Move both shoulder yaw assemblies 10 mm forward to clear hip/arm cases.

The saved-pose screen in artifacts/ape-cad-retrain-20260930 shows that this
single packaging change removes the v8 shoulder-roll/hip-pitch case overlap.
The axes, joint limits and motor orientations stay unchanged. A live learned
policy replay is still required because the moving arm mass shifts forward.
"""

from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ape_spaced_v8_casedynamics.xml"
TARGET = HERE / "ape_spaced_v10_shoulders_forward.xml"


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "ape_spaced_v10_shoulders_forward")
    for side in ("left", "right"):
        for query in (f".//body[@name='{side}_shoulder_yaw_body']",
                      f".//geom[@name='{side}_shoulder_yaw_casedynamics']"):
            node = root.find(query)
            if node is None:
                raise ValueError(query)
            pos = [float(v) for v in node.get("pos").split()]
            pos[0] += .010
            node.set("pos", " ".join(str(v) for v in pos))
    tree.write(TARGET, encoding="utf-8", xml_declaration=True)
    print(TARGET)


if __name__ == "__main__":
    main()
