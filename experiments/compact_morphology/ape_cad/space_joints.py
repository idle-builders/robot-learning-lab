"""Make a disposable serial-offset hypothesis from the learned ape model.

This changes kinematic origins, not policy weights. Run build.py against the
result to see if published motor envelopes fit before spending GPU time.
"""

from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v6_candidate_mass.xml"
TARGET = ROOT / "experiments/compact_morphology/ape_cad/ape_spaced_v7_candidate.xml"


def body(root, name):
    item = root.find(f".//body[@name='{name}']")
    if item is None:
        raise ValueError(name)
    return item


root = ET.parse(SOURCE).getroot()
root.set("model", "ape_spaced_v7_candidate")
for side, sign in (("left", 1), ("right", -1)):
    body(root, f"{side}_hip_yaw").set("pos", f"0 {sign * .037:.3f} .060")
    body(root, f"{side}_hip_roll").set("pos", "0 0 -.060")
    body(root, f"{side}_thigh").set("pos", f".025 {sign * .043:.3f} 0")
    body(root, f"{side}_shoulder_yaw_body").set("pos", f".080 {sign * .080:.3f} .085")
    body(root, f"{side}_shoulder_roll_body").set("pos", "0 0 -.050")
    body(root, f"{side}_arm").set("pos", f".025 {sign * .050:.3f} 0")
body(root, "head_yaw_body").set("pos", "0 0 .094")
body(root, "head").set("pos", "0 0 .030")
root.find(".//geom[@name='candidate_4s_battery_mass']").set("pos", "0 0 .040")
ET.indent(root, space="  ")
TARGET.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n")
print(TARGET)
