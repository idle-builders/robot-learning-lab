"""One no-exterior-change axis-spacing trial for the 20-axis direct-drive body.

Moving the two close output axes apart is a kinematic experiment, not a
buildable joint design. It tests whether ordinary source-sized cases could at
least avoid one another inside the approved visual silhouette.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco

import sketch_shell as shell
from search_shell_cases import OUT, run
from shell_aligned_model import body, make_model


def trial_model():
    original = make_model(20)
    tree = ET.parse(original)
    root = tree.getroot()
    root.set("model", "ape_sketch_shell_v2_20_axis_spaced_probe")
    for side in ("left", "right"):
        body(root, f"{side}_hip_roll").set("pos", "0 0 -.016")  # z=190
        body(root, f"{side}_thigh").set("pos", "0 0 -.035")  # z=155
        body(root, f"{side}_shin").set("pos", "0 0 -.050")  # z=105
        body(root, f"{side}_shoulder_roll_body").set("pos",
            f"0 {'.009' if side == 'left' else '-.009'} .005")  # z=263
        body(root, f"{side}_arm").set("pos",
            f".001 {'.018' if side == 'left' else '-.018'} -.035")  # z=228
        body(root, f"{side}_forearm").set("pos",
            f"0 {'.009' if side == 'left' else '-.009'} -.029")  # z=199
        thigh = body(root, f"{side}_thigh").find("geom[@type='capsule']")
        arm = body(root, f"{side}_arm").find("geom[@type='capsule']")
        assert thigh is not None and arm is not None
        thigh.set("fromto", "0 0 0 0 0 -.050")
        arm.set("fromto", "0 0 0 0 0 -.029")
    path = OUT / "shell-aligned-20-spaced-probe.xml"
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(path))
    return path


def main():
    shell.build()
    report = run(20, trial_model())
    (OUT / "case-screen-20-spaced-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print("20 spaced probe:", len(report["neutral_case_clashes_1mm"]),
          "case clashes,", report["outside_interior_corners"], "outside corners")
    print(report["neutral_case_clashes_1mm"])


if __name__ == "__main__":
    main()
