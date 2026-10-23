"""Screen roll-servo case orientation around the unchanged shoulder shaft.

Uses saved live policy poses as a kinematic collision gate. This does not prove
the altered mass/inertia and collision dynamics will preserve learned walking.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ape_spaced_v8_casedynamics.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.motion) as f:
        poses = f["qpos"]
    results = []
    for angle_deg in range(-180, 181, 15):
        root = ET.parse(SOURCE).getroot()
        root.set("model", f"ape_case_roll_{angle_deg}")
        for side, sign in (("left", 1), ("right", -1)):
            geom = root.find(f".//geom[@name='{side}_shoulder_roll_casedynamics']")
            pos = np.fromstring(geom.get("pos"), sep=" ")
            quat = np.fromstring(geom.get("quat"), sep=" ")
            old_rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            turning = Rotation.from_rotvec(np.radians(angle_deg * sign) * np.array([1., 0, 0]))
            pos[:] = np.array([0, 0, -.05]) + turning.apply(pos - np.array([0, 0, -.05]))
            q = (turning * old_rotation).as_quat()
            geom.set("pos", " ".join(f"{x:.9f}" for x in pos))
            geom.set("quat", " ".join(f"{x:.9f}" for x in [q[3], q[0], q[1], q[2]]))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        count = 0
        deepest = 0.0
        pairs = Counter()
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            hit = False
            for contact in data.contact[:data.ncon]:
                a = int(model.geom_bodyid[contact.geom1])
                b = int(model.geom_bodyid[contact.geom2])
                if a == 0 or b == 0 or a == b:
                    continue
                hit = True
                deepest = max(deepest, -float(contact.dist))
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom:{gid}"
                         for gid in (contact.geom1, contact.geom2)]
                pairs[tuple(sorted(names))] += 1
            count += hit
        results.append({"angle_deg_from_inward": angle_deg,
                        "self_contact_frames": count,
                        "max_penetration_mm": round(deepest * 1000, 3),
                        "pairs": [{"geoms": pair, "contact_points": hits} for pair, hits in pairs.most_common(5)]})
        print(angle_deg, count, round(deepest * 1000, 3), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source_model": str(SOURCE),
                                    "motion": str(args.motion),
                                    "method": "saved-pose collision screen; no dynamic policy replay",
                                    "results": results}, indent=2))


if __name__ == "__main__":
    main()
