"""Screen whether provisional 50 cm head/chest views see an object by a paw.

The exported neutral shell arm is rotated rigidly about a guessed shoulder
point. There is no elbow motion, coupled head/arm assembly, real aperture or
optical measurement. This is a visibility sensitivity test, not a CAD signoff.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from aperture_screen_50cm import (
    FIXED_PARTS, HEAD_PARTS, STL, VIEWS, _rotation_x, _screen, _target_points,
    _triangles,
)


ARM_PARTS = (
    "R_upper_arm_front", "R_upper_arm_back", "R_forearm_front",
    "R_forearm_back", "R_palm_front", "R_palm_back",
)
SHOULDER_MM = np.array([105.0, 0.0, 335.0])
PAW_NEUTRAL_MM = np.array([140.0, -10.0, 150.0])
ARM_PITCH_DEG = (-60, -90)
OBJECT_FORWARD_MM = (25, 50)
VIEW_NAMES = (
    "fixed_eye_down40", "fixed_eye_down40_yaw_right40", "pitch_eye_down",
    "pitch_eye_down_yaw_right40", "fixed_chest_down",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parts = HEAD_PARTS + FIXED_PARTS + ARM_PARTS
    triangles = {name: _triangles(STL / f"{name}.stl") for name in parts}
    fixed = {name: triangles[name] for name in HEAD_PARTS + FIXED_PARTS}
    rows = []
    for arm_pitch_deg in ARM_PITCH_DEG:
        rotation = _rotation_x(arm_pitch_deg)
        arm = {name: SHOULDER_MM + (triangles[name] - SHOULDER_MM) @ rotation.T
               for name in ARM_PARTS}
        paw = SHOULDER_MM + rotation @ (PAW_NEUTRAL_MM - SHOULDER_MM)
        for forward_mm in OBJECT_FORWARD_MM:
            center = paw + np.array([0.0, -forward_mm, 0.0])
            # Five sample points span a nominal 3 cm wide/tall object.
            points = _target_points(*center, 15.0, 15.0)
            for view_name in VIEW_NAMES:
                samples = [_screen(VIEWS[view_name], point, {**fixed, **arm})
                           for point in points]
                rows.append({
                    "arm_pitch_deg": arm_pitch_deg,
                    "object_forward_of_paw_mm": forward_mm,
                    "object_center_mm": center.tolist(),
                    "view": view_name,
                    "clear_points": sum(sample["in_frame"] and sample["occluder"] is None
                                        for sample in samples),
                    "in_frame_points": sum(sample["in_frame"] for sample in samples),
                    "occluders": [sample["occluder"] for sample in samples
                                  if sample["in_frame"] and sample["occluder"]],
                })
    result = {
        "scope": __doc__.strip(),
        "source": str(STL),
        "part_sha256": {name: hashlib.sha256((STL / f"{name}.stl").read_bytes()).hexdigest()
                        for name in parts},
        "assumed_shoulder_mm": SHOULDER_MM.tolist(),
        "assumed_paw_neutral_mm": PAW_NEUTRAL_MM.tolist(),
        "object_size_mm": [30, 30],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["arm_pitch_deg"], row["object_forward_of_paw_mm"],
              row["view"], row["clear_points"], "/", row["in_frame_points"])


if __name__ == "__main__":
    main()
