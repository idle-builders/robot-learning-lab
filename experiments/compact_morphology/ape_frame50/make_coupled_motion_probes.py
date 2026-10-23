"""Small, authored CAD clearance probes; these are not policies or safety limits."""

import json
from pathlib import Path


OUT = Path(__file__).with_name("coupled_motion_probes.json")
JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
ZERO = {name: 0.0 for name in JOINTS}

# Angles in degrees. Each path starts at the neutral frame and ends at a
# plausible task posture. Their purpose is to expose mechanism conflicts.
TARGETS = {
    "shallow_squat": ({"hip_pitch": -20, "knee": 40, "ankle_pitch": -20},
                      {"hip_pitch": -20, "knee": 40, "ankle_pitch": -20}),
    "deep_squat": ({"hip_pitch": -40, "knee": 90, "ankle_pitch": -35},
                   {"hip_pitch": -40, "knee": 90, "ankle_pitch": -35}),
    "left_turn_shape": ({"hip_yaw": 15, "hip_roll": 5, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10},
                        {"hip_yaw": -15, "hip_roll": -5, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10}),
    "right_turn_shape": ({"hip_yaw": -12, "hip_roll": -4, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10},
                         {"hip_yaw": 12, "hip_roll": 4, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10}),
    "left_shift_shape": ({"hip_roll": -7, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10},
                         {"hip_roll": 7, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10}),
    "right_shift_shape": ({"hip_roll": 8, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10},
                          {"hip_roll": -8, "hip_pitch": -10, "knee": 20, "ankle_pitch": -10}),
    "left_step_shape": ({"hip_pitch": -15, "knee": 55, "ankle_pitch": -20},
                        {"hip_pitch": 0, "knee": 0, "ankle_pitch": 0}),
}

poses = []
for motion, sides in TARGETS.items():
    for step in range(11):
        fraction = step / 10
        poses.append({
            "name": f"{motion}_{step:02d}",
            "left": {name: fraction * sides[0].get(name, 0) for name in JOINTS},
            "right": {name: fraction * sides[1].get(name, 0) for name in JOINTS},
        })
OUT.write_text(json.dumps({"schema": "ape-frame50-cad-clearance-probes.v1",
                           "provenance": "authored mechanism probes, not trained actions or balanced robot motions",
                           "poses": poses}, indent=2) + "\n")
print(OUT, len(poses))
