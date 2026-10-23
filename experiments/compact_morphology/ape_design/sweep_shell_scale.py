"""Screen uniform shell growth with fixed, source-sized servo cases.

Every joint anchor and exterior dimension grows by `scale`; servo cases,
their shaft offsets, and battery dimensions do not. Saved joint angles remain
the same. This is a geometry/cost sensitivity study, not a new robot model.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import sketch_shell as shell
from search_shell_cases import choices, choose_group
from shell_aligned_model import OUT, make_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ape_cad"))
import build as envelope  # noqa: E402

SCALES = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0)
SHELL_MASS_KG = .724


def choose_cases(axes: int, scale: float):
    model = mujoco.MjModel.from_xml_path(str(make_model(axes)))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    menu = choices(model, data, scale)
    groups = []
    for side in ("left", "right"):
        groups.append([f"{side}_{name}" for name in
                       ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")])
        arm = ["shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"]
        if axes == 22:
            arm.insert(0, "shoulder_yaw")
        groups.append([f"{side}_{name}" for name in arm])
    groups.append(["head_yaw", "head_pitch"])
    return {choice.name: choice for group in groups
            for choice in choose_group(group, menu)}


def replay_pairs(selected: dict, sample: dict, reference: dict, scale: float,
                 battery_center: np.ndarray | None = None):
    baseline = {c["joint"]: c for c in reference["cases"]}
    observed = {c["joint"]: c for c in sample["cases"]}
    moving = []
    for name, choice in selected.items():
        old = baseline[name]
        frame = observed[name]
        old_r = np.asarray(old["rotation_matrix"])
        now_r = np.asarray(frame["rotation"])
        delta_r = now_r @ old_r.T
        old_c = np.asarray(old["center_mm"]) / 1000
        now_c = np.asarray(frame["center_m"])
        delta_t = now_c - delta_r @ old_c
        pose = dict(choice.pose)
        pose["center"] = delta_r @ pose["center"] + scale * delta_t
        pose["rot"] = delta_r @ pose["rot"]
        moving.append(pose)
    battery_r = np.asarray(sample["battery"]["rotation"])
    battery_t = (np.asarray(sample["battery"]["center_m"])
                 - battery_r @ np.array([0., 0., .191]))
    battery_center = (np.array([0., 0., .191 * scale])
                      if battery_center is None else battery_center)
    moving.append({"name": "battery_4s_850", "center":
                   battery_r @ battery_center + scale * battery_t,
                   "rot": battery_r,
                   "half": np.array([.030, .015, .01525])})
    return envelope.clashes(moving, 1)


def main():
    shell.build()
    output = {"method": "Fixed manufacturer-sized cases and 60×30×30.5mm battery; uniformly scale all joint anchors and exterior, retain case-to-shaft offset and saved joint angles. Orient each case afresh around its shaft on an eight-angle menu. Poses are old-policy joint samples, not new policy outcomes. Neutral corners use scaled shell interior; sampled motion checks case-case only, not moving shell or loads. A collision-free row would still need mounts, bearing paths, cables, shell cutouts and new policy training.",
              "nominal_shell_height_mm": 349.6, "shell_mass_kg_at_2mm": SHELL_MASS_KG,
              "rows": []}
    for axes in (22, 20):
        reference = json.loads((OUT / f"case-screen-{axes}.json").read_text())
        samples = json.loads((OUT / f"motion-screen-{axes}.json").read_text())["samples"]
        fixed_mass = reference["known_motor_battery_mass_kg"]
        for scale in SCALES:
            selected = choose_cases(axes, scale)
            neutral = [choice.pose for choice in selected.values()] + [{
                "name": "battery_4s_850", "center": np.array([0., 0., .191 * scale]),
                "rot": np.eye(3), "half": np.array([.030, .015, .01525])}]
            pairs = envelope.clashes(neutral, 1)
            sampled = [len(replay_pairs(selected, sample, reference, scale))
                       for sample in samples]
            mass_lb = fixed_mass + SHELL_MASS_KG * scale ** 2
            torque_ratio = scale * mass_lb / (fixed_mass + SHELL_MASS_KG)
            row = {"axes": axes, "scale": scale,
                   "height_mm": round(349.6 * scale, 1),
                   "neutral_case_pairs": len(pairs), "neutral_pair_names": pairs,
                   "neutral_outside_interior_corners": sum(c.outside_corners for c in selected.values()),
                   "worst_saved_pose_case_pairs": max(sampled),
                   "saved_pose_frames_with_pairs": sum(n > 0 for n in sampled),
                   "sampled_frames": len(sampled),
                   "known_parts_mass_lower_bound_kg": round(mass_lb, 3),
                   "known_parts_gravity_lever_index": round(torque_ratio, 2)}
            output["rows"].append(row)
            print(f"{axes:2} axes {scale:.2f}× {row['height_mm']:5.1f}mm: "
                  f"{len(pairs):2} neutral pairs, {row['neutral_outside_interior_corners']:2} "
                  f"outside corners, worst {max(sampled):2} pairs")
    (OUT / "scale-sweep.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
