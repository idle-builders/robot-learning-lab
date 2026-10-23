"""Check exact selected-body right-to-left kinematic reach reflection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRACE = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/evaluation/step-feedback.npz"
OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-left-mirror-v1/mirror-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected selected 21-axis body")
    with np.load(TRACE) as trace:
        poses = trace["qpos"].copy()
        physical_contact = trace["finger_toy_contact"].copy()
    right = mujoco.MjData(model)
    left = mujoco.MjData(model)
    reflection = np.diag([1.0, -1.0, 1.0])
    metrics = []
    for tick, full in enumerate(poses):
        source = full[:model.nq].astype(np.float64)
        right.qpos[:] = source
        mujoco.mj_forward(model, right)
        mirrored = source.copy()
        mirrored[1] *= -1
        rotation = Rotation.from_quat(source[3:7], scalar_first=True).as_matrix()
        mirrored[3:7] = Rotation.from_matrix(reflection @ rotation @ reflection).as_quat(
            scalar_first=True)
        for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch",
                       "shoulder_yaw", "shoulder_roll", "shoulder_pitch",
                       "elbow_pitch", "gripper"):
            a = model.joint(f"left_{suffix}").qposadr[0]
            b = model.joint(f"right_{suffix}").qposadr[0]
            sign = -1 if suffix.endswith(("yaw", "roll")) else 1
            mirrored[a], mirrored[b] = sign * source[b], sign * source[a]
        mirrored[model.joint("head_yaw").qposadr[0]] *= -1
        left.qpos[:] = mirrored
        mujoco.mj_forward(model, left)
        row = {"tick": tick, "physical_right_contact": bool(physical_contact[tick])}
        for name in ("hand", "finger_tip"):
            a = right.geom_xpos[model.geom(f"right_{name}").id]
            b = left.geom_xpos[model.geom(f"left_{name}").id]
            row[f"{name}_mirror_error_mm"] = float(1000 * np.linalg.norm(b - reflection @ a))
        for name in ("sole_center",):
            for side in ("left", "right"):
                other = "right" if side == "left" else "left"
                a = right.site_xpos[model.site(f"{side}_{name}").id]
                b = left.site_xpos[model.site(f"{other}_{name}").id]
                row[f"{side}_{name}_mirror_error_mm"] = float(1000 * np.linalg.norm(
                    b - reflection @ a))
        metrics.append(row)
    fields = [key for key in metrics[0] if key.endswith("_mirror_error_mm")]
    report = {
        "schema": "robot-learning-lab.cove50-left-reach-mirror-audit.v1",
        "method": "mirror saved native right qpos and compare selected-model hand/finger/sole forward kinematics",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(BODY), "source_trace_sha256": sha(TRACE),
        "source_trace": str(TRACE.relative_to(ROOT)),
        "frames": len(metrics), "physical_right_contact_frames": int(np.sum(physical_contact)),
        "maximum_reflection_error_mm": {key: max(row[key] for row in metrics)
                                        for key in fields},
        "limits": ["Kinematic equality only; this does not replay motors, contact or a controller on the left.",
                   "A mirrored right toy position is a new left scene, not a free acquired object or perception input."],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
