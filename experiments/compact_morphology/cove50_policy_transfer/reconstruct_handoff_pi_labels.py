"""Recover observed-goal PI arm labels from retained selected-body handoff traces.

The native qpos was saved before each action. Recompute the same root-local
cube-to-hand observation and frozen PI recurrence used in that rollout; no
physics is rerun or target state inferred from a rendered video.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-squat-reach-handoff-20261010"
OUT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/pi-reconstruction"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ARRAYS = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/controller-arrays.npz"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
BOUND = .1
XY_SCALE = .01


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(fraction: float) -> float:
    x = np.clip(fraction, 0, 1)
    return float(x * x * (3 - 2 * x))


def main() -> None:
    matrix = json.loads((SOURCE / "summary-matrix.json").read_text())
    cases = [row for row in matrix["cases"] if row["body"] == "selected" and
             row["pass"] and row["stage"] in ("development route", "reserved route")]
    if len(cases) != 5:
        raise ValueError(f"Expected five passing selected-body clearance PI traces, got {len(cases)}")
    with np.load(ARRAYS) as saved:
        weight = saved["pi_weight"].astype(np.float64)
        pi_trajectory = saved["pi_trajectory"].astype(np.float64)
        v5_reference = saved["v5_reference"].astype(np.float64)
    with np.load(REFERENCE) as saved:
        names = saved["action_names"].tolist()
    indices = [names.index(name) for name in JOINTS]
    if weight.shape != (21, 2) or pi_trajectory.shape != (400, 2) or v5_reference.shape != (400, 2):
        raise ValueError("Controller arrays differ from the frozen 400-frame teacher contract")
    robot_model = mujoco.MjModel.from_xml_path(str(BODY))
    xml = BODY.read_text().replace("</worldbody>",
        '<body name="offline_toy"><freejoint name="offline_toy_free"/>'
        '<geom name="offline_toy_geom" type="box" size="0.01 0.01 0.01"/>'
        '</body></worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_center")
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in cases:
        trace_path = ROOT / row["trace"]
        with np.load(trace_path) as saved:
            qpos = saved["qpos"]
            phase = saved["phase"]
            dt = float(saved["step_dt_s"])
        reach_qpos = qpos[phase == "reach"]
        if reach_qpos.shape != (400, model.nq) or dt != .02:
            raise ValueError(f"Unexpected native trace shape or clock: {trace_path}")
        live_xy = np.empty((400, 2), dtype=np.float64)
        for step, q in enumerate(reach_qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            diff = q[robot_model.nq:robot_model.nq + 3] - data.site_xpos[hand_id]
            inv_root = q[3:7] * np.array([1, -1, -1, -1])
            local = np.empty(3)
            mujoco.mju_rotVecQuat(local, diff, inv_root)
            live_xy[step] = local[:2]
        entry_error = np.repeat(((live_xy[0] - v5_reference[0]) / XY_SCALE)[None, :], 400, axis=0)
        current_error = (live_xy - v5_reference) / XY_SCALE
        accumulated_error = np.cumsum(current_error * dt, axis=0)
        integral = np.zeros(21, dtype=np.float64)
        correction = np.zeros((400, 4), dtype=np.float64)
        for step in range(400):
            direction = weight @ (live_xy[step] - pi_trajectory[step])
            integral = np.clip(integral + 1.5 * dt * direction, -BOUND, BOUND)
            blend = smoothstep(step * dt / 4)
            residual = np.clip(.6 * direction + integral, -BOUND, BOUND) * blend
            correction[step] = residual[indices]
        previous = np.vstack((np.zeros((1, 4)), correction[:-1])) / BOUND
        blend = np.asarray([smoothstep(step * dt / 4) for step in range(400)])[:, None]
        x = np.column_stack((entry_error, current_error, accumulated_error, previous, blend))
        y = correction / BOUND
        if x.shape != (400, 11) or y.shape != (400, 4) or not np.isfinite(x).all():
            raise ValueError(f"Invalid recovered labels for {row['case']}")
        out = OUT / f"{row['case']}.npz"
        np.savez_compressed(out, features=x.astype(np.float32), labels=y.astype(np.float32),
                            live_xy=live_xy.astype(np.float32), teacher_correction=correction.astype(np.float32))
        entries.append({"case": row["case"], "seed": row["seed"],
                        "split": "validation" if row["seed"] == 9712 else "train",
                        "rail_v": row["rail_v"], "delay_ms": row["delay_ms"],
                        "toy_offset_b_m": row["toy_offset_b_m"],
                        "teacher_contact_s": row["continuous_finger_contact_s"],
                        "max_abs_teacher_correction": round(float(np.abs(correction).max()), 5),
                        "trace": row["trace"], "trace_sha256": sha(trace_path),
                        "dataset": str(out.relative_to(ROOT)), "dataset_sha256": sha(out)})
    manifest = {"schema": "robot-learning-lab.cove50-handoff-pi-reconstruction.v1",
                "body_sha256": sha(BODY), "controller_arrays_sha256": sha(ARRAYS),
                "reference_trace_sha256": sha(REFERENCE),
                "method": "Exact frozen PI recurrence on pre-action native qpos, projected to the v5 11-feature four-arm residual contract; no physics rerun.",
                "cases": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
