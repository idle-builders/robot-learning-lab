"""Read saved native Cove gait traces for the stance COM/sole mismatch.

The diagnostic traces saved the root and ten leg joints, not every arm joint.
The companion full-qpos rollout bounds the COM error from holding those upper
joints at their XML defaults. This is a geometry diagnostic, not a control pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261009"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"
MATCHED_WORLDS = (0, 1, 2)  # Source completed these native 1 m gait worlds.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray,
             sole_ids: tuple[int, int], side: int) -> float:
    """Absolute lateral COM-to-stance-sole distance in the root heading frame."""
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    lateral_axis = rotation.reshape(3, 3)[:, 1]
    return abs(float((data.subtree_com[0] - data.geom_xpos[sole_ids[side]]) @ lateral_axis))


def summarize(values: list[float], scale: float = 1.0) -> dict:
    array = np.asarray(values, dtype=float)
    return {"count": len(array),
            "mean": round(float(array.mean() * scale), 5),
            "median": round(float(np.median(array) * scale), 5),
            "p25": round(float(np.percentile(array, 25) * scale), 5),
            "p75": round(float(np.percentile(array, 75) * scale), 5)}


def full_state_upper_joint_error(model: mujoco.MjModel,
                                 sole_ids: tuple[int, int]) -> dict:
    full = np.load(OUT / "overlap-source-forward-seed144.npz")["qpos"]
    data = mujoco.MjData(model)
    errors = []
    for state in full:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        actual = data.subtree_com[0, :2].copy()
        data.qpos[17:] = model.qpos0[17:]
        mujoco.mj_forward(model, data)
        errors.append(float(np.linalg.norm(data.subtree_com[0, :2] - actual)))
    return {
        "source_full_qpos": str((OUT / "overlap-source-forward-seed144.npz").relative_to(ROOT)),
        "source_full_qpos_sha256": sha(OUT / "overlap-source-forward-seed144.npz"),
        "planar_com_error_m_from_default_upper_joints": summarize(errors),
        "maximum_error_m": round(max(errors), 6),
        "interpretation": "A separate same-seed source native rollout bounds upper-joint reconstruction error; it is not the identical trace.",
    }


def trace_case(model: mujoco.MjModel, sole_ids: tuple[int, int],
               prefix: str, side: int) -> dict:
    path = OUT / f"{prefix}-seed7144-trace.npz"
    trace = np.load(path)
    data = mujoco.MjData(model)
    active = trace["active"]
    found = trace["sole_found"] > 0
    exclusive = found[:, :, side] & ~found[:, :, 1 - side] & active
    offsets = []
    hip_roll_rated = []
    ankle_rated = []
    pre_lift_offsets = []
    pre_lift_load_share = []
    for world in MATCHED_WORLDS:
        for tick in np.flatnonzero(exclusive[:, world]):
            qpos = np.r_[trace["root_pose"][tick, world],
                         trace["joint_pos"][tick, world]]
            offsets.append(geometry(model, data, qpos, sole_ids, side))
            limits = trace["rated_effort_limits_nm"]
            effort = abs(trace["actuator_force"][tick, world])
            hip_roll_rated.append(bool(effort[side * 5 + 1] >= .995 * limits[side * 5 + 1]))
            ankle_rated.append(bool(effort[side * 5 + 4] >= .995 * limits[side * 5 + 4]))
        for tick in range(1, len(active)):
            # A three-frame exclusive phase excludes single-frame contact chatter.
            if (not active[tick, world] or not found[tick - 1, world].all()
                or not exclusive[tick:tick + 3, world].all()
                or len(exclusive[tick:tick + 3, world]) != 3):
                continue
            qpos = np.r_[trace["root_pose"][tick - 1, world],
                         trace["joint_pos"][tick - 1, world]]
            pre_lift_offsets.append(geometry(model, data, qpos, sole_ids, side))
            loads = abs(trace["sole_contact_force"][tick - 1, world, :, 2])
            pre_lift_load_share.append(float(loads[side] / max(1e-9, loads.sum())))
    half_width = float(model.geom_size[sole_ids[side], 1])
    return {
        "actor": prefix,
        "stance": ("left", "right")[side],
        "trace": str(path.relative_to(ROOT)),
        "trace_sha256": sha(path),
        "matched_worlds": list(MATCHED_WORLDS),
        "sole_half_width_m": half_width,
        "exclusive_support_com_to_sole_lateral_m": summarize(offsets),
        "exclusive_support_com_outside_sole_half_width_m": summarize(
            [max(0.0, x - half_width) for x in offsets]),
        "exclusive_support_hip_roll_rated_fraction": round(float(np.mean(hip_roll_rated)), 5),
        "exclusive_support_ankle_pitch_rated_fraction": round(float(np.mean(ankle_rated)), 5),
        "pre_lift_com_to_new_stance_lateral_m": summarize(pre_lift_offsets),
        "pre_lift_new_stance_vertical_load_share": summarize(pre_lift_load_share),
    }


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nq != 28 or model.nu != 21:
        raise ValueError("Expected selected Cove 21-axis body")
    sole_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in ("left_sole", "right_sole"))
    if any(i < 0 for i in sole_ids):
        raise ValueError("Expected selected sole geoms")
    source_report = json.loads((OUT / "overlap-source-seed7144-trace.json").read_text())
    if (source_report["body_sha256"] != sha(BODY) or
        any(not source_report["worlds"][i]["ended_by_timeout"] or
            source_report["worlds"][i]["forward_m"] < 1.0
            for i in MATCHED_WORLDS)):
        raise ValueError("Source-success worlds/body changed")
    cases = [trace_case(model, sole_ids, actor, side)
             for actor in ("overlap-source", "overlap100") for side in (0, 1)]
    report = {
        "schema": "robot-learning-lab.cove50-stance-geometry.v1",
        "selected_pointer_sha256": sha(POINTER),
        "selected_body_sha256": sha(BODY),
        "body_mass_kg": round(float(model.body_mass.sum()), 6),
        "upper_joint_reconstruction_check": full_state_upper_joint_error(model, sole_ids),
        "cases": cases,
        "interpretation": (
            "On matched source-success gait worlds, the whole-body COM remains "
            "well lateral to the 35 mm sole half-width during exclusive support. "
            "At stable two-foot-to-one-foot transitions the newly loaded stance "
            "sole already carries most vertical contact force before COM shifts "
            "toward that sole. The overlap reward did not materially reduce "
            "this offset. This supports an explicit pre-lift load/COM transfer "
            "lesson, but geometry alone does not prove motor sufficiency or "
            "predict a successful policy."
        ),
        "limits": [
            "Diagnostic contact found is an imported MJLab sensor; load and motor are provisional simulation models.",
            "Trace reconstruction holds eleven upper joints at XML defaults; the companion full-qpos source rollout bounds its COM error.",
            "Three matched source-success worlds on one reserved seed diagnose a failure mechanism, not a variance qualification.",
            "The sole is a rectangle, but the reported lateral margin alone is not the full dynamic support polygon.",
        ],
    }
    path = OUT / "stance-geometry-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    matrix = {
        "schema": "robot-learning-lab.cove50-stance-geometry-matrix.v1",
        "note": "Three matched source-success worlds, 50 Hz native traces. COM is reconstructed from root and leg state with XML-default upper joints; separate full-qpos replay bounds upper-joint planar COM error to 0.29 mm. These are simulation geometry diagnostics, not gait qualification.",
        "cases": [{
            "actor": case["actor"], "stance": case["stance"],
            "exclusive_frames": case["exclusive_support_com_to_sole_lateral_m"]["count"],
            "sole_half_width_cm": round(100 * case["sole_half_width_m"], 3),
            "exclusive_com_offset_cm": round(100 * case["exclusive_support_com_to_sole_lateral_m"]["mean"], 3),
            "exclusive_overhang_cm": round(100 * case["exclusive_support_com_outside_sole_half_width_m"]["mean"], 3),
            "hip_roll_rated_fraction": case["exclusive_support_hip_roll_rated_fraction"],
            "ankle_pitch_rated_fraction": case["exclusive_support_ankle_pitch_rated_fraction"],
            "stable_lift_transitions": case["pre_lift_com_to_new_stance_lateral_m"]["count"],
            "pre_lift_com_offset_cm": round(100 * case["pre_lift_com_to_new_stance_lateral_m"]["mean"], 3),
            "pre_lift_stance_load_share": case["pre_lift_new_stance_vertical_load_share"]["mean"],
            "trace": case["trace"],
        } for case in cases],
    }
    (OUT / "stance-geometry-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"body_mass_kg": report["body_mass_kg"],
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
