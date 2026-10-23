"""Score native gait and contact-timed landing traces on the selected body."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-swing-landing-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CASES = ("selected", "inward05", "inward10")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def com_offset(model: mujoco.MjModel, data: mujoco.MjData, trace, tick: int,
               world: int, sole_id: int) -> float:
    data.qpos[:] = model.qpos0
    data.qpos[:7] = trace["root_pose"][tick, world]
    data.qpos[7:17] = trace["joint_pos"][tick, world]
    mujoco.mj_forward(model, data)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    lateral = rotation.reshape(3, 3)[:, 1]
    return abs(float((data.subtree_com[0] - data.geom_xpos[sole_id]) @ lateral))


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    if sha(BODY) != plan["identity"][str(BODY.relative_to(ROOT))]:
        raise ValueError("Selected body changed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    data = mujoco.MjData(model)
    soles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
             for side in ("left", "right")]
    rows = []
    for case in CASES:
        trace_path = OUT / f"{case}.npz"
        report_path = OUT / f"{case}.json"
        case_plan = json.loads((OUT / f"{case}-plan.json").read_text())
        if sha(BODY) != case_plan["sources"]["body"]:
            raise ValueError(f"Wrong body in {case}")
        trace = np.load(trace_path)
        report = json.loads(report_path.read_text())
        active = trace["active"]
        found = trace["sole_found"] > 0
        torque = abs(trace["actuator_force"])
        rated = trace["rated_effort_limits_nm"]
        leg_rated = (torque >= .995 * rated[None, None, :]).any(axis=2)
        dt = float(trace["step_dt_s"])
        support = []
        for side in (0, 1):
            exclusive = active & found[:, :, side] & ~found[:, :, 1-side]
            entry = exclusive & ~np.vstack((np.zeros((1, active.shape[1]), bool), exclusive[:-1]))
            offsets = [com_offset(model, data, trace, int(t), int(w), soles[side])
                       for t, w in zip(*np.where(entry))]
            support.append({
                "side": ("left", "right")[side],
                "exclusive_fraction": round(float(exclusive.sum() / active.sum()), 6),
                "entries": len(offsets),
                "mean_abs_com_offset_at_entry_cm": round(100 * float(np.mean(offsets)), 3)
                if offsets else None,
                "sole_half_width_cm": round(100 * float(model.geom_size[soles[side], 1]), 3),
                "hip_roll_rated_fraction_exclusive": round(float(np.mean(
                    (torque[:, :, side*5+1] >= .995 * rated[side*5+1])[exclusive])), 6)
                if exclusive.any() else None,
                "ankle_rated_fraction_exclusive": round(float(np.mean(
                    (torque[:, :, side*5+4] >= .995 * rated[side*5+4])[exclusive])), 6)
                if exclusive.any() else None,
            })
        worlds = report["worlds"]
        rows.append({
            "case": case, "offset_rad": case_plan["offset_rad"],
            "survived_8s": sum(w["duration_s"] >= 8 for w in worlds),
            "one_meter_alternating": report["one_meter_with_alternating_steps"],
            "mean_forward_m": round(float(np.mean([w["forward_m"] for w in worlds])), 4),
            "mean_abs_lateral_m": round(float(np.mean([abs(w["lateral_m"]) for w in worlds])), 4),
            "moving_rated_leg_fraction": round(float(np.mean(leg_rated[active])), 6),
            "left_hip_roll_rated_fraction": round(float(np.mean(
                (torque[:, :, 1] >= .995 * rated[1])[active])), 6),
            "right_hip_roll_rated_fraction": round(float(np.mean(
                (torque[:, :, 6] >= .995 * rated[6])[active])), 6),
            "total_sole_slip_m": round(float(sum(
                (trace["sole_speed"][:, :, side] * found[:, :, side] * active).sum() * dt / len(worlds)
                for side in (0, 1))), 4),
            "max_hand_force_n": round(float(trace["hand_force"][active].max()), 4),
            "body_contact_frames": int(trace["body_contact"][active].sum()),
            "left_com_entry_cm": support[0]["mean_abs_com_offset_at_entry_cm"],
            "right_com_entry_cm": support[1]["mean_abs_com_offset_at_entry_cm"],
            "left_hip_roll_rated_exclusive": support[0]["hip_roll_rated_fraction_exclusive"],
            "right_hip_roll_rated_exclusive": support[1]["hip_roll_rated_fraction_exclusive"],
            "support": support,
            "report": str(report_path.relative_to(ROOT)),
            "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": sha(trace_path),
            "clip": str((OUT / f"{case}.mp4").relative_to(ROOT)),
            "initial_state_sha256": report["initial_state_sha256"],
            "first_action_sha256": report["first_action_sha256"],
        })
    baseline = rows[0]
    gates = []
    for row in rows[1:]:
        entry_improved = all(row["support"][i]["mean_abs_com_offset_at_entry_cm"] is not None and
                             row["support"][i]["mean_abs_com_offset_at_entry_cm"] <
                             baseline["support"][i]["mean_abs_com_offset_at_entry_cm"]
                             for i in (0, 1))
        pass_all = (row["survived_8s"] == 4 and row["one_meter_alternating"] == 4 and
                    row["max_hand_force_n"] == 0 and row["body_contact_frames"] == 0 and
                    row["total_sole_slip_m"] <= 1.2 * baseline["total_sole_slip_m"] and
                    entry_improved and row["moving_rated_leg_fraction"] < .70)
        gates.append({"case": row["case"], "entry_improved_both_sides": entry_improved,
                      "nominal_screen_pass": pass_all})
    matrix = {"schema": "robot-learning-lab.cove50-swing-landing-matrix.v1",
              "definition": "Fresh 4-world native 8 s replays; 50 Hz rated-leg fraction, sole-slip path and exclusive-support COM at first contact-transition frame. COM uses saved root and ten leg joints, XML-default upper joints.",
              "cases": rows}
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    result = {"schema": "robot-learning-lab.cove50-swing-landing-screen.v1",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "selected_body_sha256": sha(BODY), "matrix": str((OUT / "matrix.json").relative_to(ROOT)),
              "nominal_gates": gates, "held_out_stress": "eligible" if any(g["nominal_screen_pass"] for g in gates)
              else "skipped_by_nominal_gate",
              "decision": "reject_inward_landing_shim" if not any(g["nominal_screen_pass"] for g in gates)
              else "advance_to_reserved_stress",
              "interpretation": "Both inward offsets move COM closer to the arriving sole and lower sole slip, but neither reduces moving rated-leg time below the predeclared 70% screen; both increase absolute lateral drift. Right exclusive-support hip-roll load remains high. No actor retraining follows a rejected nominal diagnostic.",
              "qualification": "Replay-only authored contact shim; no learned checkpoint or selected-design change."}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["case"], row["survived_8s"], row["one_meter_alternating"],
              row["moving_rated_leg_fraction"], row["total_sole_slip_m"],
              [s["mean_abs_com_offset_at_entry_cm"] for s in row["support"]])
    print(gates)


if __name__ == "__main__":
    main()
