"""Search robot-to-cube placement for a static opposing-side grip."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import contacts, with_cube


OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/repositioned-side-grip-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
NEAR = ("right_palm_rail_frame_budget", "right_jaw_bridge_frame_budget",
        "right_jaw_rail_frame_budget", "right_palm_root_frame_budget",
        "right_jaw_root_frame_budget", "right_gripper_motor_case",
        "right_forearm_frame_budget")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    trace_path = ROOT / plan["native_source"]
    with np.load(trace_path) as trace:
        source = trace["qpos"][round(plan["source_time_s"] / float(trace["step_dt_s"]))].copy()
    rows = []
    for body_name, body_rel in plan["bodies"].items():
        body_path = ROOT / body_rel
        model = with_cube(body_path)
        data = mujoco.MjData(model)
        cube = model.geom("audit_cube_geom").id
        pads = (model.geom("right_hand").id, model.geom("right_finger_tip").id)
        near = [model.geom(name).id for name in NEAR]
        arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
        jaw_adr = model.joint("right_gripper").qposadr[0]
        original = source[arm_adr].copy()
        envelope = plan["arm_envelope_rad"]
        low = np.maximum([model.joint(name).range[0] for name in ARM], original-envelope)
        high = np.minimum([model.joint(name).range[1] for name in ARM], original+envelope)
        cube_xy = source[-7:-5].copy()
        limit = plan["cube_xy_offset_mm"] / 1000
        bounds = list(zip(low, high)) + [(-limit, limit), (-limit, limit)]

        def assess(values: np.ndarray, yaw_deg: float) -> dict:
            data.qpos[:] = source
            data.qpos[arm_adr] = values[:4]
            data.qpos[jaw_adr] = plan["jaw_rad"]
            data.qpos[-7:-5] = cube_xy + values[4:6]
            yaw = math.radians(yaw_deg)
            data.qpos[-4:] = (math.cos(yaw/2), 0, 0, math.sin(yaw/2))
            mujoco.mj_forward(model, data)
            center = data.geom_xpos[cube]
            to_local = data.geom_xmat[cube].reshape(3, 3).T
            pad_rows = []
            for geom in pads:
                nearest = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, data, cube, geom, .05, nearest)
                pad_rows.append({"signed_distance_mm": float(distance * 1000),
                                 "cube_point_mm": ((to_local @ (nearest[:3]-center))*1000).tolist()})
            other = min(float(mujoco.mj_geomDistance(model, data, cube, geom,
                                                     .05, np.zeros(6))) * 1000
                        for geom in near)
            return {"pads": pad_rows, "nonpad_clearance_mm": other,
                    "contacts": contacts(model, data)}

        def cost(values: np.ndarray, yaw_deg: float) -> float:
            state = assess(values, yaw_deg)
            residual = []
            for pad, x_side in zip(state["pads"], (10., -10.)):
                x, y, z = pad["cube_point_mm"]
                residual.extend(((pad["signed_distance_mm"]+.35)*2,
                                 (x-x_side)*.5, y*.5,
                                 max(0., z-5.)*1.5,
                                 max(0., -z-6.)*1.5,
                                 max(0., -1.-pad["signed_distance_mm"])*2))
            residual.extend((max(0., 1.-state["nonpad_clearance_mm"])*2,
                             np.linalg.norm(values[:4]-original)*.1))
            return float(np.linalg.norm(residual))

        for yaw in plan["cube_yaw_deg"]:
            fit = differential_evolution(lambda x: cost(x, yaw), bounds=bounds,
                                         seed=plan["optimizer_seed"] + round(yaw) +
                                         (100 if body_name != "narrow_bridge" else 0),
                                         maxiter=plan["global_maxiter"],
                                         popsize=plan["global_popsize"],
                                         tol=.001, polish=False)
            state = assess(fit.x, yaw)
            side = all(abs(pad["cube_point_mm"][1]) <= 8.
                       and -6. <= pad["cube_point_mm"][2] <= 5.
                       for pad in state["pads"])
            pad_ok = all(-1. <= pad["signed_distance_mm"] <= -.1
                         for pad in state["pads"])
            clean = (state["contacts"]["two_sole_contact"] and
                     not state["contacts"]["nonsole_floor_or_self_penetration_mm"])
            passed = side and pad_ok and clean and state["nonpad_clearance_mm"] >= 1.
            rows.append({"case": f"{body_name}-yaw{yaw}", "body": body_name,
                         "cube_yaw_deg": yaw,
                         "cube_xy_offset_mm": (fit.x[4:6]*1000).tolist(),
                         "arm_rad": fit.x[:4].tolist(), "objective": float(fit.fun),
                         "face_contact_pass": passed, "side_ok": side,
                         "pad_overlap_ok": pad_ok, "clean_robot": clean,
                         **state})
            print(f"{body_name} yaw={yaw} face={passed} cost={fit.fun:.3f}", flush=True)
    report = {"schema": "robot-learning-lab.cove50-repositioned-side-grip.v1",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "source_trace_sha256": sha(trace_path),
              "body_sha256": {name: sha(ROOT / path) for name, path in plan["bodies"].items()},
              "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__, "cases": rows,
              "passes": sum(row["face_contact_pass"] for row in rows),
              "limit": "Optimistic static final-jaw search allowing a different cube XY/yaw at fixed native robot root/legs. No feasible approach path, powered control, actual cube reset, force, retention or robustness is established."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    (OUT / "matrix.json").write_text(json.dumps({"cases": rows}, indent=2)+"\n")
    print(json.dumps({"passes": report["passes"], "cases": len(rows)}))


if __name__ == "__main__":
    main()
