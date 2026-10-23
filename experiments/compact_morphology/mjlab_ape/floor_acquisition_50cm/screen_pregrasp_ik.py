"""Screen selected-hand pregrasp poses from an actual Cove reach contact state.

Only the four right-arm joints and the one gripper joint can change. Legs,
root, left arm and the 20 mm toy remain at the saved native replay state. This
is a kinematic screen, not a continuous path, native motor replay or grasp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
EVAL = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/evaluation"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/pregrasp-ik-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll",
       "right_shoulder_pitch", "right_elbow_pitch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="step-feedback")
    parser.add_argument("--starts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=29017)
    parser.add_argument("--floor-weight", type=float, default=100.0)
    parser.add_argument("--axis-weight", type=float, default=0.8)
    parser.add_argument("--global-starts", action="store_true",
                        help="Sample the full legal arm joint box after the first attained pose")
    args = parser.parse_args()
    trace_path = EVAL / f"{args.case}.npz"
    with np.load(trace_path) as trace:
        contact_ticks = np.flatnonzero(trace["finger_toy_contact"])
        if not len(contact_ticks):
            raise ValueError("Reach source has no actual finger/toy contact")
        source_tick = int(contact_ticks[-1])
        native_qpos = trace["qpos"][source_tick].astype(np.float64)
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nu != 21 or len(native_qpos) != model.nq + 7:
        raise ValueError("Expected selected 21-axis body and one free toy")
    data = mujoco.MjData(model)
    qpos = native_qpos[:model.nq].copy()
    toy = native_qpos[model.nq:model.nq + 3].copy()
    arm_qadr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_qadr = model.joint("right_gripper").qposadr[0]
    lower = np.array([model.joint(name).range[0] for name in ARM])
    upper = np.array([model.joint(name).range[1] for name in ARM])
    start_arm = qpos[arm_qadr].copy()
    palm_id = model.geom("right_hand").id
    finger_id = model.geom("right_finger_tip").id

    def state(arm: np.ndarray) -> dict:
        data.qpos[:] = qpos
        data.qpos[arm_qadr] = arm
        data.qpos[jaw_qadr] = 0.0  # actual selected-hand open joint limit
        mujoco.mj_forward(model, data)
        palm = data.geom_xpos[palm_id].copy()
        finger = data.geom_xpos[finger_id].copy()
        midpoint = (palm + finger) / 2
        axis = (finger - palm) / np.linalg.norm(finger - palm)
        clearance = []
        for geom_id in (palm_id, finger_id):
            rot = data.geom_xmat[geom_id].reshape(3, 3)
            half_z = sum(abs(rot[2, j]) * model.geom_size[geom_id, j]
                         for j in range(3))
            clearance.append(float(data.geom_xpos[geom_id, 2] - half_z))
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            if palm_id not in (contact.geom1, contact.geom2) and finger_id not in (contact.geom1, contact.geom2):
                continue
            contacts.append({
                "a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                "b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
                "distance_m": round(float(contact.dist), 6),
            })
        return {"midpoint": midpoint, "axis": axis,
                "palm_floor_clearance_m": clearance[0],
                "finger_floor_clearance_m": clearance[1],
                "contacts": contacts}

    def residual(arm: np.ndarray) -> np.ndarray:
        geometry = state(arm)
        return np.concatenate((
            100 * (geometry["midpoint"] - toy),
            [args.floor_weight * max(0.0, 0.001 - geometry["palm_floor_clearance_m"]),
             args.floor_weight * max(0.0, 0.001 - geometry["finger_floor_clearance_m"]),
             args.axis_weight * geometry["axis"][2]],
            0.03 * (arm - start_arm),
        ))

    rng = np.random.default_rng(args.seed)
    results = []
    for index in range(args.starts):
        guess = (np.clip(start_arm, lower + 1e-5, upper - 1e-5) if index == 0 else
                 rng.uniform(lower + 1e-5, upper - 1e-5) if args.global_starts else
                 np.clip(start_arm + rng.normal(0, 0.45, len(ARM)),
                         lower + 1e-5, upper - 1e-5))
        fit = least_squares(residual, guess, bounds=(lower, upper),
                            max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        geometry = state(fit.x)
        results.append({
            "start": index, "objective": float(np.linalg.norm(residual(fit.x))),
            "arm_rad": [round(float(v), 6) for v in fit.x],
            "max_arm_delta_rad": round(float(np.max(np.abs(fit.x - start_arm))), 5),
            "midpoint_error_mm": round(float(1000 * np.linalg.norm(geometry["midpoint"] - toy)), 3),
            "jaw_axis_vertical_component": round(float(geometry["axis"][2]), 4),
            "palm_floor_clearance_mm": round(1000 * geometry["palm_floor_clearance_m"], 3),
            "finger_floor_clearance_mm": round(1000 * geometry["finger_floor_clearance_m"], 3),
            "hand_contacts": geometry["contacts"],
        })
    results.sort(key=lambda item: item["objective"])
    best = results[0]
    candidate_qpos = qpos.copy()
    candidate_qpos[arm_qadr] = best["arm_rad"]
    candidate_qpos[jaw_qadr] = 0.0
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ("-global" if args.global_starts else "") + (
        "" if args.floor_weight == 100 and args.axis_weight == 0.8 else "-unconstrained")
    np.savez_compressed(OUT / f"{args.case}{suffix}.npz", source_qpos=qpos,
                        candidate_qpos=candidate_qpos, toy_xyz_m=toy,
                        arm_names=np.asarray(ARM), source_tick=source_tick)
    report = {
        "schema": "robot-learning-lab.cove50-pregrasp-ik-screen.v1",
        "method": "selected-body static arm IK from actual saved native finger-contact state",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(BODY), "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha(trace_path), "case": args.case,
        "source_contact_tick": source_tick, "source_toy_xyz_m": toy.tolist(),
        "source_arm_rad": start_arm.tolist(), "starts": args.starts,
        "seed": args.seed, "global_starts": args.global_starts,
        "floor_weight": args.floor_weight,
        "axis_weight": args.axis_weight,
        "best": best, "solutions": results,
        "screen_pass": (best["midpoint_error_mm"] <= 3 and
                        best["palm_floor_clearance_mm"] >= 0 and
                        best["finger_floor_clearance_mm"] >= 0 and
                        not any(contact["distance_m"] < -1e-4
                                for contact in best["hand_contacts"])),
        "limits": [
            "The body root, legs and other arm are frozen at an attained replay state; this does not prove balance during transition.",
            "No movable toy contact, pinch force, motor effort, continuous arm path, object lift or hold is simulated by IK.",
            "The selected fully closed jaw remains 0.135 mm wider than the 20 mm toy edge in the ideal aligned face calculation.",
        ],
    }
    (OUT / f"{args.case}{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": args.case, "screen_pass": report["screen_pass"],
                      "best": best}, indent=2))


if __name__ == "__main__":
    main()
