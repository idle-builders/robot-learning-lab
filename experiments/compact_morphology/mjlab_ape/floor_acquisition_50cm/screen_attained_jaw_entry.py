"""Screen jaw entry from an attained continuous Cove squat-to-reach state.

Only the right arm and jaw may change. This is static geometry triage; a
passing pose still requires a native motor-attained transition and possession.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[4]
CASES = ROOT / "artifacts/ape-cove50-squat-reach-handoff-20261010"
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
HAND = ("right_hand", "right_finger_tip")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def with_cube(path: Path) -> mujoco.MjModel:
    tree = ET.parse(path)
    world = tree.getroot().find("worldbody")
    cube = ET.SubElement(world, "body", name="screen_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="screen_cube_free")
    ET.SubElement(cube, "geom", name="screen_cube_geom", type="box", size="0.01 0.01 0.01",
                  mass="0.00625", friction="0.8 0.01 0.001")
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="candidate-continuous-2906-roll025")
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=5010)
    parser.add_argument("--penetration-weight", type=float, default=500.0)
    parser.add_argument("--time-s", type=float, default=None,
                        help="Attained source time; defaults to the saved closest jaw-contact time")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = next(r for r in json.loads((CASES / "jaw-entry-matrix.json").read_text())["cases"]
               if r["case"] == args.case)
    if row["body"] != "candidate":
        raise ValueError("This diagnostic uses the unselected narrow-bridge candidate")
    trace_path = ROOT / row["trace"]
    model = with_cube(BODY)
    data = mujoco.MjData(model)
    with np.load(trace_path) as trace:
        dt = float(trace["step_dt_s"])
        # The saved qpos is pre-action; index t/dt is the state after the
        # contact frame ending at t (the next control step's initial state).
        tick = round((row["time_s"] if args.time_s is None else args.time_s) / dt)
        if tick < 0 or tick >= len(trace["qpos"]):
            raise ValueError("Source time falls outside the retained trace")
        source = trace["qpos"][tick].astype(np.float64)
    if len(source) != model.nq or model.nu != 21:
        raise ValueError("Expected exact 21-axis robot plus one free cube")
    qadr = np.array([model.joint(n).qposadr[0] for n in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    initial = source[qadr].copy()
    lower = np.array([model.joint(n).range[0] for n in ARM]) + 1e-5
    upper = np.array([model.joint(n).range[1] for n in ARM]) - 1e-5
    cube_xyz = source[model.joint("screen_cube_free").qposadr[0]:][:3]
    ids = {name: model.geom(name).id for name in (*HAND, "screen_cube_geom")}

    def pose(arm: np.ndarray, jaw: float = 0.0) -> dict:
        data.qpos[:] = source
        data.qpos[qadr] = arm
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        palm = data.geom_xpos[ids["right_hand"]].copy()
        finger = data.geom_xpos[ids["right_finger_tip"]].copy()
        axis = (finger - palm) / np.linalg.norm(finger - palm)
        floor = {}
        for name in HAND:
            gid = ids[name]
            rot = data.geom_xmat[gid].reshape(3, 3)
            extent = sum(abs(rot[2, j]) * model.geom_size[gid, j] for j in range(3))
            floor[name] = float(data.geom_xpos[gid, 2] - extent)
        contact_names = []
        cube_names = set()
        bad = []
        robot_cube_depth = 0.0
        for k in range(data.ncon):
            c = data.contact[k]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if "screen_cube_geom" in (n1, n2):
                other = n2 if n1 == "screen_cube_geom" else n1
                cube_names.add(other)
                contact_names.append((other, round(float(c.dist) * 1000, 3)))
                if other != "floor":
                    robot_cube_depth = max(robot_cube_depth, -float(c.dist))
                if other not in (*HAND, "floor") and c.dist < -0.0005:
                    bad.append((other, round(float(c.dist) * 1000, 3)))
            elif "floor" in (n1, n2) and (n1.startswith("right_") or n2.startswith("right_")):
                other = n2 if n1 == "floor" else n1
                if other != "right_sole" and c.dist < -0.0005:
                    bad.append((other, round(float(c.dist) * 1000, 3)))
        return {"midpoint": (palm + finger) / 2, "axis": axis, "floor": floor,
                "cube_contacts": sorted(cube_names), "cube_contact_dist_mm": contact_names,
                "bad_contacts": bad, "robot_cube_depth_m": robot_cube_depth}

    rng = np.random.default_rng(args.seed)
    results = []
    for height_mm in (0, 6, 10, 12, 16, 18, 20, 24):
        target = cube_xyz + np.array([0, 0, height_mm / 1000])

        def residual(arm: np.ndarray) -> np.ndarray:
            s = pose(arm)
            return np.r_[100 * (s["midpoint"] - target),
                         100 * max(0, .001 - s["floor"]["right_hand"]),
                         100 * max(0, .001 - s["floor"]["right_finger_tip"]),
                         args.penetration_weight * max(0, s["robot_cube_depth_m"] - .0003),
                         .5 * s["axis"][2], .03 * (arm - initial)]

        for i in range(args.starts):
            guess = np.clip(initial if i == 0 else initial + rng.normal(0, .55, 4), lower, upper)
            fit = least_squares(residual, guess, bounds=(lower, upper), max_nfev=120)
            opened = pose(fit.x)
            closed = pose(fit.x, .5)
            both = all(n in closed["cube_contacts"] for n in HAND)
            row_out = {
                "height_target_mm": height_mm, "start": i,
                "arm_rad": [round(float(v), 6) for v in fit.x],
                "arm_delta_rad": [round(float(v), 5) for v in fit.x - initial],
                "objective": round(float(np.linalg.norm(residual(fit.x))), 6),
                "open_midpoint_error_mm": round(float(1000 * np.linalg.norm(opened["midpoint"] - target)), 3),
                "open_axis_z": round(float(opened["axis"][2]), 4),
                "open_floor_mm": {k: round(v * 1000, 3) for k, v in opened["floor"].items()},
                "open_cube_contacts": opened["cube_contacts"],
                "open_contact_dist_mm": opened["cube_contact_dist_mm"],
                "open_robot_cube_depth_mm": round(1000 * opened["robot_cube_depth_m"], 3),
                "closed_cube_contacts": closed["cube_contacts"],
                "closed_contact_dist_mm": closed["cube_contact_dist_mm"],
                "closed_bad_contacts": closed["bad_contacts"],
                "static_gate": bool(both and opened["robot_cube_depth_m"] <= .0005 and
                                    not opened["bad_contacts"] and not closed["bad_contacts"] and
                                    min(closed["floor"].values()) >= -0.0005 and
                                    opened["floor"]["right_hand"] >= -0.0005 and
                                    opened["floor"]["right_finger_tip"] >= -0.0005),
            }
            results.append(row_out)
    passing = [r for r in results if r["static_gate"]]
    ranked = sorted(results, key=lambda r: (not r["static_gate"], r["objective"]))
    best_arm = np.asarray(ranked[0]["arm_rad"])
    corridor = []
    for fraction in np.linspace(0, 1, 21):
        state = pose(initial + fraction * (best_arm - initial), 0.0)
        corridor.append({
            "fraction": round(float(fraction), 3),
            "cube_contacts": state["cube_contacts"],
            "cube_contact_dist_mm": state["cube_contact_dist_mm"],
            "bad_contacts": state["bad_contacts"],
            "minimum_jaw_floor_mm": round(1000 * min(state["floor"].values()), 3),
        })
    report = {
        "schema": "robot-learning-lab.cove50-attained-jaw-entry-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body": str(BODY.relative_to(ROOT)), "candidate_body_sha256": sha(BODY),
        "trace": row["trace"], "trace_sha256": sha(trace_path),
        "case": args.case, "source_tick": tick, "source_time_s": round(tick * dt, 3),
        "cube_xyz_m": cube_xyz.tolist(), "source_arm_rad": initial.tolist(),
        "starts_per_height": args.starts, "seed": args.seed,
        "penetration_weight": args.penetration_weight,
        "source_time_override_s": args.time_s,
        "attempted": len(results), "static_gate_count": len(passing),
        "best": ranked[0], "linear_arm_corridor_to_best": corridor, "results": results,
        "limits": ["The unselected candidate is a diagnostic body, not a replacement for the design pointer.",
                   "Root, legs, toy and other arm are frozen at one native motor-attained frame. Arm IK is not a motor-attained path.",
                   "Static closed contacts do not establish compressive force, possession, lift, balance or release."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("case", "attempted", "static_gate_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
