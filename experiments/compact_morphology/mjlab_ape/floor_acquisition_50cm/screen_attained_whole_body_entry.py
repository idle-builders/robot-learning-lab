"""Bounded sole-constrained static jaw-entry screen from an attained Cove state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from screen_attained_jaw_entry import BODY, CASES, ROOT, HAND, with_cube


LEGS = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in
             ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"))
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
JOINTS = LEGS + ARM


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="candidate-continuous-2906-roll025")
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5011)
    parser.add_argument("--cube-clearance-scale-mm", type=float, default=1.0,
                        help="Smaller values penalize open-hand cube penetration more strongly")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = next(r for r in json.loads((CASES / "jaw-entry-matrix.json").read_text())["cases"]
               if r["case"] == args.case)
    if row["body"] != "candidate":
        raise ValueError("Expected the unselected narrow-bridge diagnostic candidate")
    trace_path = ROOT / row["trace"]
    model = with_cube(BODY)
    data = mujoco.MjData(model)
    with np.load(trace_path) as trace:
        dt = float(trace["step_dt_s"])
        tick = round((row["time_s"] if args.time_s is None else args.time_s) / dt)
        source = trace["qpos"][tick].astype(np.float64)
    if model.nu != 21 or len(source) != model.nq:
        raise ValueError("Expected 21 axes plus one free cube")
    source_rot = Rotation.from_quat(source[3:7], scalar_first=True)
    joint_adr = np.array([model.joint(n).qposadr[0] for n in JOINTS])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    joint_lower = np.array([model.joint(n).range[0] for n in JOINTS]) + 1e-5
    joint_upper = np.array([model.joint(n).range[1] for n in JOINTS]) - 1e-5
    sole_ids = np.array([model.site(f"{side}_sole_center").id for side in ("left", "right")])
    jaw_ids = {n: model.geom(n).id for n in HAND}
    cube_xyz = source[model.joint("screen_cube_free").qposadr[0]:][:3]
    data.qpos[:] = source
    mujoco.mj_forward(model, data)
    sole_origin = data.site_xpos[sole_ids].copy()
    normal_origin = data.site_xmat[sole_ids].reshape(2, 3, 3)[:, :, 2].copy()
    start_joints = np.clip(source[joint_adr], joint_lower, joint_upper)
    start = np.r_[source[:3], np.zeros(3), start_joints]
    lower = np.r_[source[:3] - [.03, .03, .04], [-.20, -.20, -.15], joint_lower]
    upper = np.r_[source[:3] + [.03, .03, .025], [.20, .20, .15], joint_upper]

    def state(x: np.ndarray, jaw: float = 0.0) -> dict:
        data.qpos[:] = source
        data.qpos[:3] = x[:3]
        data.qpos[3:7] = (source_rot * Rotation.from_rotvec(x[3:6])).as_quat(scalar_first=True)
        data.qpos[joint_adr] = x[6:]
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        p, f = (data.geom_xpos[jaw_ids[n]].copy() for n in HAND)
        axis = (f - p) / np.linalg.norm(f - p)
        floor = []
        for n in HAND:
            gid = jaw_ids[n]
            rot = data.geom_xmat[gid].reshape(3, 3)
            floor.append(float(data.geom_xpos[gid, 2] - np.abs(rot[2]) @ model.geom_size[gid]))
        cube_contacts = set()
        cube_depth = 0.0
        bad = []
        for index in range(data.ncon):
            c = data.contact[index]
            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if "screen_cube_geom" in (a, b):
                other = b if a == "screen_cube_geom" else a
                cube_contacts.add(other)
                if other != "floor":
                    cube_depth = max(cube_depth, -float(c.dist))
                if other not in (*HAND, "floor") and c.dist < -.0005:
                    bad.append((other, round(-1000 * float(c.dist), 3)))
            elif "floor" in (a, b):
                other = b if a == "floor" else a
                if other not in ("left_sole", "right_sole") and c.dist < -.0005:
                    bad.append((other, round(-1000 * float(c.dist), 3)))
        sole = data.site_xpos[sole_ids].copy()
        normal = data.site_xmat[sole_ids].reshape(2, 3, 3)[:, :, 2].copy()
        return {"midpoint": (p + f) / 2, "axis": axis, "floor": np.array(floor),
                "cube_contacts": sorted(cube_contacts), "cube_depth_m": cube_depth,
                "bad_contacts": bad, "sole": sole, "normal": normal}

    rng = np.random.default_rng(args.seed)
    results = []
    for height in (12, 16, 20, 24):
        target = cube_xyz + np.array([0, 0, height / 1000])

        def residual(x: np.ndarray) -> np.ndarray:
            s = state(x)
            return np.r_[
                (s["midpoint"] - target) / .003,
                (s["sole"] - sole_origin).ravel() / .002,
                (s["normal"] - normal_origin).ravel() / .04,
                np.minimum(s["floor"] - .001, 0) / .001,
                max(0, s["cube_depth_m"] - .0003) / (args.cube_clearance_scale_mm / 1000),
                s["axis"][2] / .2,
                (x[:3] - source[:3]) / [.04, .04, .06],
                x[3:6] / .4,
                (x[6:] - start_joints) / 1.5,
            ]

        for index in range(args.starts):
            guess = start.copy()
            if index:
                guess[:3] += rng.normal(0, [.008, .008, .012])
                guess[3:6] += rng.normal(0, [.06, .06, .04])
                guess[6:] += rng.normal(0, .22, len(JOINTS))
            guess = np.clip(guess, lower + 1e-5, upper - 1e-5)
            fit = least_squares(residual, guess, bounds=(lower, upper), max_nfev=100)
            opened = state(fit.x)
            closed = state(fit.x, .5)
            sole_drift = np.linalg.norm(opened["sole"] - sole_origin, axis=1) * 1000
            normal_deg = np.rad2deg(np.arccos(np.clip(np.sum(opened["normal"] * normal_origin, axis=1), -1, 1)))
            both = all(n in closed["cube_contacts"] for n in HAND)
            passed = bool(both and opened["cube_depth_m"] <= .0005 and
                          max(sole_drift) <= 2 and max(normal_deg) <= 3 and
                          min(opened["floor"]) >= -.0005 and min(closed["floor"]) >= -.0005 and
                          not opened["bad_contacts"] and not closed["bad_contacts"])
            results.append({
                "height_target_mm": height, "start": index,
                "objective": round(float(np.linalg.norm(residual(fit.x))), 5),
                "jaw_midpoint_error_mm": round(float(np.linalg.norm(opened["midpoint"] - target) * 1000), 3),
                "open_cube_depth_mm": round(opened["cube_depth_m"] * 1000, 3),
                "open_contacts": opened["cube_contacts"], "closed_contacts": closed["cube_contacts"],
                "sole_drift_mm": np.round(sole_drift, 3).tolist(),
                "sole_normal_error_deg": np.round(normal_deg, 3).tolist(),
                "jaw_floor_mm": np.round(opened["floor"] * 1000, 3).tolist(),
                "open_bad_contacts": opened["bad_contacts"], "closed_bad_contacts": closed["bad_contacts"],
                "root_delta_xyz_mm": np.round((fit.x[:3] - source[:3]) * 1000, 3).tolist(),
                "root_rotation_delta_rad": np.round(fit.x[3:6], 5).tolist(),
                "joint_rad": {n: round(float(v), 5) for n, v in zip(JOINTS, fit.x[6:])},
                "static_gate": passed,
            })
    ranked = sorted(results, key=lambda r: (not r["static_gate"], r["objective"]))
    report = {
        "schema": "robot-learning-lab.cove50-attained-whole-body-entry.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY), "trace_sha256": sha(trace_path),
        "case": args.case, "source_time_s": round(tick * dt, 3),
        "source_cube_xyz_m": cube_xyz.tolist(), "starts_per_height": args.starts,
        "seed": args.seed, "cube_clearance_scale_mm": args.cube_clearance_scale_mm,
        "attempted": len(results),
        "static_gate_count": sum(r["static_gate"] for r in results),
        "best": ranked[0], "results": results,
        "limits": ["Static sole-constrained geometry only; no motor-attained transition, balance or possession.",
                   "The narrow-bridge body is unselected; source material and contact properties remain unqualified."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("case", "attempted", "static_gate_count", "best")}, indent=2))


if __name__ == "__main__":
    main()
