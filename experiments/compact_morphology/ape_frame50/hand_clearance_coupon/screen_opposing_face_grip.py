"""Check whether the attained open grip can reach opposing cube side faces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import differential_evolution, least_squares


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import contacts, with_cube


OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/opposing-face-screen-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
NEAR_CUBE = ("right_palm_rail_frame_budget", "right_jaw_bridge_frame_budget",
             "right_jaw_rail_frame_budget", "right_palm_root_frame_budget",
             "right_jaw_root_frame_budget", "right_gripper_motor_case",
             "right_forearm_frame_budget")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUT / "plan.json")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    body_path = ROOT / plan["body"]
    trace_path = ROOT / plan["native_source"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    with np.load(trace_path) as trace:
        source = trace["qpos"][round(plan["source_time_s"] / float(trace["step_dt_s"]))].copy()
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube = model.geom("audit_cube_geom").id
    pads = (model.geom("right_hand").id, model.geom("right_finger_tip").id)
    nearby = [model.geom(name).id for name in NEAR_CUBE]
    original = source[arm_adr].copy()
    envelope = float(plan["arm_envelope_rad"])
    low = np.maximum([model.joint(name).range[0] for name in ARM], original - envelope)
    high = np.minimum([model.joint(name).range[1] for name in ARM], original + envelope)
    rng = np.random.default_rng(plan["optimizer_seed"])
    prior = json.loads((ROOT / plan["prior_waypoints"]).read_text())["waypoints"]

    def assess(arm: np.ndarray, jaw: float) -> dict:
        data.qpos[:] = source
        data.qpos[arm_adr] = arm
        data.qpos[jaw_adr] = jaw
        mujoco.mj_forward(model, data)
        center = data.geom_xpos[cube]
        to_local = data.geom_xmat[cube].reshape(3, 3).T
        pad_rows = []
        for geom in pads:
            nearest = np.zeros(6)
            distance = mujoco.mj_geomDistance(model, data, cube, geom, .05, nearest)
            local_point = (to_local @ (nearest[:3] - center)) * 1000
            local_center = (to_local @ (data.geom_xpos[geom] - center)) * 1000
            pad_rows.append({"signed_distance_mm": float(distance * 1000),
                             "cube_nearest_local_mm": local_point.tolist(),
                             "pad_center_local_mm": local_center.tolist()})
        other = min(float(mujoco.mj_geomDistance(model, data, cube, geom,
                                                .05, np.zeros(6))) * 1000
                    for geom in nearby)
        return {"pads": pad_rows, "nonpad_clearance_mm": other,
                "contacts": contacts(model, data)}

    def error(arm: np.ndarray, jaw: float, seed: np.ndarray) -> np.ndarray:
        row = assess(arm, jaw)
        residual = []
        for pad, x_side in zip(row["pads"], (10., -10.)):
            x, y, z = pad["cube_nearest_local_mm"]
            cx, cy, cz = pad["pad_center_local_mm"]
            # A robust pinch needs side-face contact below the top rim and
            # away from the front/back corners, not just two geom overlaps.
            residual.extend(((pad["signed_distance_mm"] + .45) * 2,
                             (x - x_side) * .4, y * .28,
                             max(0., z - 5.) * 1.2,
                             max(0., -z - 6.) * 1.2,
                             (cx - x_side * 1.3) * .15,
                             cy * .25, (cz - 6.) * .45))
        residual.append(max(0., 1. - row["nonpad_clearance_mm"]) * 2.)
        residual.extend((arm - seed) * .08)
        return np.asarray(residual)

    results = []
    for jaw in plan["jaw_cases_rad"]:
        waypoint = min(prior, key=lambda item: abs(item["jaw_rad"] - jaw))
        seeds = [original, np.asarray(waypoint["arm_rad"])]
        seeds += [np.asarray(waypoint["arm_rad"]) + rng.normal(0, .12, 4)
                  for _ in range(plan["random_starts"])]
        global_fit = differential_evolution(
            lambda arm: float(np.linalg.norm(error(arm, jaw, original))),
            bounds=list(zip(low, high)), seed=plan["optimizer_seed"] + round(jaw * 1000),
            popsize=plan["global_popsize"], maxiter=plan["global_maxiter"],
            tol=.001, polish=False)
        seeds.append(global_fit.x)
        candidates = []
        for seed in seeds:
            seed = np.clip(seed, low + 1e-6, high - 1e-6)
            fit = least_squares(lambda arm: error(arm, jaw, seed), seed,
                                bounds=(low, high), max_nfev=plan["max_nfev"],
                                ftol=1e-3, xtol=1e-3, gtol=1e-3)
            row = assess(fit.x, jaw)
            p, f = row["pads"]
            contact_ok = all(pad["signed_distance_mm"] <= -.1 for pad in (p, f))
            face_ok = all(abs(pad["cube_nearest_local_mm"][1]) <= 8.
                          and -6. <= pad["cube_nearest_local_mm"][2] <= 5.
                          for pad in (p, f))
            clean = row["contacts"]["two_sole_contact"] and not row["contacts"]["nonsole_floor_or_self_penetration_mm"]
            objective = float(np.linalg.norm(error(fit.x, jaw, seed)))
            candidates.append({"arm_rad": fit.x.tolist(), "objective": objective,
                               "contact_ok": contact_ok, "side_face_ok": face_ok,
                               "clean_robot": clean,
                               "nonpad_ok": row["nonpad_clearance_mm"] >= 1.,
                               **row})
        candidates.sort(key=lambda item: item["objective"])
        passing = [x for x in candidates if x["contact_ok"] and x["side_face_ok"]
                   and x["clean_robot"] and x["nonpad_ok"]]
        results.append({"jaw_rad": jaw, "searched": len(candidates),
                        "kinematic_face_candidates": len(passing),
                        "best": candidates[0],
                        "best_passing": passing[0] if passing else None})
        print(f"jaw={jaw:.3f} face candidates={len(passing)}/{len(candidates)} "
              f"best objective={candidates[0]['objective']:.3f}", flush=True)

    report = {"schema": "robot-learning-lab.cove50-opposing-face-screen.v1",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "body_sha256": sha(body_path), "source_trace_sha256": sha(trace_path),
              "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__, "source_time_s": plan["source_time_s"],
              "results": results,
              "face_candidate_any": any(row["kinematic_face_candidates"] for row in results),
              "limit": "Static fixed-cube kinematic search only. A found pose still needs a continuous route, native powered free-cube force and retention, then varied starts. Search failure within this envelope does not prove global impossibility."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"face_candidate_any": report["face_candidate_any"],
                      "counts": [x["kinematic_face_candidates"] for x in results]}))


if __name__ == "__main__":
    main()
