"""Bounded static arm-only jaw-entry check from saved native reach states."""

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
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/attained-aperture-ik-v1"
PLAN = OUT / "plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def with_cube(body_path: Path) -> mujoco.MjModel:
    root = ET.parse(body_path).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected candidate 21-axis body and one free cube")
    return model


def contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    cube = model.geom("audit_cube_geom").id
    floor = model.geom("floor").id
    palm = model.geom("right_hand").id
    finger = model.geom("right_finger_tip").id
    soles = {model.geom(f"{side}_sole").id for side in ("left", "right")}
    pad = {"palm": False, "finger": False}
    other_cube = {}
    robot = {}
    cube_floor_mm = 0.0
    feet = set()
    for contact in data.contact[:data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        depth_mm = -float(contact.dist) * 1000
        if cube in (a, b):
            partner = b if a == cube else a
            if partner == palm:
                pad["palm"] = True
            elif partner == finger:
                pad["finger"] = True
            elif partner == floor:
                cube_floor_mm = max(cube_floor_mm, depth_mm)
            elif depth_mm > 0.1:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                other_cube[name] = max(other_cube.get(name, 0.0), depth_mm)
            continue
        floor_partner = b if a == floor else a
        if floor in (a, b) and floor_partner in soles:
            feet.add(floor_partner)
            continue
        if depth_mm > 0.1:
            names = sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                           for g in (a, b))
            key = "/".join(names)
            robot[key] = max(robot.get(key, 0.0), depth_mm)
    return {
        "palm_contact": pad["palm"], "finger_contact": pad["finger"],
        "cube_floor_penetration_mm": round(cube_floor_mm, 4),
        "other_cube_robot_penetration_mm": {k: round(v, 4) for k, v in other_cube.items()},
        "nonsole_floor_or_self_penetration_mm": {k: round(v, 4) for k, v in robot.items()},
        "two_sole_contact": len(feet) == 2,
        "clean_two_jaw": (pad["palm"] and pad["finger"] and cube_floor_mm <= 0.1
                          and not other_cube and not robot),
    }


def load_sources(plan: dict) -> list[dict]:
    sources = []
    for group in plan["source_groups"]:
        for name in group["cases"]:
            stem = ROOT / group["root"] / name
            result_path = stem.with_suffix(".json")
            trace_path = stem.with_suffix(".npz")
            result = json.loads(result_path.read_text())
            with np.load(trace_path) as trace:
                qpos = trace["qpos"].copy()
            timeseries = result["timeseries"]
            if len(timeseries) != len(qpos) or qpos.shape[1] != 35:
                raise ValueError(f"Trace/metrics mismatch: {name}")
            ticks = [i for i, row in enumerate(timeseries)
                     if row["phase"] == "reach" and row["finger_contact"]]
            if not ticks:
                raise ValueError(f"No native fingertip contact: {name}")
            for point, tick in (("middle", ticks[len(ticks) // 2]),
                                ("last", ticks[-1])):
                pose_tick = min(tick + 1, len(qpos) - 1)
                sources.append({
                    "group": group["name"], "case": name, "point": point,
                    "contact_tick": tick, "pose_tick": pose_tick,
                    "native_time_s": timeseries[tick]["time_s"],
                    "source_report": str(result_path.relative_to(ROOT)),
                    "source_report_sha256": sha(result_path),
                    "source_trace": str(trace_path.relative_to(ROOT)),
                    "source_trace_sha256": sha(trace_path),
                    "qpos": qpos[pose_tick].astype(float),
                })
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text())
    body_path = ROOT / plan["candidate_body"]
    model = with_cube(body_path)
    data = mujoco.MjData(model)
    site_id = model.site("right_hand_center").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    arm_names = plan["solve"]["joint_names"]
    arm_adr = np.array([model.joint(name).qposadr[0] for name in arm_names])
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    sources = load_sources(plan)
    source_rows = []
    fit_rows = []
    screen_rows = []
    fitted_states = []
    for source in sources:
        saved = source["qpos"]
        qbase = saved.copy()
        if "cube_floor_height_m" in plan["solve"]:
            qbase[cube_adr + 2] = plan["solve"]["cube_floor_height_m"]
        qbase[jaw_adr] = plan["solve"]["jaw_open_rad"]
        data.qpos[:] = qbase
        mujoco.mj_forward(model, data)
        initial_site = data.site_xpos[site_id].copy()
        cube_xyz = qbase[cube_adr:cube_adr + 3].copy()
        initial_contacts = contacts(model, data)
        source_id = f"{source['case']}-{source['point']}"
        source_rows.append({
            "case": source_id, "group": source["group"],
            "contact_tick": source["contact_tick"], "pose_tick": source["pose_tick"],
            "native_time_s": source["native_time_s"],
            "source_report": source["source_report"],
            "source_report_sha256": source["source_report_sha256"],
            "source_trace": source["source_trace"],
            "source_trace_sha256": source["source_trace_sha256"],
            "cube_xyz_m": np.round(cube_xyz, 6).tolist(),
            "native_cube_height_m": round(float(saved[cube_adr + 2]), 6),
            "source_open_site_xyz_m": np.round(initial_site, 6).tolist(),
            "cube_to_open_site_mm": np.round((cube_xyz - initial_site) * 1000, 3).tolist(),
            "source_open_contacts_on_candidate": initial_contacts,
        })
        arm0 = saved[arm_adr].copy()
        lower = np.array([model.joint(name).range[0] for name in arm_names])
        upper = np.array([model.joint(name).range[1] for name in arm_names])
        travel = plan["solve"]["maximum_absolute_arm_delta_rad"]
        lower = np.maximum(lower, arm0 - travel)
        upper = np.minimum(upper, arm0 + travel)
        if np.any(upper <= lower):
            raise ValueError(f"No right-arm bounds: {source_id}")
        for above_mm in plan["solve"]["target_site_above_cube_mm"]:
            target = cube_xyz + np.array([0.0, 0.0, above_mm / 1000])

            def site_at(x: np.ndarray) -> np.ndarray:
                data.qpos[:] = qbase
                data.qpos[arm_adr] = x
                mujoco.mj_forward(model, data)
                return data.site_xpos[site_id].copy()

            def residual(x: np.ndarray) -> np.ndarray:
                xyz = site_at(x)
                return np.r_[(xyz - target) * 1000 / plan["solve"]["site_residual_scale_mm"],
                             (x - arm0) / plan["solve"]["arm_regularization_scale_rad"]]

            guess = np.clip(arm0, lower + 1e-8, upper - 1e-8)
            fit = least_squares(residual, guess, bounds=(lower, upper),
                                max_nfev=plan["solve"]["max_solver_evaluations"],
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            xyz = site_at(fit.x)
            fitted = data.qpos.copy()
            fit_id = f"{source_id}-z{above_mm}"
            fit_contacts = contacts(model, data)
            err_mm = float(np.linalg.norm((xyz - target) * 1000))
            fit_row = {
                "case": fit_id, "source_case": source_id, "group": source["group"],
                "target_site_above_cube_mm": above_mm,
                "target_site_xyz_m": np.round(target, 6).tolist(),
                "fitted_site_xyz_m": np.round(xyz, 6).tolist(),
                "site_error_mm": round(err_mm, 4),
                "arm_delta_rad": {name: round(float(delta), 6)
                                  for name, delta in zip(arm_names, fit.x - arm0)},
                "arm_joint_rad": {name: round(float(value), 6)
                                  for name, value in zip(arm_names, fit.x)},
                "solver_success": bool(fit.success), "solver_nfev": int(fit.nfev),
                "open_contacts": fit_contacts,
            }
            fit_rows.append(fit_row)
            fitted_states.append(fitted)
            for dx_mm, dy_mm in plan["screen"]["cube_xy_offset_mm"]:
                for yaw_deg in plan["screen"]["cube_yaw_deg"]:
                    placement = f"{fit_id}-x{dx_mm}-y{dy_mm}-yaw{yaw_deg}"
                    for angle in plan["screen"]["jaw_angles_rad"]:
                        data.qpos[:] = fitted
                        data.qpos[jaw_adr] = angle
                        data.qpos[cube_adr] += dx_mm / 1000
                        data.qpos[cube_adr + 1] += dy_mm / 1000
                        saved_quat = saved[cube_adr + 3:cube_adr + 7]
                        yaw = np.deg2rad(yaw_deg)
                        yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                        quat = np.empty(4)
                        mujoco.mju_mulQuat(quat, yaw_quat, saved_quat)
                        data.qpos[cube_adr + 3:cube_adr + 7] = quat
                        mujoco.mj_forward(model, data)
                        contact = contacts(model, data)
                        screen_rows.append({
                            "case": f"{placement}-jaw{angle}",
                            "fit_case": fit_id, "placement": placement,
                            "cube_xy_offset_mm": [dx_mm, dy_mm], "cube_yaw_deg": yaw_deg,
                            "jaw_rad": angle, **contact,
                        })
    by_fit = {row["case"]: [] for row in fit_rows}
    for row in screen_rows:
        by_fit[row["fit_case"]].append(row)
    for row in fit_rows:
        cases = by_fit[row["case"]]
        placement = {}
        for case in cases:
            placement.setdefault(case["placement"], []).append(case)
        clean = [name for name, trio in placement.items()
                 if len(trio) == 3 and all(c["clean_two_jaw"] for c in trio)]
        row["clean_static_triplets"] = len(clean)
        row["clean_static_triplet_placements"] = clean
        row["two_jaw_static_cases"] = sum(c["palm_contact"] and c["finger_contact"]
                                          for c in cases)
        row["clean_two_jaw_static_cases"] = sum(c["clean_two_jaw"] for c in cases)
        row["static_ik_screen_pass"] = (row["site_error_mm"] <=
                                        plan["screen"]["site_xyz_error_limit_mm"]
                                        and not row["open_contacts"]["nonsole_floor_or_self_penetration_mm"]
                                        and row["clean_static_triplets"] > 0)
    source_summary = []
    for group in plan["source_groups"]:
        subset = [r for r in fit_rows if r["group"] == group["name"]]
        source_summary.append({
            "case": group["name"], "source_states": len([r for r in source_rows
                                                         if r["group"] == group["name"]]),
            "fits": len(subset),
            "site_fits_within_2mm": sum(r["site_error_mm"] <= 2 for r in subset),
            "fits_with_clean_static_triplet": sum(r["clean_static_triplets"] > 0
                                                  for r in subset),
            "static_ik_screen_pass": sum(r["static_ik_screen_pass"] for r in subset),
            "best_site_error_mm": min(r["site_error_mm"] for r in subset),
            "max_clean_static_triplets": max(r["clean_static_triplets"] for r in subset),
        })
    np.savez_compressed(out / "fitted-states.npz", qpos=np.asarray(fitted_states),
                        fit_case=np.asarray([r["case"] for r in fit_rows]))
    report = {
        "schema": "robot-learning-lab.cove50-attained-aperture-ik.v1",
        "selected_pointer_sha256": sha(ROOT / plan["selected_pointer"]),
        "candidate_body_sha256": sha(body_path),
        "plan_sha256": sha(plan_path),
        "source_script_sha256": sha(Path(__file__)),
        "fitted_states_sha256": sha(out / "fitted-states.npz"),
        "mujoco_version": mujoco.__version__,
        "source_states": len(source_rows), "fits": len(fit_rows),
        "static_contact_cases": len(screen_rows),
        "summary": source_summary,
        "limits": [plan["interpretation_limit"],
                   "The learned-v8 group comes from the selected hand and is a counterfactual pose transfer; only the earlier hybrid group was motor-attained on the unselected candidate.",
                   "The cube is kinematically held while the jaw angle changes; no motor, contact-force, lift or possession gate runs here."],
    }
    (out / "source-matrix.json").write_text(json.dumps({"cases": source_rows}, indent=2) + "\n")
    (out / "fit-matrix.json").write_text(json.dumps({"cases": fit_rows}, indent=2) + "\n")
    (out / "contact-matrix.json").write_text(json.dumps({"cases": screen_rows}, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
