"""Search small right-pad collision changes for a bounded side-face grip coupon."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import contacts


OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/side-pad-sweep-v1"
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
    source_path = ROOT / plan["source_body"]
    trace_path = ROOT / plan["native_source"]
    with np.load(trace_path) as trace:
        source = trace["qpos"][round(plan["source_time_s"] / float(trace["step_dt_s"]))].copy()
    rows = []
    candidates = itertools.product(plan["pad_drop_mm"], plan["finger_inward_mm"],
                                   plan["pad_half_width_mm"], plan["pad_half_height_mm"])
    for index, (drop, inward, width, height) in enumerate(candidates):
        tree = ET.parse(source_path)
        root = tree.getroot()
        palm = root.find(".//geom[@name='right_hand']")
        finger = root.find(".//geom[@name='right_finger_tip']")
        assert palm is not None and finger is not None
        palm.set("pos", f"-0.035 0.01 {-0.141-drop/1000}")
        palm.set("size", f"0.006 {width/1000} {height/1000}")
        finger.set("pos", f"-0.04 {0.004+inward/1000} {-drop/1000}")
        finger.set("size", f"0.006 {width/1000} {height/1000}")
        cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
        ET.SubElement(cube, "freejoint", name="audit_cube_free")
        ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                      size="0.01 0.01 0.01", mass="0.00625", friction="0.8 0.01 0.001")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM])
        jaw_adr = model.joint("right_gripper").qposadr[0]
        cube_id = model.geom("audit_cube_geom").id
        pad_ids = [model.geom(name).id for name in ("right_hand", "right_finger_tip")]
        near_ids = [model.geom(name).id for name in NEAR]
        original = source[arm_adr]
        low = np.maximum([model.joint(name).range[0] for name in ARM],
                         original - plan["arm_envelope_rad"])
        high = np.minimum([model.joint(name).range[1] for name in ARM],
                          original + plan["arm_envelope_rad"])

        def assess(arm: np.ndarray, jaw: float) -> dict:
            data.qpos[:] = source
            data.qpos[arm_adr] = arm
            data.qpos[jaw_adr] = jaw
            mujoco.mj_forward(model, data)
            center = data.geom_xpos[cube_id]
            to_local = data.geom_xmat[cube_id].reshape(3, 3).T
            pads = []
            for geom in pad_ids:
                nearest = np.zeros(6)
                dist = mujoco.mj_geomDistance(model, data, cube_id, geom, .05, nearest)
                pads.append({"distance_mm": float(dist * 1000),
                             "cube_point_mm": ((to_local @ (nearest[:3] - center)) * 1000).tolist()})
            other = min(float(mujoco.mj_geomDistance(model, data, cube_id, geom,
                                                     .05, np.zeros(6))) * 1000
                        for geom in near_ids)
            return {"pads": pads, "nonpad_clearance_mm": other, "contacts": contacts(model, data)}

        entry = assess(original, 0.)
        entry_distances = [pad["distance_mm"] for pad in entry["pads"]]
        if any(d < -plan["entry_max_overlap_mm"] for d in entry_distances):
            rows.append({"index": index, "drop_mm": drop, "inward_mm": inward,
                         "half_width_mm": width, "half_height_mm": height,
                         "entry_distances_mm": entry_distances, "rejected": "entry_penetration"})
            continue

        def cost(arm: np.ndarray) -> float:
            state = assess(arm, .5)
            residual = []
            for pad, x_side in zip(state["pads"], (10., -10.)):
                x, y, z = pad["cube_point_mm"]
                residual.extend(((pad["distance_mm"] + .35) * 2,
                                 (x - x_side) * .5, y * .7, z * .25,
                                 max(0., z - 5.) * 2,
                                 max(0., -z - 6.) * 2))
            residual.extend((max(0., 1. - state["nonpad_clearance_mm"]) * 3,
                             np.linalg.norm(arm - original) * .1))
            return float(np.linalg.norm(residual))

        fit = differential_evolution(cost, bounds=list(zip(low, high)),
                                     seed=plan["optimizer_seed"] + index,
                                     popsize=plan["global_popsize"],
                                     maxiter=plan["global_maxiter"],
                                     tol=.001, polish=False)
        final = assess(fit.x, .5)
        p, f = final["pads"]
        side = all(abs(pad["cube_point_mm"][1]) <= 8.
                   and -6. <= pad["cube_point_mm"][2] <= 5.
                   for pad in (p, f))
        pad_ok = all(-1. <= pad["distance_mm"] <= -.1 for pad in (p, f))
        clean = final["contacts"]["two_sole_contact"] and not final["contacts"]["nonsole_floor_or_self_penetration_mm"]
        passed = side and pad_ok and clean and final["nonpad_clearance_mm"] >= 1.
        rows.append({"index": index, "drop_mm": drop, "inward_mm": inward,
                     "half_width_mm": width, "half_height_mm": height,
                     "entry_distances_mm": entry_distances,
                     "arm_rad": fit.x.tolist(), "objective": float(fit.fun),
                     "final": final, "face_contact_pass": passed})
    feasible = sorted((r for r in rows if r.get("face_contact_pass")),
                      key=lambda r: r["objective"])
    searched = sorted((r for r in rows if "objective" in r),
                      key=lambda r: r["objective"])
    report = {"schema": "robot-learning-lab.cove50-side-pad-sweep.v1",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "source_body_sha256": sha(source_path), "source_trace_sha256": sha(trace_path),
              "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
              "mujoco_version": mujoco.__version__, "total_cases": len(rows),
              "screened_cases": len(searched), "feasible_side_face_cases": len(feasible),
              "best_feasible": feasible[0] if feasible else None,
              "best_searched": searched[0] if searched else None,
              "limit": "One fixed cube/root/leg state, one final jaw angle and collision-only pad changes. A geometric pass is not a reachable continuous route or powered grip; a failure is not global impossibility."}
    (OUT / "matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("best_feasible", "best_searched")}, indent=2))


if __name__ == "__main__":
    main()
