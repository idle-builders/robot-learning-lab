"""Optimistic, varied-start dynamic support gate for the narrow Cove hand coupon."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.motor_close import contact_info
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_static_contact import mirror_qpos


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2"
BODY = BASE / "candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"
STATIC = BASE / "static-contact-v1/cube-matrix.json"
OUT = BASE / "motor-sweep-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases() -> list[dict]:
    all_rows = json.loads(STATIC.read_text())["cases"]
    grouped = defaultdict(dict)
    for row in all_rows:
        if row["cube_height_mm"] != 10.0 or row["raise_target_mm"] not in (12, 15):
            continue
        key = (row["side"], row["source_ik_start"], row["raise_target_mm"],
               *row["xy_offset_mm"], row["yaw_deg"])
        grouped[key][row["jaw_rad"]] = row
    candidates = defaultdict(list)
    for key, by_angle in grouped.items():
        if not all(by_angle.get(a, {}).get("clean_two_jaw") for a in (0.0, .25, .5)):
            continue
        penetration = max(-by_angle[a][f"{pad}_min_contact_dist_mm"]
                          for a in (0.0, .25, .5) for pad in ("palm", "finger"))
        candidates[key[:3]].append((penetration, key))
    chosen = []
    for side in ("right", "left"):
        for height in (12, 15):
            starts = sorted(start for s, start, h in candidates if s == side and h == height)
            if len(starts) != 25:
                raise ValueError(f"Expected 25 clean starts for {side} +{height} mm")
            for start in starts:
                penetration, key = min(candidates[(side, start, height)])
                chosen.append({"side": side, "source_ik_start": start,
                               "raise_target_mm": height,
                               "cube_height_mm": 10.0,
                               "xy_offset_mm": list(key[3:5]),
                               "yaw_deg": key[5],
                               "screen_max_pad_penetration_mm": penetration})
    return chosen


def main() -> None:
    cases = select_cases()
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="sweep_cube", pos="0 0 .01")
    ET.SubElement(cube, "freejoint", name="sweep_cube_free")
    ET.SubElement(cube, "geom", name="diagnostic_cube_geom", type="box",
                  size=".01 .01 .01", mass=".00625", friction=".8 .01 .001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.opt.timestep != .002:
        raise ValueError("Motor interface changed")
    with np.load(STATES) as saved:
        states = {(int(start), int(height)): q.copy()
                  for start, height, q in zip(saved["start"], saved["raise_mm"], saved["qpos"])}
    cube_adr = model.joint("sweep_cube_free").qposadr[0]
    target_adrs = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    leg_ids = [i for i in range(model.nu) if any(
        part in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for part in ("hip", "knee", "ankle"))]
    rows = []
    for case in cases:
        side = case["side"]
        data = mujoco.MjData(model)
        right_pose = states[(case["source_ik_start"], case["raise_target_mm"])]
        pose = right_pose if side == "right" else mirror_qpos(model, right_pose)
        data.qpos[:28] = pose
        jaw_adr = model.joint(f"{side}_gripper").qposadr[0]
        data.qpos[jaw_adr] = 0
        mujoco.mj_forward(model, data)
        hand = data.site_xpos[model.site(f"{side}_hand_center").id].copy()
        dx, dy = case["xy_offset_mm"]
        data.qpos[cube_adr:cube_adr + 3] = (hand[0] + dx/1000, hand[1] + dy/1000, .01)
        yaw = np.deg2rad(case["yaw_deg"])
        data.qpos[cube_adr + 3:cube_adr + 7] = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))
        mujoco.mj_forward(model, data)
        initial_cube = data.qpos[cube_adr:cube_adr + 3].copy()
        initial_root_z = float(data.qpos[2])
        targets = pose[target_adrs].copy()
        active = next(i for i in range(model.nu)
                      if model.actuator_trnid[i, 0] == model.joint(f"{side}_gripper").id)
        max_drift = 0.0
        max_root = 0.0
        nonsole_frames = 0
        both_sole_frames = 0
        two_jaw_last100 = 0
        leg_limit_frames = 0
        first_floor_s = None
        first_support_loss_s = None
        final_info = None
        for step in range(301):
            time_s = step * .002
            targets[active] = min(.5, .5*time_s/.3)
            data.ctrl[:] = targets
            mujoco.mj_forward(model, data)
            info = contact_info(model, data, side)
            final_info = info
            drift = float(np.linalg.norm((data.qpos[cube_adr:cube_adr + 3] - initial_cube)[:2]))*1000
            root_delta = (float(data.qpos[2]) - initial_root_z)*1000
            max_drift = max(max_drift, drift)
            max_root = max(max_root, abs(root_delta))
            if info["nonsole_floor"]:
                nonsole_frames += 1
                if first_floor_s is None:
                    first_floor_s = time_s
            if set(info["sole_floor_sides"]) == {"left", "right"}:
                both_sole_frames += 1
            elif first_support_loss_s is None:
                first_support_loss_s = time_s
            if time_s >= .5 and info["palm"] and info["finger"]:
                two_jaw_last100 += 1
            if any(abs(float(data.actuator_force[i])) >=
                   max(abs(float(model.actuator_forcerange[i, 0])),
                       abs(float(model.actuator_forcerange[i, 1])))*.99
                   for i in leg_ids):
                leg_limit_frames += 1
            if step < 300:
                mujoco.mj_step(model, data)
        final_jaw = float(data.qpos[jaw_adr])
        passed = (final_jaw >= .45 and two_jaw_last100 >= 40 and
                  nonsole_frames == 0 and both_sole_frames == 301 and
                  max_root <= 10 and max_drift <= 5)
        rows.append({**case,
                     "final_jaw_rad": round(final_jaw, 4),
                     "two_jaw_final_100ms_frames": two_jaw_last100,
                     "both_sole_frames": both_sole_frames,
                     "nonsole_floor_frames": nonsole_frames,
                     "leg_limit_frames": leg_limit_frames,
                     "first_nonsole_floor_s": first_floor_s,
                     "first_two_sole_loss_s": first_support_loss_s,
                     "max_abs_root_z_delta_mm": round(max_root, 3),
                     "max_cube_xy_drift_mm": round(max_drift, 3),
                     "final_cube_xy_drift_mm": round(float(np.linalg.norm(
                         (data.qpos[cube_adr:cube_adr + 3] - initial_cube)[:2]))*1000, 3),
                     "final_nonsole_floor": final_info["nonsole_floor"],
                     "diagnostic_gate_pass": passed})
    summary = []
    for side in ("right", "left"):
        for height in (12, 15):
            sub = [row for row in rows if row["side"] == side and row["raise_target_mm"] == height]
            summary.append({
                "case": f"{side} +{height} mm", "episodes": len(sub),
                "diagnostic_gate_pass": sum(r["diagnostic_gate_pass"] for r in sub),
                "jaw_reached_0p45": sum(r["final_jaw_rad"] >= .45 for r in sub),
                "retained_two_sole_all_frames": sum(r["both_sole_frames"] == 301 for r in sub),
                "no_nonsole_floor": sum(r["nonsole_floor_frames"] == 0 for r in sub),
                "cube_drift_le_5_mm": sum(r["max_cube_xy_drift_mm"] <= 5 for r in sub),
                "leg_limit_episodes": sum(r["leg_limit_frames"] > 0 for r in sub),
                "median_max_root_delta_mm": round(float(np.median(
                    [r["max_abs_root_z_delta_mm"] for r in sub])), 3),
                "median_max_cube_drift_mm": round(float(np.median(
                    [r["max_cube_xy_drift_mm"] for r in sub])), 3),
                "best_max_root_delta_mm": min(r["max_abs_root_z_delta_mm"] for r in sub),
            })
    report = {
        "schema": "robot-learning-lab.cove50-narrow-bridge-motor-sweep.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "static_grid_sha256": sha(STATIC),
        "raised_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "mj_version": mujoco.__version__,
        "episodes": len(rows), "summary": summary,
        "limits": [
            "The chosen cube placements are optimistic selections from the same static screen, not held-out variants.",
            "All starts are motor-unattained static IK. Closure holds non-gripper joint targets but uses no learned balance or acquisition controller.",
            "This gate tests support and jaw tracking only; no commanded lift, possession, release or learned-policy pass is implied.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "summary-matrix.json").write_text(json.dumps({"cases": summary}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
