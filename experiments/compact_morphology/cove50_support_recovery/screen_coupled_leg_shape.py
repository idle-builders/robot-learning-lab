"""Contact and clearance screen for a coupled swing pitch/knee release shape."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-coupled-leg-shape-20261010"
PLAN = json.loads((OUT / "plan.json").read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sole_corner_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    half = model.geom_size[geom_id, :3]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    center = data.geom_xpos[geom_id]
    return min((center + rotation @ (np.asarray((a, b, c)) * half))[2]
               for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))


def inspect(model: mujoco.MjModel, data: mujoco.MjData, stance: str,
            qpos: np.ndarray, qvel: np.ndarray,
            pitch_offset: float, knee_offset: float) -> dict:
    swing = "left" if stance == "right" else "right"
    pitch = model.joint(f"{swing}_hip_pitch")
    knee = model.joint(f"{swing}_knee")
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qpos[pitch.qposadr[0]] += pitch_offset
    data.qpos[knee.qposadr[0]] += knee_offset
    mujoco.mj_forward(model, data)
    stance_site = model.site(f"{stance}_sole_center").id
    swing_site = model.site(f"{swing}_sole_center").id
    swing_geom = model.geom(f"{swing}_sole").id
    knee_geom = model.geom(f"{swing}_knee_motor_case").id
    ankle_geom = model.geom(f"{stance}_ankle_pitch_motor_case").id
    com = data.subtree_com[model.body("pelvis").id, :2]
    sole = data.site_xpos[stance_site, :2]
    contacts = []
    for contact in data.contact:
        names = [model.geom(g).name for g in (contact.geom1, contact.geom2)]
        if any("floor" in name or "terrain" in name for name in names):
            continue
        contacts.append({"geoms": sorted(names),
                         "penetration_mm": round(max(0, -float(contact.dist)) * 1000, 3)})
    return {
        "pitch_rad": round(float(data.qpos[pitch.qposadr[0]]), 5),
        "knee_rad": round(float(data.qpos[knee.qposadr[0]]), 5),
        "joint_limits_ok": bool(
            pitch.range[0] + .02 <= data.qpos[pitch.qposadr[0]] <= pitch.range[1] - .02 and
            knee.range[0] + .02 <= data.qpos[knee.qposadr[0]] <= knee.range[1] - .02),
        "swing_sole_center_mm": round(float(data.site_xpos[swing_site, 2]) * 1000, 3),
        "lowest_swing_sole_corner_mm": round(sole_corner_z(model, data, swing_geom) * 1000, 3),
        "com_minus_stance_xy_mm": np.round((com - sole) * 1000, 3).tolist(),
        "knee_ankle_case_distance_mm": round(float(mujoco.mj_geomDistance(
            model, data, knee_geom, ankle_geom, 1.0, None)) * 1000, 3),
        "self_contacts": contacts,
    }


def main() -> None:
    body = ROOT / PLAN["body"]
    pointer = ROOT / PLAN["design_pointer"]
    if digest(body) != PLAN["body_sha256"] or digest(pointer) != PLAN["design_pointer_sha256"]:
        raise RuntimeError("Selected Cove body or design pointer changed")
    model = mujoco.MjModel.from_xml_path(str(body))
    if model.nu != 21 or model.nq != 28:
        raise RuntimeError("Expected selected 21-axis body")
    data = mujoco.MjData(model)
    cfg = PLAN["offline_screen"]
    times = cfg["times_s"]
    trials, sources = [], {}
    for stance in ("right", "left"):
        path = ROOT / PLAN["sources"][stance]
        sources[stance] = {"path": PLAN["sources"][stance], "sha256": digest(path)}
        with np.load(path) as trace:
            dt = float(trace["dt_s"])
            if dt != .02:
                raise RuntimeError("Unexpected source trace period")
            saved = {round(t, 2): (trace["qpos"][round(t / dt) - 1].copy(),
                                    trace["qvel"][round(t / dt) - 1].copy())
                     for t in times}
        baseline = {t: inspect(model, data, stance, *saved[round(t, 2)], 0, 0)
                    for t in times}
        for pitch_offset in cfg["swing_hip_pitch_offsets_rad"]:
            for knee_offset in cfg["swing_knee_offsets_rad"]:
                rows = []
                for t in times:
                    row = inspect(model, data, stance, *saved[round(t, 2)],
                                  pitch_offset, knee_offset)
                    row["time_s"] = t
                    row["extra_abs_lateral_com_mm"] = round(
                        abs(row["com_minus_stance_xy_mm"][1]) -
                        abs(baseline[t]["com_minus_stance_xy_mm"][1]), 3)
                    rows.append(row)
                at_41 = next(row for row in rows if row["time_s"] == 4.1)
                candidate = (all(row["joint_limits_ok"] and not row["self_contacts"] and
                                 row["extra_abs_lateral_com_mm"] <= 5 for row in rows) and
                             at_41["swing_sole_center_mm"] > 25 and
                             at_41["lowest_swing_sole_corner_mm"] > 0)
                trials.append({"stance": stance, "pitch_offset_rad": pitch_offset,
                               "knee_offset_rad": knee_offset,
                               "static_candidate": bool(candidate), "states": rows})
    bilateral = []
    for pitch_offset in cfg["swing_hip_pitch_offsets_rad"]:
        for knee_offset in cfg["swing_knee_offsets_rad"]:
            pair = [t for t in trials if t["pitch_offset_rad"] == pitch_offset and
                    t["knee_offset_rad"] == knee_offset]
            if all(t["static_candidate"] for t in pair):
                bilateral.append({"pitch_offset_rad": pitch_offset,
                                  "knee_offset_rad": knee_offset})
    selected = min(bilateral, key=lambda x: (abs(x["pitch_offset_rad"]) +
                                            abs(x["knee_offset_rad"]),
                                            abs(x["pitch_offset_rad"]),
                                            x["knee_offset_rad"])) if bilateral else None
    result = {
        "schema": "robot-learning-lab.cove50-coupled-leg-shape-screen.v1",
        "classification": "Offline exact-state MuJoCo forward geometry screen; no native motor replay or policy claim",
        "mujoco_version": mujoco.__version__,
        "design_pointer_sha256": digest(pointer), "body_sha256": digest(body),
        "sources": sources, "plan": str((OUT / "plan.json").relative_to(ROOT)),
        "candidate_rule": cfg["candidate_rule"], "trials": trials,
        "bilateral_static_candidates": bilateral,
        "selected_for_native_replay": selected,
    }
    (OUT / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print("side candidates", {stance: sum(t["static_candidate"] for t in trials
                                      if t["stance"] == stance)
                              for stance in ("right", "left")})
    print("bilateral candidates", len(bilateral), "selected", selected)


if __name__ == "__main__":
    main()
