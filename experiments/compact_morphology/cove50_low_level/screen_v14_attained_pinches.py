"""Static local geometry screen near *attained* V14 one-pad touch states.

This never steps a motor or scores a grasp. It only asks whether the current
Direct Frame jaw can put both pads on the saved free cube near a real learned
touch pose. CUDA contact/transition claims require a separate native visit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
BODY_SHA256 = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
TRACES = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-contact-phase-v14/native-development"
OUT = ROOT / "artifacts/ape-low-level-training-20261015/object-attained-pinch-static-v1/report.json"
SEEDS = (37301, 37302, 37303, 37304, 37305)
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
          "right_elbow_pitch", "right_gripper")
SAMPLES_PER_FAMILY = 5000
ARM_DELTA_RAD = 0.30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen_model() -> mujoco.MjModel:
    root = ET.parse(BODY).getroot()
    world = root.find("worldbody")
    table = ET.SubElement(world, "body", name="screen_table", pos="0.05 -0.14 0.08")
    ET.SubElement(table, "geom", name="screen_table_top", type="box",
                  size="0.025 0.04 0.05", friction="0.8 0.01 0.001")
    cube = ET.SubElement(world, "body", name="screen_cube", pos="0 0 0.14")
    ET.SubElement(cube, "freejoint", name="screen_cube_free")
    ET.SubElement(cube, "geom", name="screen_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (model.nq, model.nv, model.nu) != (35, 33, 21):
        raise ValueError("body/free-cube dimensions differ from saved traces")
    return model


def main() -> None:
    if sha(BODY) != BODY_SHA256:
        raise ValueError("compiled Direct Frame body differs from the training program")
    model = screen_model()
    data = mujoco.MjData(model)
    pad_ids = [model.geom(name).id for name in ("right_hand", "right_finger_tip")]
    cube_id = model.geom("screen_cube_geom").id
    table_id = model.geom("screen_table_top").id
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINTS]
    bounds = np.array([model.joint(name).range for name in JOINTS])
    rng = np.random.default_rng(4138)
    rows = []
    for seed in SEEDS:
        trace = TRACES / f"seed-{seed}-residual.npz"
        with np.load(trace) as saved:
            touched = np.flatnonzero(saved["physical_touch"])
            if not len(touched):
                raise ValueError(f"V14 seed {seed} lacks physical touch")
            frame = int(touched[len(touched) // 2])
            base = saved["qpos"][frame].astype(float)
            recorded_cube = saved["cube_xyz_m"][frame].astype(float)
        if np.linalg.norm(base[28:31] - recorded_cube) > 1e-5:
            raise ValueError(f"V14 seed {seed} cube qpos/trace mismatch")
        best = None
        clean_two_pad = 0
        raw_two_pad = 0
        jaw_only_two_pad = 0
        jaw_only_best_gap_mm = float("inf")
        for sample in range(SAMPLES_PER_FAMILY + 21):
            qpos = base.copy()
            if sample < 21:
                qpos[addresses[-1]] = sample / 20 * 0.5  # jaw only
            else:
                qpos[addresses[:4]] = np.clip(
                    qpos[addresses[:4]] + rng.uniform(-ARM_DELTA_RAD, ARM_DELTA_RAD, 4),
                    bounds[:4, 0], bounds[:4, 1])
                qpos[addresses[-1]] = rng.uniform(0, 0.5)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            witness = np.empty(6)
            gaps = [float(mujoco.mj_geomDistance(model, data, pad, cube_id, 1.0, witness))
                    for pad in pad_ids]
            robot_cube_other = []
            hand_table = []
            cube_table = False
            for index in range(data.ncon):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if cube_id in pair:
                    partner = (pair - {cube_id}).pop()
                    cube_table |= partner == table_id
                    if partner not in (*pad_ids, table_id) and contact.dist < 0:
                        robot_cube_other.append(mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, partner))
                if table_id in pair and pair.intersection(pad_ids) and contact.dist < 0:
                    hand_table.append(True)
            two_pad = max(gaps) <= 0
            raw_two_pad += int(two_pad)
            if sample < 21:
                jaw_only_two_pad += int(two_pad)
                jaw_only_best_gap_mm = min(jaw_only_best_gap_mm, max(gaps) * 1000)
            clean = two_pad and cube_table and not robot_cube_other and not hand_table
            clean_two_pad += int(clean)
            candidate = {
                "sample": sample, "jaw_rad": round(float(qpos[addresses[-1]]), 6),
                "arm_delta_rad": np.round(qpos[addresses[:4]] - base[addresses[:4]], 6).tolist(),
                "fixed_gap_mm": round(gaps[0] * 1000, 4),
                "moving_gap_mm": round(gaps[1] * 1000, 4),
                "max_pad_gap_mm": round(max(gaps) * 1000, 4),
                "cube_table_contact": bool(cube_table),
                "other_cube_contacts": sorted(set(robot_cube_other)),
                "hand_table_contact": bool(hand_table), "clean_static_two_pad": bool(clean),
            }
            if best is None or (clean, -max(gaps)) > (best["clean_static_two_pad"],
                                                      -best["max_pad_gap_mm"] / 1000):
                best = candidate
        rows.append({"seed": seed, "trace": str(trace.relative_to(ROOT)),
                     "trace_sha256": sha(trace), "attained_touch_frame": frame,
                     "attained_touch_time_s": round((frame + 1) / 50, 3),
                     "recorded_cube_xyz_m": recorded_cube.tolist(),
                     "jaw_only_two_pad_cases": jaw_only_two_pad,
                     "jaw_only_best_max_pad_gap_mm": round(jaw_only_best_gap_mm, 4),
                     "raw_two_pad_cases": raw_two_pad,
                     "clean_static_two_pad_cases": clean_two_pad,
                     "best_static_candidate": best})
    report = {
        "schema": "robot-learning-lab.v14-attained-pinch-static.v1",
        "claim": "Local static kinematic feasibility near five attained V14 learned-touch states; no motor transition, retained grasp, lift, CUDA score or new-body transfer",
        "body": str(BODY.relative_to(ROOT)), "body_sha256": sha(BODY),
        "screen_source_sha256": sha(Path(__file__)),
        "local_mujoco_version": mujoco.__version__,
        "sampling": {"seed": 4138, "arm_half_range_rad": ARM_DELTA_RAD,
                     "samples_per_family": SAMPLES_PER_FAMILY, "jaw_only_samples": 21,
                     "jaw_range_rad": [0, 0.5]},
        "rows": rows,
        "summary": {
            "jaw_only_two_pad_families": sum(r["jaw_only_two_pad_cases"] > 0 for r in rows),
            "raw_two_pad_arm_adjusted_families": sum(r["raw_two_pad_cases"] > 0 for r in rows),
            "clean_two_pad_arm_adjusted_families": sum(r["clean_static_two_pad_cases"] > 0 for r in rows),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"clean_families": sum(r["clean_static_two_pad_cases"] > 0 for r in rows),
                      "report": str(OUT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
