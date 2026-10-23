"""Exact Direct Frame free-cube entry and authored motor-target feasibility probe.

This is setup evidence, not a learned policy or a native MJLab/BAM score. The
optional retained trace supplies physical joint targets, never object pose
targets after reset; the cube is always a simulated free body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"
TRACE = ROOT / ("artifacts/ape-cove50-reach-goal-channels-20261010/v158/"
                "native-development-contact/seed-24607-contact-24607-prescribed.npz")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_cube_model() -> mujoco.MjModel:
    source = ET.parse(BODY).getroot()
    cube = ET.SubElement(source.find("worldbody"), "body", name="free_cube")
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(cube, "geom", name="cube_geom", type="box", size=".01 .01 .01",
                  mass=".00625", friction=".8 .01 .001", rgba=".95 .35 .05 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    assert model.nq == 35 and model.nu == 21
    return model


def contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    cube = model.geom("cube_geom").id
    return {name: any(cube in (c.geom1, c.geom2) and
                      model.geom(name).id in (c.geom1, c.geom2)
                      for c in data.contact)
            for name in ("right_hand", "right_finger_tip")}


def projected_aperture(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    fixed, moving = [model.geom(n).id for n in ("right_hand", "right_finger_tip")]
    direction = data.geom_xpos[moving] - data.geom_xpos[fixed]
    center_distance = float(np.linalg.norm(direction))
    direction /= center_distance
    radii = []
    for geom in (fixed, moving):
        rotation = data.geom_xmat[geom].reshape(3, 3)
        radii.append(float(np.abs(direction @ rotation) @ model.geom_size[geom]))
    return {
        "pad_center_distance_mm": round(center_distance * 1000, 3),
        "projected_free_gap_mm": round((center_distance - sum(radii)) * 1000, 3),
        "fixed_pad_xyz_m": np.round(data.geom_xpos[fixed], 5).tolist(),
        "moving_pad_xyz_m": np.round(data.geom_xpos[moving], 5).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pointer = json.loads(POINTER.read_text())
    assert pointer["id"] == "cove-50-direct-frame"
    assert digest(BODY) == "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
    model = free_cube_model()
    data = mujoco.MjData(model)
    with np.load(TRACE) as saved:
        qpos = saved["qpos"].astype(float)
        step_dt = float(saved["step_dt_s"])
    assert qpos.shape[1] == model.nq and abs(step_dt - .02) < 1e-9
    assert model.opt.timestep == .002
    cube_start = qpos[0, 28:31].copy()
    assert abs(cube_start[2] - .01) < 1e-6
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    entry_contacts = contacts(model, data)
    entry_pad_distance = min(float(np.linalg.norm(data.geom_xpos[model.geom(n).id] - cube_start))
                             for n in entry_contacts)
    assert not any(entry_contacts.values()) and entry_pad_distance > .03
    # Aperture width is measured on the attained terminal arm pose; changing
    # the gripper angle cannot translate a pad centered on its hinge.
    aperture = {}
    for angle in (0., .25, .5):
        data.qpos[:] = qpos[-1]
        data.qpos[model.joint("right_gripper").qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        aperture[str(angle)] = projected_aperture(model, data)
    # A raw-MJCF motor-target replay of a saved reach, from an initially free
    # cube. This tests the object scene and a candidate teacher path only. Its
    # position servos differ from the MJLab/BAM path that generated the trace.
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    data.ctrl[:] = np.array([qpos[0, model.jnt_qposadr[model.actuator_trnid[a, 0]]]
                             for a in range(model.nu)])
    mujoco.mj_forward(model, data)
    first_contact = None
    touch_frames = both_frames = lifted_both_frames = 0
    pad_frames = {"right_hand": 0, "right_finger_tip": 0}
    max_cube_z = float(data.qpos[30])
    min_pelvis_z = float(data.qpos[2])
    samples = []
    for frame in range(len(qpos)):
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator, 0]
            target = qpos[frame, model.jnt_qposadr[joint]]
            # The saved reach did not attempt closing. Diagnose the simple
            # gripper-closure hypothesis during its final 2 s.
            if (model.joint(joint).name == "right_gripper" and frame >= len(qpos) - 100):
                target = .5
            data.ctrl[actuator] = target
        for _ in range(10):
            mujoco.mj_step(model, data)
        state = contacts(model, data)
        touch = any(state.values())
        both = all(state.values())
        touch_frames += int(touch)
        for pad in pad_frames:
            pad_frames[pad] += int(state[pad])
        both_frames += int(both)
        lifted_both_frames += int(both and data.qpos[30] >= .04)
        max_cube_z = max(max_cube_z, float(data.qpos[30]))
        min_pelvis_z = min(min_pelvis_z, float(data.qpos[2]))
        if touch and first_contact is None:
            first_contact = frame * step_dt
        if frame % 50 == 0 or frame == len(qpos) - 1:
            samples.append({"time_s": round(frame * step_dt, 2),
                            "cube_xyz_m": np.round(data.qpos[28:31], 5).tolist(),
                            "pelvis_z_m": round(float(data.qpos[2]), 5),
                            "fixed_pad_contact": state["right_hand"],
                            "moving_pad_contact": state["right_finger_tip"]})
    report = {
        "schema": "robot-learning-lab.cove50-free-object-setup.v1",
        "scope": "Exact-body raw-MJCF free-cube scene and authored motor-target diagnostic; not learned or native MJLab/BAM qualification",
        "source": {"body": str(BODY.relative_to(ROOT)), "body_sha256": digest(BODY),
                   "pointer": str(POINTER.relative_to(ROOT)), "pointer_sha256": digest(POINTER),
                   "retained_trace": str(TRACE.relative_to(ROOT)), "retained_trace_sha256": digest(TRACE)},
        "model": {"nq": model.nq, "nu": model.nu, "cube_size_m": [.02] * 3,
                  "cube_mass_kg": .00625, "step_dt_s": model.opt.timestep,
                  "integrator": int(model.opt.integrator)},
        "entry": {"cube_xyz_m": np.round(cube_start, 5).tolist(),
                  "cube_free_joint": True, "initial_pad_contacts": entry_contacts,
                  "min_pad_center_to_cube_center_m": round(entry_pad_distance, 5),
                  "outside_open_aperture": True},
        "terminal_pose_aperture": aperture,
        "raw_servo_candidate_teacher": {
            "source": "saved reached robot qpos targets, right jaw commanded to .5 rad in final 2 s",
            "trace_duration_s": round(len(qpos) * step_dt, 3),
            "touch_frames": touch_frames, "pad_contact_frames": pad_frames,
            "first_touch_s": first_contact,
            "both_pad_frames": both_frames, "both_pad_lifted_frames": lifted_both_frames,
            "max_free_cube_center_z_m": round(max_cube_z, 5),
            "min_pelvis_z_m": round(min_pelvis_z, 5),
            "two_pad_lift_pass": bool(lifted_both_frames >= 50),
            "samples": samples},
        "next_native_gate": "Run a source-matched MJLab/BAM free-object reset and object-aware closed-loop teacher; require two opposing pad contacts, cube rise >= 4 cm for >= 1 s, stable support and rated actuator duty before imitation collection.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "entry": report["entry"],
                      "aperture_mm": {k: v["projected_free_gap_mm"] for k,v in aperture.items()},
                      "teacher": {k:v for k,v in report["raw_servo_candidate_teacher"].items() if k != "samples"}}, indent=2))


if __name__ == "__main__":
    main()
