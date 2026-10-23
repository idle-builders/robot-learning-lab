"""Screen representative saved-policy leg poses against the bare CAD hardware.

This is a pose sample, not a swept-volume or tolerance certificate. It catches
hard contradictions between the replay and the mechanism before calling a gait
transfer successful. CAD coordinates are right/rear/up; MJCF is forward/left/up.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import build123d as bd
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hip_knee_package import (
    turn_knee_side,
)
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import (
    candidate, intersection,
)
from experiments.compact_morphology.ape_frame50.shin_ankle_package import (
    foot, package,
)
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_design.swarm_cove_hip import build as old

RUN = Path("artifacts/ape-frame50-supported-body/learned-replay")
JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")


def bbox_overlap(a: bd.Shape, b: bd.Shape) -> bool:
    aa, bb = a.bounding_box(), b.bounding_box()
    return all(getattr(aa.min, c) < getattr(bb.max, c) - .05 and
               getattr(bb.min, c) < getattr(aa.max, c) - .05 for c in "XYZ")


def hard_hit(a: bd.Shape, b: bd.Shape) -> float:
    return intersection(a, b) if bbox_overlap(a, b) else 0.0


def sample_frames(qpos: np.ndarray, model: mujoco.MjModel) -> list[int]:
    frames = {0, len(qpos)//4, len(qpos)//2, 3*len(qpos)//4, len(qpos)-1}
    for side in ("left", "right"):
        for joint in JOINTS:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}")
            values = qpos[:, model.jnt_qposadr[j]]
            frames.update((int(np.argmin(values)), int(np.argmax(values))))
    return sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", default="artifacts/ape-frame50-supported-body/ape-frame50-supported-body.xml")
    parser.add_argument("--state", default=str(RUN / "forward-9mm-belt.npz"))
    parser.add_argument("--metrics", default=str(RUN / "forward-9mm-belt.json"))
    parser.add_argument("--out", default=str(RUN / "hardware-pose-screen.json"))
    poses_group = parser.add_mutually_exclusive_group()
    poses_group.add_argument("--candidate-envelope", action="store_true",
                             help="screen a finite candidate joint-angle grid rather than saved policy poses")
    poses_group.add_argument("--pose-file", help="JSON list of named bilateral joint poses in degrees")
    parser.add_argument("--all-frames", action="store_true",
                        help="screen every saved world-0 frame rather than selected extrema")
    args = parser.parse_args()
    body, state, metrics, out = (ROOT / getattr(args, key) for key in
                                 ("body_xml", "state", "metrics", "out"))
    model = mujoco.MjModel.from_xml_path(str(body))
    pose_labels = None
    pose_file = ROOT / args.pose_file if args.pose_file else None
    if pose_file:
        entries = json.loads(pose_file.read_text())["poses"]
        qpos = np.repeat(model.qpos0[None, :], len(entries), axis=0)
        for frame, entry in enumerate(entries):
            for side in ("left", "right"):
                for name, value in entry[side].items():
                    if name not in JOINTS:
                        raise ValueError(f"unknown lower-body joint: {side}_{name}")
                    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          f"{side}_{name}")
                    qpos[frame, model.jnt_qposadr[j]] = np.radians(value)
        samples = list(range(len(entries)))
        pose_labels = [entry["name"] for entry in entries]
        source_metrics = None
        dt = None
    elif args.candidate_envelope:
        poses = [(yaw, roll, pitch, 70, -25) for yaw, roll, pitch in
                 itertools.product((-20, 0, 20), (-12, 0, 12), (-45, 0, 15))]
        poses += [(yaw, roll, pitch, knee, ankle) for yaw, roll, pitch, knee, ankle in
                  itertools.product((-20, 20), (-12, 12), (-45, 15), (0, 105), (-25, 25))]
        qpos = np.repeat(model.qpos0[None, :], len(poses), axis=0)
        for frame, values in enumerate(poses):
            for side in ("left", "right"):
                for name, value in zip(JOINTS, values):
                    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          f"{side}_{name}")
                    qpos[frame, model.jnt_qposadr[j]] = np.radians(value)
        samples = list(range(len(poses)))
        source_metrics = None
        dt = None
    else:
        trace = np.load(state)
        source_metrics = json.loads(metrics.read_text())
        qpos = trace["qpos"]
        samples = list(range(len(qpos))) if args.all_frames else sample_frames(qpos, model)
        dt = float(trace["dt"])
    hw, shared = candidate()
    thigh, shin, fixed_ankle = package()
    # The CAD source is authored on its +X side, then mirrored for the
    # simulator's right side. Both MJCF ankle joints use the same positive
    # pitch convention; mirroring the source foot already accounts for the
    # side change, so the foot(angle) input must not be signed again.
    sole_marker_local = np.array((.012, 0, -.039))
    pose_mapping_error_mm = {}
    for side, sign in (("left", 1), ("right", -1)):
        probe = mujoco.MjData(model)
        probe.qpos[:] = model.qpos0
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"{side}_ankle_pitch")
        probe.qpos[model.jnt_qposadr[joint_id]] = np.radians(10)
        mujoco.mj_forward(model, probe)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{side}_foot")
        sim_marker = (probe.xpos[body_id] +
                      probe.xmat[body_id].reshape(3, 3) @ sole_marker_local)
        source_marker = turn_knee_side(foot(10)["sole_plate"], 0, sign)
        source_center = source_marker.center()
        cad_marker = np.array((-source_center.Y, source_center.X,
                               source_center.Z)) / 1000
        error_mm = float(np.linalg.norm(sim_marker - cad_marker) * 1000)
        pose_mapping_error_mm[side] = round(error_mm, 4)
        if error_mm > .5:
            raise ValueError(f"CAD/MJCF {side} ankle pose map differs by {error_mm:.2f} mm")
    addresses = {
        f"{side}_{joint}": int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}")])
        for side in ("left", "right") for joint in JOINTS
    }
    allowed_ankle = {
        ("output_shaft", "bearing_A"), ("output_shaft", "bearing_B"),
        ("output_shaft", "seat_A"), ("output_shaft", "seat_B"),
        ("motor_spline_adapter_reserve", "motor_case"),
        ("motor_crank", "motor_case"),
    }
    reports = []
    for frame in samples:
        per_side = {}
        for side, sign in (("left", 1), ("right", -1)):
            q = {name: float(np.degrees(qpos[frame, addresses[f"{side}_{name}"]]))
                 for name in JOINTS}
            # The right-side Cove source geometry becomes the simulator's
            # left leg. CAD rear is -sim forward; CAD lateral is sim left.
            yaw, roll, pitch = q["hip_yaw"] / sign, -q["hip_roll"], q["hip_pitch"]
            stages = hip.frame(hw, sign, yaw, roll, pitch)
            support = {**shared, **{f"fixed_{n}": s for n, s in stages["fixed"].items()},
                       **{f"yaw_{n}": s for n, s in stages["yaw"].items()},
                       **{f"roll_{n}": s for n, s in stages["roll"].items()}}
            hits = []
            stationary = {**shared, **stages["fixed"]}
            for stage, group in stages.items():
                if stage == "fixed":
                    continue
                for name, moving in group.items():
                    for other, fixed in stationary.items():
                        if (name == "yaw_spindle" and other in (
                            "yaw_bearing_low_F606_ZZ", "yaw_bearing_high_F606_ZZ",
                            "yaw_housing_tube")) or (name == "yaw_driven_20T" and
                                                      other == "yaw_belt_123_3MGT_15"):
                            continue
                        amount = hard_hit(moving, fixed)
                        if amount > .05:
                            hits.append([f"hip_{stage}_{name}", other,
                                         round(amount, 3)])
            # A moving-versus-fixed screen missed the full yaw spindle
            # passing through the roll spindle and pitch-fork bridge. Check
            # the nonmating neighboring stages as well; a journal should
            # occupy its bearing bore, not a second solid part.
            for parent, child in (("yaw", "roll"), ("yaw", "thigh"),
                                  ("roll", "thigh")):
                for parent_name, parent_part in stages[parent].items():
                    for child_name, child_part in stages[child].items():
                        amount = hard_hit(parent_part, child_part)
                        if amount > .05:
                            hits.append([f"hip_{parent}_{parent_name}",
                                         f"hip_{child}_{child_name}", round(amount, 3)])
            for prefix, group in (("shin", shin), ("ankle", fixed_ankle),
                                  ("foot", foot(q["ankle_pitch"]))):
                for name, raw in group.items():
                    bent = turn_knee_side(raw, q["knee"], sign)
                    shape = old.transform(bent, q["hip_yaw"], roll, pitch,
                                          "thigh", sign)
                    for other, fixed in support.items():
                        amount = hard_hit(shape, fixed)
                        if amount > .05:
                            hits.append([f"{prefix}_{name}", other, round(amount, 3)])
            # Check the foot against its own ankle carrier at the observed
            # relative pitch, excluding the declared shaft/bearing mates.
            for name, moving in foot(q["ankle_pitch"]).items():
                for other, fixed in fixed_ankle.items():
                    if (name, other) in allowed_ankle:
                        continue
                    amount = hard_hit(moving, fixed)
                    if amount > .05:
                        hits.append([f"foot_{name}", f"ankle_{other}",
                                     round(amount, 3)])
            per_side[side] = {"joint_deg": {n: round(v, 3) for n, v in q.items()},
                              "hard_hits": hits}
        reports.append({"frame": frame, "name": pose_labels[frame] if pose_labels else None,
                        "time_s": round(frame * dt, 3) if dt else None,
                        "sides": per_side})
    result = {
        "stage": ("named bilateral poses against exposed lower hardware" if pose_file else
                  "finite candidate joint-angle grid against exposed lower hardware" if args.candidate_envelope else
                  "selected actual policy poses against exposed lower hardware"),
        "body_xml": str(body.relative_to(ROOT)),
        "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "trace": None if (args.candidate_envelope or pose_file) else str(state.relative_to(ROOT)),
        "trace_source_body_sha256": source_metrics["body_sha256"] if source_metrics else None,
        "trace_on_exact_body": (source_metrics["body_sha256"] == hashlib.sha256(body.read_bytes()).hexdigest()
                                if source_metrics else None),
        "pose_file": str(pose_file.relative_to(ROOT)) if pose_file else None,
        "pose_file_sha256": hashlib.sha256(pose_file.read_bytes()).hexdigest() if pose_file else None,
        "sampled_frames": len(samples),
        "ankle_pose_mapping_probe_10deg_error_mm": pose_mapping_error_mm,
        "scope": ("named bilateral lower-body poses; hip moving versus pelvis/fixed and adjacent nonmating hip stages, shin/ankle/foot versus shared and hip fixed/yaw/roll, and foot versus ankle carrier" if pose_file else
                  "27 yaw/roll/pitch combinations at knee 70, ankle -25, plus 32 extreme combinations at knee 0/105 and ankle +/-25; both legs mirror the same grid"
                  if args.candidate_envelope else
                  ("world 0; every saved frame" if args.all_frames else
                   "world 0; endpoints of each leg joint plus five timeline frames") +
                  "; hip moving versus pelvis/fixed and adjacent nonmating hip stages, shin/ankle/foot versus shared and hip fixed/yaw/roll, and foot versus ankle carrier"),
        "not_proved": "all worlds, continuous sweeps, simultaneous bilateral/upper collision, force, tolerance, fasteners and wire routing",
        "reports": reports,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": len(samples),
                      "hard_hits": sum(len(side["hard_hits"]) for row in reports
                                       for side in row["sides"].values()),
                      "output": str(out.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
