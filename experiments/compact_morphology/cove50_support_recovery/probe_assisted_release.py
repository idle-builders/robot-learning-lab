"""One nominal, paired, upright release feasibility visit per side.

This is an authored diagnostic with a simulator-applied pelvis wrench. It does
not train a policy, and assisted motion can never count as a skill score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import (
    clip_frame50_leg_targets, configure_frame50_motors,
    give_soles_contact_priority, require_direct_frame,
)
from release_reference_action import ReleaseReferenceActionCfg
from train_release_residual_v19 import (
    CoveSupportCycleCommandCfg, support_action_over_limit,
)


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261015/support-assist-feasibility/method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def ramp(t: float, rise: float, hold: float, fall: float) -> float:
    if t < rise:
        return smooth((t - 2.15) / (rise - 2.15))
    if t < hold:
        return 1.0
    return 1.0 - smooth((t - hold) / (fall - hold))


def sole_corners(model: mujoco.MjModel, data: mujoco.MjData,
                 qpos: np.ndarray, side: str) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    geom = model.geom(f"{side}_sole").id
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
        raise RuntimeError("Physical corner score requires a box sole")
    sx, sy, sz = model.geom_size[geom]
    local = np.asarray([(x, y, -sz) for x in (-sx, sx) for y in (-sy, sy)])
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def longest(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def source_check(method: dict) -> None:
    if method["schema"] != "robot-learning-lab.support-assist-feasibility.v2":
        raise RuntimeError("Wrong method schema")
    for rel, expected in method["source_sha256"].items():
        path = ROOT / rel
        if sha(path) != expected:
            raise RuntimeError(f"Source hash changed: {rel}")
    if method["body_sha256"] != method["source_sha256"][method["body"]]:
        raise RuntimeError("Body identity differs inside method")
    if method["visits"] != [
        {"seed": 71701, "side": "right"},
        {"seed": 71702, "side": "left"},
    ]:
        raise RuntimeError("Method must contain exactly one family per side")


def build_env(method: dict, side: str, seed: int) -> tuple[ManagerBasedRlEnv, mujoco.MjModel]:
    body = ROOT / method["body"]
    model = require_direct_frame(body)
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if model.nu != 21 or model.nq != 28 or "head_pitch" in names:
        raise RuntimeError("Expected Direct Frame 21-axis control body")
    pose_root = ROOT / "artifacts/ape-cove50-support-recovery-20261009"
    specs = {s: json.loads((pose_root / f"{s}-low-swing-endpoint.json").read_text())
             for s in ("right", "left")}
    poses = {s: [specs[s]["joints"][name] for name in names] for s in specs}
    z, roll, pitch = specs["right"]["root_xyz_roll_pitch"][2:5]
    cfg = load_env_cfg(ape.ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID, play=False)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    give_soles_contact_priority(cfg)
    cfg.scene.entities["robot"].collisions[0].friction = {
        r"^(left|right)_sole$": (1.0,),
    }
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    prior = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = ReleaseReferenceActionCfg(
        entity_name=prior.entity_name,
        actuator_names=prior.actuator_names,
        preserve_order=prior.preserve_order,
        scale=0.05, offset=0.0, clip=prior.clip,
        shift_roll_rad=0.35,
        reference_native_scale=prior.scale,
        reference_native_offset=prior.offset,
        reference_use_default_offset=prior.use_default_offset,
        raw_residual_clip=1.0,
    )
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(model.qpos0[2]))
    cfg.rewards["pose_track"].params.update({"pose_right": poses["right"],
                                              "pose_left": poses["left"]})
    cfg.rewards["gravity_track"].params.update({
        "gravity_right": (math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                          -math.cos(roll) * math.cos(pitch)),
        "gravity_left": (math.sin(pitch), math.sin(roll) * math.cos(pitch),
                         -math.cos(roll) * math.cos(pitch)),
    })
    cfg.rewards["action_over_limit"].func = support_action_over_limit
    cfg.rewards["action_over_limit"].params = {"overshoot": 0.1}
    cfg.events["set_flamingo_cycle_state"].params.update({
        "pose_right": poses["right"], "pose_left": poses["left"],
        "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.002,
        "standing_z_min": float(model.qpos0[2]),
        "standing_z_max": float(model.qpos0[2]),
        "tilt_noise": math.radians(0.25),
        "joint_noise_std": 0.0025,
        "in_pose_prob": 0.0,
    })
    cfg.commands["twist"] = CoveSupportCycleCommandCfg(
        **{**vars(cfg.commands["twist"]), "resampling_time_range": (100.0, 100.0)},
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="self_body_contact",
            primary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        ape.ContactSensorCfg(
            name="nonsole_ground_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(?!(?:left_sole|right_sole)$).*", entity="robot"),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    # Probe only: no optimizer, randomization or learned actor. The wrench is
    # written explicitly in visit() after a measured load entry.
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = 8.0
    env = ManagerBasedRlEnv(cfg=cfg, device=method["device"])
    env.reset()
    requested = 1.0 if side == "right" else -1.0
    for _ in range(32):
        if float(env.command_manager.get_term("twist").stance_side[0]) == requested:
            break
        env.reset()
    else:
        raise RuntimeError("Could not draw requested side in 32 resets")
    return env, model


def visit(method: dict, side: str, seed: int, assisted: bool, out: Path) -> dict:
    env, model = build_env(method, side, seed)
    robot = env.scene["robot"]
    sensors = env.scene.sensors
    stance_slot = 1 if side == "right" else 0
    swing = "left" if side == "right" else "right"
    sign = 1.0 if side == "right" else -1.0
    initial_qpos = env.sim.data.qpos[0].cpu().numpy().copy()
    initial_qvel = env.sim.data.qvel[0].cpu().numpy().copy()
    initial_hash = hashlib.sha256(initial_qpos.tobytes() + initial_qvel.tobytes()).hexdigest()
    audit = mujoco.MjData(model)
    initial_patch = sole_corners(model, audit, initial_qpos, side)
    rated = np.asarray([a.effort_limit for a in env.cfg.scene.entities["robot"].articulation.actuators])
    mass = float(model.body_mass.sum())
    pelvis_id = robot.body_names.index("pelvis")
    dt = env.step_dt
    if abs(dt - 0.02) > 1e-9:
        raise RuntimeError("Expected 50 Hz control")
    rows, qposes, qvels = [], [], []
    load_streak = 0
    load_attained = False
    load_latch_time_s = None
    peak_assist_n = 0.0
    # Append each scored native state before the next step. If the current
    # visit crashes, the initial state and every completed step remain local.
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.jsonl")
    partial.write_text(json.dumps({
        "schema": "robot-learning-lab.support-assist-partial-trace.v1",
        "seed": seed, "side": side, "assisted": assisted,
        "initial_state_sha256": initial_hash,
        "initial_qpos": initial_qpos.tolist(),
        "initial_qvel": initial_qvel.tolist(),
    }) + "\n")
    for step in range(400):
        t = step * dt
        # Same deterministic action in both paired visits. This ramp avoids
        # the V19 learned actor's abrupt full-authority onset at 2.0 s.
        action = torch.zeros((1, 7), device=env.device)
        fraction = ramp(t, rise=3.0, hold=3.65, fall=4.5)
        if load_attained:
            action[0, 4:] = torch.tensor([-1.0, 1.0, -1.0], device=env.device) * fraction
        assist_fraction = (smooth((t - load_latch_time_s) / 0.3)
                           * (1.0 - smooth((t - 4.0) / 0.5))) if load_latch_time_s is not None else 0.0
        force = torch.zeros((1, 1, 3), device=env.device)
        if assisted and load_attained:
            force[0, 0, 1] = -sign * mass * 9.81 * method["assist"]["lateral_bodyweight"] * assist_fraction
            force[0, 0, 2] = mass * 9.81 * method["assist"]["vertical_bodyweight"] * assist_fraction
        peak_assist_n = max(peak_assist_n, float(force.norm()))
        robot.write_external_wrench_to_sim(
            force, torch.zeros_like(force), body_ids=[pelvis_id],
        )
        with torch.inference_mode():
            _, _, terminated, time_out, _ = env.step(action)
        qpos = env.sim.data.qpos[0].cpu().numpy().copy()
        qvel = env.sim.data.qvel[0].cpu().numpy().copy()
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            with partial.open("a") as stream:
                stream.write(json.dumps({"step": step + 1, "nonfinite_qpos": qpos.tolist(),
                                         "nonfinite_qvel": qvel.tolist()}, allow_nan=True) + "\n")
            raise RuntimeError(f"Nonfinite native state at step {step + 1}")
        stance_patch = sole_corners(model, audit, qpos, side)
        swing_patch = sole_corners(model, audit, qpos, swing)
        slip = float(np.linalg.norm(stance_patch[:, :2] - initial_patch[:, :2], axis=1).max())
        corner_z = float(swing_patch[:, 2].min())
        feet = [bool(v) for v in sensors["feet_ground_contact"].data.found[0].cpu().tolist()]
        # This sensor uses reduce="netforce", so its z component is world-frame
        # vertical ground reaction. Use magnitude to avoid depending on the
        # primary/secondary sign convention; tangential shear counts as zero.
        normals = sensors["feet_ground_contact"].data.force[0, :, 2].abs().cpu().numpy()
        load = float(normals[stance_slot] / max(1.0, normals.sum()))
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        upright_cos = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        tilt = math.degrees(math.acos(float(np.clip(upright_cos, -1.0, 1.0))))
        effort = robot.data.actuator_force[0].abs().cpu().numpy()
        rated_leg = bool(np.any(effort[:10] >= 0.995 * rated[:10]))
        rated_arm = bool(np.any(effort[10:20] >= 0.995 * rated[10:20]))
        bad = (bool(sensors["nonsole_ground_contact"].data.found[0].any())
               or bool(sensors["self_body_contact"].data.found[0].any())
               or float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()) >= 1.0)
        height = float(robot.data.root_link_pos_w[0, 2])
        safe = height >= 0.20 and tilt <= 15.0 and slip <= 0.015 and not bad
        this_load = (1.8 <= (step + 1) * dt <= 2.2 and safe and
                     feet == [True, True] and load >= 0.75)
        load_streak = load_streak + 1 if this_load else 0
        load_attained |= load_streak >= 10
        if load_attained and load_latch_time_s is None:
            load_latch_time_s = (step + 1) * dt
        clean_release = (load_attained and 2.0 <= (step + 1) * dt < 5.0 and safe
                         and feet == ([False, True] if side == "right" else [True, False])
                         and normals[stance_slot] >= 8.0 and corner_z >= 0.025)
        rows.append({
            "step": step + 1, "time_s": round((step + 1) * dt, 3),
            "load_fraction": load, "load_attained": load_attained,
            "load_latch_time_s": load_latch_time_s,
            "normal_loads_n": normals.tolist(),
            "feet_grounded": feet, "minimum_swing_corner_z_m": corner_z,
            "stance_patch_slip_m": slip, "pelvis_z_m": height, "tilt_deg": tilt,
            "rated_leg": rated_leg, "rated_arm": rated_arm,
            "forbidden_contact": bad, "clean_release": clean_release,
            "assist_force_n": force[0, 0].cpu().tolist(),
            "terminated": bool(terminated[0]), "time_out": bool(time_out[0]),
        })
        qposes.append(qpos)
        qvels.append(qvel)
        with partial.open("a") as stream:
            stream.write(json.dumps({"row": rows[-1], "qpos": qpos.tolist(),
                                     "qvel": qvel.tolist()}) + "\n")
        if bool(terminated[0]) or (bool(time_out[0]) and step < 399):
            break
    robot.write_external_wrench_to_sim(
        torch.zeros((1, 1, 3), device=env.device),
        torch.zeros((1, 1, 3), device=env.device), body_ids=[pelvis_id],
    )
    clean_s = longest([r["clean_release"] for r in rows]) * dt
    report = {
        "schema": "robot-learning-lab.support-assist-native-visit.v2",
        "classification": "assisted_feasibility_only" if assisted else "zero_assist_control_only",
        "skill_score": None, "optimizer_updates": 0,
        "body_sha256": method["body_sha256"], "method_sha256": sha(METHOD),
        "source_sha256": method["source_sha256"],
        "device": method["device"], "seed": seed, "side": side,
        "initial_state_sha256": initial_hash,
        "assisted": assisted, "peak_assist_force_n": peak_assist_n,
        "load_attained": load_attained, "load_latch_time_s": load_latch_time_s,
        "steps": len(rows),
        "longest_clean_release_s": round(clean_s, 3),
        "half_second_clean_release": clean_s >= 0.5,
        "max_stance_patch_slip_m": max(r["stance_patch_slip_m"] for r in rows),
        "rated_leg_fraction": sum(r["rated_leg"] for r in rows) / len(rows),
        "rated_arm_fraction": sum(r["rated_arm"] for r in rows) / len(rows),
        "forbidden_contact_frames": sum(r["forbidden_contact"] for r in rows),
        "ended_early": len(rows) < 400,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    trace = out.with_suffix(".npz")
    np.savez_compressed(trace, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                        qpos=np.asarray(qposes), qvel=np.asarray(qvels), dt_s=dt)
    report["trace_sha256"] = sha(trace)
    out.write_text(json.dumps(report, indent=2) + "\n")
    partial.unlink()
    env.close()
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=Path, default=METHOD)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.method.resolve() != METHOD.resolve():
        p.error("Use the frozen method path")
    method = json.loads(METHOD.read_text())
    source_check(method)
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("Output directory is not empty; use a fresh run directory")
    args.out.mkdir(parents=True, exist_ok=True)
    run_status = args.out / "run-status.json"
    run_status.write_text(json.dumps({
        "schema": "robot-learning-lab.support-assist-run-status.v1",
        "complete": False, "expected_visits": 4, "claim": "no skill score",
    }, indent=2) + "\n")
    summary = []
    completed_visits = 0
    for family in method["visits"]:
        reports = []
        for assisted, label in ((False, "zero"), (True, "assisted")):
            current = {"seed": family["seed"], "side": family["side"],
                       "assisted": assisted, "label": label}
            run_status.write_text(json.dumps({
                "schema": "robot-learning-lab.support-assist-run-status.v1",
                "complete": False, "expected_visits": 4,
                "completed_visits": completed_visits, "current_visit": current,
                "claim": "no skill score",
            }, indent=2) + "\n")
            try:
                reports.append(visit(method, family["side"], family["seed"], assisted,
                                     args.out / f'{family["side"]}-{label}.json'))
            except Exception as exc:
                run_status.write_text(json.dumps({
                    "schema": "robot-learning-lab.support-assist-run-status.v1",
                    "complete": False, "expected_visits": 4,
                    "completed_visits": completed_visits, "failed_visit": current,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "claim": "incomplete feasibility diagnostic; no skill score",
                }, indent=2) + "\n")
                raise
            completed_visits += 1
        if reports[0]["initial_state_sha256"] != reports[1]["initial_state_sha256"]:
            raise RuntimeError(f"Paired {family['side']} physical starts differ")
        summary.append({
            "side": family["side"], "seed": family["seed"],
            "same_initial_state": True,
            "zero_assist_clean_s": reports[0]["longest_clean_release_s"],
            "assisted_clean_s": reports[1]["longest_clean_release_s"],
            "assisted_load_attained": reports[1]["load_attained"],
        })
    (args.out / "summary.json").write_text(json.dumps({
        "schema": "robot-learning-lab.support-assist-feasibility-summary.v2",
        "complete": True,
        "classification": "four bounded nominal feasibility visits; no skill score",
        "method_sha256": sha(METHOD), "optimizer_updates": 0,
        "pairs": summary,
    }, indent=2) + "\n")
    run_status.write_text(json.dumps({
        "schema": "robot-learning-lab.support-assist-run-status.v1",
        "complete": True, "expected_visits": 4, "completed_visits": 4,
        "claim": "feasibility only; no skill score",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
