"""Train and evaluate supported stance, posture, bilateral-arm, and independent-arm transitions on selected R3.

Uses the exact 34.0 s (1,700-control) multi-stage transition schedule and four start
families (145001..145004) from data/basic-skills-retry-20261027/handoff-harness/proposal-config.json:
  - Ticks 0..100 (0-2 s): Walker quiet stand (candidate200 467b12a1)
  - Ticks 100..600 (2-12 s): Shallow posture crouch, hold, and return (CommandActor d2f5f40e)
  - Ticks 600..1100 (12-22 s): Bilateral arm command raise, hold, and return (CommandNet 23760523, EMA 0.20)
  - Ticks 1100..1700 (22-34 s): Independent overlapping left/right arm command (CommandNet 23760523, EMA 0.05)

Couples knee-extension forward COM compensation (-0.055 rad hip pitch and +0.18 rad shoulder pitch
scaled by knee extension relative to Walker's 0.36 rad bent-knee stance) with the 120-input 22-DOF
causal IMU/load sensor-feedback residual actor, trained for 12 PPO rounds (240 Adam updates).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/supported-skill-transitions-20261030"

sys.path.insert(0, str(ROOT / "robot/model/build"))
sys.path.insert(0, str(ROOT / "control/skills/basic-skills"))
import yaml
from runtime import Runtime
from arm_command import CommandNet

STAND_METHOD_PATH = ROOT / "data/basic-skills-retry-20261027/stand/method.json"
ENV_YAML_PATH = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml"


class MaterializedWalkingInterface:
    """Identical tensor mapping to walk-resume/policy/adapter.py:Interface without reloading unmaterialized XMLs."""

    def __init__(self):
        method = json.loads(STAND_METHOD_PATH.read_text())
        audit = method["posture"]
        self.source_names = audit["source_order"]
        self.rows = audit["mapping"]
        self.current_names = [r["current_name"] for r in self.rows]
        self.new_names = [r["current_name"] for r in self.rows if r["source_index"] is None]
        home_arr = np.array(
            [
                -0.2 if n.endswith(("hip_pitch", "ankle_pitch")) else 0.4 if n.endswith("knee") else 0.0
                for n in self.source_names
            ],
            dtype=np.float32,
        )
        self.home = torch.tensor(home_arr, dtype=torch.float32)
        self.new_rows = {n: 21 + i for i, n in enumerate(self.new_names)}
        saved_env = yaml.load(ENV_YAML_PATH.read_text(), Loader=yaml.BaseLoader)
        clips = saved_env["actions"]["joint_pos"]["clip"]
        self.source_clips = {n.strip("^$"): tuple(map(float, limits)) for n, limits in clips.items()}
        self._device_tensors = {}

    def tensors(self, device):
        if str(device) not in self._device_tensors:
            indices = [0] * 21
            signs = [0.0] * 21
            for r in self.rows:
                if r["source_index"] is not None:
                    indices[r["source_index"]] = r["current_index"]
                    signs[r["source_index"]] = r["sign"]
            action_ids = [
                r["source_index"] if r["source_index"] is not None else self.new_rows[r["current_name"]]
                for r in self.rows
            ]
            action_scale = [
                r["sign"] * 0.3
                if r["source_index"] is not None and r["source_index"] < 10
                else 0.3
                if r["source_index"] is None
                else 0.0
                for r in self.rows
            ]
            action_offset = [
                r["sign"] * self.home[r["source_index"]].item() if r["source_index"] is not None else 0.0
                for r in self.rows
            ]
            bounds = []
            for r in self.rows:
                lower, upper = self.source_clips.get(r["source_name"], (-float("inf"), float("inf")))
                bounds.append((lower, upper) if r.get("sign", 1.0) > 0 else (-upper, -lower))
            self._device_tensors[str(device)] = dict(
                qids=torch.tensor(indices, device=device),
                signs=torch.tensor(signs, device=device),
                new_ids=torch.tensor([self.current_names.index(n) for n in self.new_names], device=device),
                action_ids=torch.tensor(action_ids, device=device),
                scale=torch.tensor(action_scale, device=device),
                offset=torch.tensor(action_offset, device=device),
                home=self.home.to(device),
                lower=torch.tensor([x[0] for x in bounds], device=device),
                upper=torch.tensor([x[1] for x in bounds], device=device),
            )
        return self._device_tensors[str(device)]

    def observe(self, q, qd, gyro, gravity, previous_raw, command):
        t = self.tensors(q.device)
        old_q = q[:, t["qids"]] * t["signs"]
        old_qd = qd[:, t["qids"]] * t["signs"]
        return torch.cat(
            (
                gyro,
                gravity,
                old_q - t["home"],
                old_qd,
                previous_raw[:, :21],
                command,
                q[:, t["new_ids"]],
                qd[:, t["new_ids"]],
                previous_raw[:, 21:],
            ),
            dim=1,
        )

    def target(self, raw):
        t = self.tensors(raw.device)
        return torch.clamp(raw[:, t["action_ids"]] * t["scale"] + t["offset"], t["lower"], t["upper"])


WalkingInterface = MaterializedWalkingInterface

POINTER_PATH = ROOT / "robot/model/current-model.json"
MATERIALIZED_MJCF = (
    ROOT
    / ".cache/work/cove50-forward-stop-20261030-r1-774a216e07ed/display-source-v2"
    / "artifacts/robot/model/timestep/cove50-rigid.xml"
)
WALKER_CKPT_PATH = (
    ROOT
    / ".cache/forward-stop-source-preparation-20261030/input-objects"
    / "467b12a17a71f06d951294b3d9185503ea78c6edda431ae7e7870aa648fc02bd"
)
POSTURE_CKPT_PATH = (
    ROOT
    / ".cache/work/cove50-coordinated-arm-feedback-next-20261030-r1-b35ee2f11a01/preparation/input-objects"
    / "d2f5f40efd5e91a5ca2fbd837ac5286e7b515bacf66492509757e317c86a7e16"
)
GRU_CKPT_PATH = (
    ROOT
    / "data/basic-skills-retry-20261027/arm-body-learning-harness/input-snapshot/objects"
    / "23760523ead3e54bc0a804d7b2f357902eb61c8526fd54fa2c529cdeb81a350c"
)
WARM_FB_CKPT_PATH = (
    ROOT
    / "artifacts/runs/cove50-coordinated-arm-feedback-next-20261030-r1-b35ee2f11a01/evidence/train/candidate.pt"
)
PAWS = ("ape_body_shell__left_smooth_paw", "ape_body_shell__right_smooth_paw")

FAMILIES = [
    {"seed": 145001, "label": "nominal", "joint_sigma_rad": 0.0, "rpy_envelope_rad": 0.0, "joint_velocity_sigma_rad_s": 0.0},
    {"seed": 145002, "label": "small", "joint_sigma_rad": 0.002, "rpy_envelope_rad": 0.004, "joint_velocity_sigma_rad_s": 0.0},
    {"seed": 145003, "label": "moderate", "joint_sigma_rad": 0.005, "rpy_envelope_rad": 0.01, "joint_velocity_sigma_rad_s": 0.005},
    {"seed": 145004, "label": "wider_small", "joint_sigma_rad": 0.008, "rpy_envelope_rad": 0.015, "joint_velocity_sigma_rad_s": 0.01},
]


def ref(path: Path) -> dict:
    p = path.resolve()
    with p.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    try:
        rel = p.relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(p)
    return {"path": rel, "sha256": sha, "bytes": p.stat().st_size}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def dist_summary(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=float)
    q = np.quantile(arr, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]).tolist()
    return dict(zip(("min", "p10", "p50", "p90", "p99", "max"), q))


def make_posture_net() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 22),
    )


def actor_forward(sd: dict[str, torch.Tensor], obs120: torch.Tensor) -> torch.Tensor:
    x = (obs120 - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 0.01)
    for layer in (0, 2, 4):
        x = F.linear(x, sd[f"mlp.{layer}.weight"], sd[f"mlp.{layer}.bias"])
        if layer != 4:
            x = F.elu(x)
    return x


def critic_forward(csd: dict[str, torch.Tensor], obs120: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros((obs120.shape[0], 174 - 120), dtype=obs120.dtype, device=obs120.device)
    x174 = torch.cat([obs120, pad], dim=1)
    x = (x174 - csd["obs_normalizer._mean"]) / (csd["obs_normalizer._std"] + 0.01)
    for layer in (0, 2, 4):
        x = F.linear(x, csd[f"mlp.{layer}.weight"], csd[f"mlp.{layer}.bias"])
        if layer != 4:
            x = F.elu(x)
    return x


def walker_step(
    mapping,
    w_sd: dict[str, torch.Tensor],
    pkt: dict | None,
    prev_raw24: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if pkt is None:
        obs81 = np.zeros(81, dtype=np.float32)
    else:
        v = pkt["values"]
        obs81 = (
            mapping.observe(
                torch.tensor(v[:22][None], dtype=torch.float32),
                torch.tensor(v[22:44][None], dtype=torch.float32),
                torch.tensor(v[44:47][None], dtype=torch.float32),
                torch.tensor(v[47:50][None], dtype=torch.float32),
                torch.tensor(prev_raw24[None], dtype=torch.float32),
                torch.zeros(1, 3),
            )[0]
            .numpy()
            .astype(np.float32)
        )
    with torch.inference_mode():
        x = (torch.tensor(obs81[None], dtype=torch.float32) - w_sd["obs_normalizer._mean"]) / (
            w_sd["obs_normalizer._std"] + 0.01
        )
        for layer in (0, 2, 4, 6):
            x = F.linear(x, w_sd[f"mlp.{layer}.weight"], w_sd[f"mlp.{layer}.bias"])
            if layer != 6:
                x = F.elu(x)
        target22 = mapping.target(x)[0].numpy().astype(np.float32)
        raw24 = x[0].numpy().astype(np.float32)
    return raw24, target22


def posture_beta_for_local_tick(local_tick: int) -> float:
    """Local tick = global_tick - 100 (0..499):
    0..100: 0.0
    100..200: cosine ramp 0 -> 1
    200..300: hold 1.0
    300..400: cosine ramp 1 -> 0
    400..499: tail 0.0
    """
    if local_tick < 100:
        return 0.0
    if local_tick < 200:
        u = (local_tick - 100) / 100.0
        return float(0.5 - 0.5 * np.cos(np.pi * u))
    if local_tick < 300:
        return 1.0
    if local_tick < 400:
        u = (local_tick - 300) / 100.0
        return float(0.5 + 0.5 * np.cos(np.pi * u))
    return 0.0


def arm_flags_for_global_tick(c: int) -> tuple[float, float, float]:
    """Returns (flag_L, flag_R, ema_coeff) for c >= 600:
    600..1100: bilateral arms (EMA 0.20), flags raised 700..900 (14-18 s)
    1100..1700: independent arms (EMA 0.05), L raised 1200..1400 (24-28 s), R raised 1300..1500 (26-30 s)
    """
    if c < 1100:
        f = 1.0 if 700 <= c < 900 else 0.0
        return f, f, 0.20
    f_l = 1.0 if 1200 <= c < 1400 else 0.0
    f_r = 1.0 if 1300 <= c < 1500 else 0.0
    return f_l, f_r, 0.05


def transition_corrective_raw_target(v: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Bounded causal sensor-feedback target in raw space (tanh(raw)*0.03 = delta_fb)."""
    q_del = v[:22]
    qd_del = v[22:44]
    gyro = v[44:47]
    grav = v[47:50]
    arm_L = float(abs(q_ref[4]))
    arm_R = float(abs(q_ref[0]))
    mean_load = 0.5 * (arm_L + arm_R) / 1.176
    asym_load = 0.5 * (arm_L - arm_R) / 1.176
    pitch_trim = 0.012 * mean_load + 0.16 * grav[0] + 0.040 * gyro[1]
    roll_trim = 0.004 * asym_load + 0.10 * grav[1] - 0.025 * gyro[0]
    delta = np.zeros(22, dtype=np.float32)
    delta[17] = np.clip(pitch_trim, -0.026, 0.026)
    delta[20] = np.clip(pitch_trim, -0.026, 0.026)
    delta[11] = np.clip(0.65 * pitch_trim, -0.020, 0.020)
    delta[14] = np.clip(0.65 * pitch_trim, -0.020, 0.020)
    delta[18] = np.clip(-roll_trim, -0.018, 0.018)
    delta[21] = np.clip(+roll_trim, -0.018, 0.018)
    delta[[0, 4]] = np.clip(0.18 * (q_ref[[0, 4]] - q_del[[0, 4]]) - 0.015 * qd_del[[0, 4]], -0.022, 0.022)
    ratio = np.clip(delta / 0.03, -0.92, 0.92)
    return np.arctanh(ratio).astype(np.float32)


def knee_extension_com_trim(q_ref: np.ndarray) -> np.ndarray:
    """Forward COM compensation when transitioning from Walker's bent-knee stance (~0.36 rad)
    to straighter posture/arm references so the torso does not pitch backward over the heels.
    """
    mean_knee_ref = 0.5 * float(q_ref[16] + q_ref[19])
    knee_ext = float(np.clip((0.36 - mean_knee_ref) / 0.36, 0.0, 1.0))
    trim = np.zeros(22, dtype=np.float32)
    trim[11] = -0.055 * knee_ext
    trim[14] = -0.055 * knee_ext
    trim[0] = +0.18 * knee_ext
    trim[4] = +0.18 * knee_ext
    return trim


_WORKER_CACHE = {}


def _init_worker():
    torch.set_num_threads(1)
    m = mujoco.MjModel.from_xml_path(str(MATERIALIZED_MJCF))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)

    mapping = WalkingInterface()
    w_ckpt = torch.load(WALKER_CKPT_PATH, map_location="cpu", weights_only=False)
    w_sd = {k: v.detach().cpu().contiguous().clone() for k, v in w_ckpt["actor_state_dict"].items()}

    p_ckpt = torch.load(POSTURE_CKPT_PATH, map_location="cpu", weights_only=False)
    p_net = make_posture_net()
    p_net.load_state_dict({k: v.detach().cpu().contiguous().clone() for k, v in p_ckpt["actor_state_dict"].items()})
    p_net.eval()

    g_ckpt = torch.load(GRU_CKPT_PATH, map_location="cpu", weights_only=False)
    g_net = CommandNet()
    g_net.load_state_dict({k: v.detach().cpu().contiguous().clone() for k, v in g_ckpt["model_state_dict"].items()})
    g_net.eval()

    paw_geoms = [m.geom(n).id for n in PAWS]
    paw_centers = []
    for geom in paw_geoms:
        mesh = int(m.geom_dataid[geom])
        start = int(m.mesh_vertadr[mesh])
        num = int(m.mesh_vertnum[mesh])
        verts = m.mesh_vert[start : start + num]
        paw_centers.append((verts.min(0) + verts.max(0)) / 2.0)

    _WORKER_CACHE["model"] = m
    _WORKER_CACHE["mapping"] = mapping
    _WORKER_CACHE["w_sd"] = w_sd
    _WORKER_CACHE["p_net"] = p_net
    _WORKER_CACHE["g_net"] = g_net
    _WORKER_CACHE["paw_geoms"] = paw_geoms
    _WORKER_CACHE["paw_centers"] = paw_centers


def run_transition_episode(spec: dict) -> dict:
    if "model" not in _WORKER_CACHE:
        _init_worker()
    m = _WORKER_CACHE["model"]
    mapping = _WORKER_CACHE["mapping"]
    w_sd = _WORKER_CACHE["w_sd"]
    p_net = _WORKER_CACHE["p_net"]
    g_net = _WORKER_CACHE["g_net"]
    paw_geoms = _WORKER_CACHE["paw_geoms"]
    paw_centers = _WORKER_CACHE["paw_centers"]

    seed = int(spec["seed"])
    controls_n = int(spec["controls"])
    controller = spec["controller"]
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in spec["actor_sd"].items()} if spec.get("actor_sd") else None
    explore_sigma = float(spec.get("explore_sigma", 0.0))
    save_trace_path = Path(spec["save_trace_path"]) if spec.get("save_trace_path") else None
    compressed_train = bool(spec.get("compressed_train", False))

    d = mujoco.MjData(m)
    rt = Runtime(m, d, rail_V=12.0, command_delay_s=0.02, feedback_delay_s=0.01, peak=False, seed=seed)
    rng = np.random.default_rng(seed)
    d.qpos[rt.qids] = rng.normal(0.0, float(spec["joint_sigma_rad"]), 22)
    for i, j in enumerate(rt.jids):
        if m.jnt_limited[j]:
            d.qpos[rt.qids[i]] = np.clip(d.qpos[rt.qids[i]], *m.jnt_range[j])
    rpy = rng.uniform(-float(spec["rpy_envelope_rad"]), float(spec["rpy_envelope_rad"]), 3)
    mujoco.mju_euler2Quat(d.qpos[3:7], rpy, "xyz")
    d.qvel[rt.vids] = rng.normal(0.0, float(spec["joint_velocity_sigma_rad_s"]), 22)
    mujoco.mj_forward(m, d)
    floor = m.geom("floor").id
    feet = [i for i in range(m.ngeom) if m.geom(i).name.startswith(("collision_R_foot_", "collision_L_foot_"))]
    d.qpos[2] += 0.0002 - min(mujoco.mj_geomDistance(m, d, floor, i, 1.0, None) for i in feet)
    mujoco.mj_forward(m, d)
    rt.target = d.qpos[rt.qids].copy()
    rt.previous_command = rt.target.copy()

    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()
    free_joints = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    root_body = int(m.jnt_bodyid[free_joints[0]])
    init_com_xy = d.subtree_com[root_body][:2].copy()
    initial_yaw = float(
        np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
    )

    prev_raw24 = np.zeros(24, dtype=np.float32)
    h = None
    filt = None
    prev_raw22 = np.zeros(22, dtype=np.float32)
    prev_qref = d.qpos[rt.qids].astype(np.float32).copy()
    prior_requested22 = rt.target.astype(np.float32).copy()
    walker_handoff_target22 = rt.target.astype(np.float32).copy()
    exp_rng = np.random.default_rng(seed + int(spec.get("round_idx", 0)) * 1009)

    native = {}
    controls = {}
    obs_list = []
    act_list = []
    target_raw_list = []
    rew_list = []
    first_event = None
    physics_steps = 0
    force_buf = np.zeros(6)
    t_start = time.monotonic()

    def add(dst, **vals):
        for k, val in vals.items():
            dst.setdefault(k, []).append(np.asarray(val).copy())

    def sample_state():
        nonlocal first_event
        load = np.zeros(2)
        other = 0.0
        selfdepth = 0.0
        for i, c_obj in enumerate(d.contact):
            if c_obj.geom1 == floor or c_obj.geom2 == floor:
                g = c_obj.geom2 if c_obj.geom1 == floor else c_obj.geom1
                name = m.geom(g).name
                mujoco.mj_contactForce(m, d, i, force_buf)
                normal = max(0.0, float(force_buf[0]))
                if name.startswith("collision_R_foot_"):
                    load[0] += normal
                elif name.startswith("collision_L_foot_"):
                    load[1] += normal
                else:
                    other += normal
            else:
                selfdepth = max(selfdepth, -float(c_obj.dist))
        tilt = float(np.arccos(np.clip(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1.0, 1.0)))
        yaw = float(
            np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2))
        )
        heading_err = float(np.arctan2(np.sin(initial_yaw - yaw), np.cos(initial_yaw - yaw)))
        if first_event is None and (tilt > 1.0 or d.qpos[2] < 0.15 or other > 1.0):
            first_event = float(d.time)
        paws = np.array([d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ c for g, c in zip(paw_geoms, paw_centers)])
        return dict(
            time_s=float(d.time),
            qpos=d.qpos,
            qvel=d.qvel,
            foot_normal_N=load,
            other_floor_normal_N=other,
            self_contact_depth_m=selfdepth,
            torso_tilt_rad=tilt,
            truth_heading_error_rad=heading_err,
            whole_body_com_m=d.subtree_com[root_body],
            sole_positions_m=np.array([d.site(s + "_sole_contact_frame").xpos for s in ("R", "L")]),
            paw_positions_m_L_R=paws,
            target22_rad=rt.target,
            torque22_Nm=rt.torque,
            current22_A=rt.current,
            saturated22=rt.saturated,
        )

    add(native, **sample_state())

    for step_i in range(controls_n):
        # During 300-step training episodes, map steps across the 1700-step transition schedule
        if compressed_train:
            if step_i < 100:
                c = step_i
            elif step_i < 200:
                c = 100 + (step_i - 100) * 5
            elif step_i < 250:
                c = 600 + (step_i - 200) * 10
            else:
                c = 1100 + (step_i - 250) * 12
        else:
            c = step_i

        pkt = rt.observation()
        stage_id = 0 if c < 100 else (1 if c < 600 else (2 if c < 1100 else 3))

        if c == 0:
            target = rt.target.copy()
            q_ref = target.astype(np.float32).copy()
            raw22 = np.zeros(22, dtype=np.float32)
            delta = np.zeros(22, dtype=np.float32)
            obs120 = np.zeros(120, dtype=np.float32)
        elif c < 100:
            raw24, w_target22 = walker_step(mapping, w_sd, pkt, prev_raw24)
            prev_raw24 = raw24.copy()
            target = w_target22.astype(float)
            q_ref = w_target22.copy()
            prev_qref = q_ref.copy()
            prior_requested22 = q_ref.copy()
            if c == 99:
                walker_handoff_target22 = (
                    pkt["values"][:22].astype(np.float32).copy() if pkt is not None else q_ref.copy()
                )
            raw22 = np.zeros(22, dtype=np.float32)
            delta = np.zeros(22, dtype=np.float32)
            obs120 = np.zeros(120, dtype=np.float32)
            rt.command(rt.names, target)
        else:
            # First-use bootstrap at ticks 100 and 600 in open-loop baseline
            if c in (100, 600) and controller == "open-loop-handoff":
                q_ref = prior_requested22.copy()
            elif c < 600:
                beta_val = posture_beta_for_local_tick(c - 100)
                with torch.inference_mode():
                    q_post = p_net(torch.tensor([[beta_val]], dtype=torch.float32))[0].numpy().astype(np.float32)
                if controller == "coordinated-transition-feedback" and c < 170:
                    # Smooth 1.4 s (70-control) cosine handoff from attained Walker state + knee-extension COM trim
                    u = (c - 100 + 1) / 70.0
                    alpha_h = float(0.5 - 0.5 * np.cos(np.pi * u))
                    q_ref = (1.0 - alpha_h) * walker_handoff_target22 + alpha_h * (q_post + knee_extension_com_trim(q_post))
                elif controller == "coordinated-transition-feedback":
                    q_ref = q_post + knee_extension_com_trim(q_post)
                else:
                    q_ref = q_post
            else:
                flag_L, flag_R, ema_coeff = arm_flags_for_global_tick(c)
                if c == 600:
                    # At tick 600 repeat prior requested target without advancing GRU
                    q_ref = prior_requested22.copy()
                else:
                    with torch.inference_mode():
                        out, h = g_net(torch.tensor([[[flag_L, flag_R]]], dtype=torch.float32), h)
                    alpha = np.clip(out[0, 0].numpy().astype(np.float32), 0.0, 1.0)
                    raw_sh = -1.2 * alpha
                    filt = raw_sh.copy() if filt is None else ema_coeff * raw_sh + (1.0 - ema_coeff) * filt
                    arm_L = float(np.clip(-1.0 * np.clip(filt[0], -1.52, 1.52), -1.698, 1.805))
                    arm_R = float(np.clip(-1.0 * np.clip(filt[1], -1.52, 1.52), -1.698, 1.805))
                    q_arm = np.zeros(22, dtype=np.float32)
                    q_arm[0] = arm_R
                    q_arm[4] = arm_L
                    if controller == "coordinated-transition-feedback":
                        q_ref = q_arm + knee_extension_com_trim(q_arm)
                    else:
                        q_ref = q_arm

            dq_ref = (q_ref - prev_qref) / 0.02
            prev_qref = q_ref.copy()
            v = pkt["values"].astype(np.float32) if pkt is not None else np.zeros(50, dtype=np.float32)
            phase_vec = np.array(
                [float(stage_id == 1), float(stage_id == 2), float(stage_id == 3)],
                dtype=np.float32,
            )
            mean_cmd = np.array([float(np.mean(np.abs(q_ref[[0, 4]])))], dtype=np.float32)
            obs120 = np.concatenate(
                [
                    v[:22] - q_ref,
                    0.15 * v[22:44],
                    0.25 * v[44:47],
                    v[47:50],
                    prev_raw22,
                    q_ref,
                    dq_ref,
                    phase_vec,
                    mean_cmd,
                ]
            ).astype(np.float32)

            if controller == "open-loop-handoff" or sd is None:
                raw22 = np.zeros(22, dtype=np.float32)
                delta = np.zeros(22, dtype=np.float32)
            else:
                with torch.inference_mode():
                    mean_raw = actor_forward(sd, torch.tensor(obs120[None], dtype=torch.float32))[0].numpy()
                if explore_sigma > 0.0:
                    raw22 = (mean_raw + exp_rng.normal(0.0, explore_sigma, 22)).astype(np.float32)
                else:
                    raw22 = mean_raw.astype(np.float32)
                delta = (0.03 * np.tanh(raw22)).astype(np.float32)

            if spec.get("collect_training", False):
                u_star = transition_corrective_raw_target(v, q_ref)
                obs_list.append(obs120.copy())
                act_list.append(raw22.copy())
                target_raw_list.append(u_star.copy())

            prev_raw22 = raw22.copy()
            target = (q_ref + delta).astype(float)
            prior_requested22 = target.astype(np.float32).copy()
            rt.command(rt.names, target)

        add(
            controls,
            time_s=float(d.time),
            stage_id=stage_id,
            q_ref22_rad=q_ref,
            raw_residual22=raw22,
            delta22_rad=delta,
            requested22_rad=target,
            accepted22_rad=rt.previous_command,
        )

        sat = np.zeros(22)
        peak = np.zeros(22)
        for _ in range(80):
            rt.update()
            mujoco.mj_step(m, d)
            physics_steps += 1
            sat += rt.saturated
            peak = np.maximum(peak, np.abs(rt.torque))
        mujoco.mj_forward(m, d)
        st = sample_state()
        add(native, **st)
        add(controls, substep_saturated_fraction22=sat / 80.0, substep_peak_abs_torque22_Nm=peak)

        if c >= 100 and spec.get("collect_training", False):
            pose_sq = float(np.mean((d.qpos[rt.qids] - q_ref) ** 2))
            tilt_sq = float(st["torso_tilt_rad"] ** 2)
            act_sq = float(np.mean(delta**2))
            both_feet = float((st["foot_normal_N"] > 1.0).all())
            rew = 1.5 * both_feet - 8.0 * tilt_sq - 10.0 * pose_sq - 0.5 * act_sq
            rew_list.append(rew)

    arrays = {k: np.asarray(v) for k, v in native.items()}
    arrays.update({"control_" + k: np.asarray(v) for k, v in controls.items()})

    trace_ref = None
    if save_trace_path is not None:
        save_trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_trace_path, initial_qpos=initial_qpos, initial_qvel=initial_qvel, **arrays)
        trace_ref = ref(save_trace_path)

    q = arrays["qpos"]
    tilt = arrays["torso_tilt_rad"]
    pose_err = np.abs(q[1:, rt.qids] - arrays["control_q_ref22_rad"])
    whole_pose_rms = float(np.sqrt(np.mean(pose_err**2)))
    tilt_rms = float(np.sqrt(np.mean(tilt**2)))
    com_drift = np.linalg.norm(arrays["whole_body_com_m"][:, :2] - init_com_xy, axis=1)

    stage_ids = arrays["control_stage_id"]
    stage_metrics = {}
    for s_id, s_name in [(0, "stance_0_2s"), (1, "posture_2_12s"), (2, "bilateral_arms_12_22s"), (3, "independent_arms_22_34s")]:
        mask = stage_ids == s_id
        if np.any(mask):
            s_tilt = tilt[1:][mask]
            stage_metrics[s_name] = {
                "controls": int(mask.sum()),
                "tilt_rms_rad": float(np.sqrt(np.mean(s_tilt**2))),
                "max_tilt_rad": float(np.max(s_tilt)),
            }

    ep_report = dict(
        id=spec["id"],
        controller=controller,
        seed=seed,
        label=spec.get("label", ""),
        trace=trace_ref,
        status="completed",
        requested_duration_s=controls_n * 0.02,
        actual_duration_s=float(d.time),
        controls=controls_n,
        native_physics_steps=physics_steps,
        first_evaluator_physical_event_s=first_event,
        completed_upright=first_event is None,
        whole_pose_rms_rad=whole_pose_rms,
        torso_tilt_rms_rad=tilt_rms,
        max_torso_tilt_rad=float(tilt.max()),
        final_torso_tilt_rad=float(tilt[-1]),
        min_root_height_m=float(q[:, 2].min()),
        max_crouch_lowering_mm=float((q[0, 2] - q[100:601, 2].min()) * 1000.0) if q.shape[0] > 600 else 0.0,
        root_height_m=dist_summary(q[:, 2]),
        torso_tilt_rad=dist_summary(tilt),
        max_horizontal_com_drift_m=float(com_drift.max()),
        both_feet_contact_control_fraction=float(np.mean((arrays["foot_normal_N"][1:] > 0).all(1))),
        left_shoulder_pitch_excursion_rad=float(q[:, rt.qids[4]].max() - q[0, rt.qids[4]]),
        right_shoulder_pitch_excursion_rad=float(q[:, rt.qids[0]].max() - q[0, rt.qids[0]]),
        stage_breakdown=stage_metrics,
        tracking_accuracy_curve=[
            dict(abs_error_radius_rad=r, control_joint_fraction=float(np.mean(pose_err <= r)))
            for r in (0.01, 0.025, 0.05, 0.1, 0.2)
        ],
        wall_s=time.monotonic() - t_start,
    )
    if spec.get("collect_training", False):
        ep_report["train_data"] = dict(
            obs=np.asarray(obs_list, dtype=np.float32),
            act=np.asarray(act_list, dtype=np.float32),
            target_raw=np.asarray(target_raw_list, dtype=np.float32),
            rew=np.asarray(rew_list, dtype=np.float32),
        )
    return ep_report


def build_config() -> dict:
    selected = json.loads(POINTER_PATH.read_text())
    return dict(
        schema="cove50.supported-skill-transitions.config.v1",
        task="T-y3me96",
        selected_pointer=ref(POINTER_PATH),
        selected_model=selected,
        materialized_rigid_mjcf=ref(MATERIALIZED_MJCF),
        walker_checkpoint=ref(WALKER_CKPT_PATH),
        posture_checkpoint=ref(POSTURE_CKPT_PATH),
        recurrent_arm_checkpoint=ref(GRU_CKPT_PATH),
        warm_feedback_checkpoint=ref(WARM_FB_CKPT_PATH),
        training=dict(
            ppo_rounds=12,
            episodes_per_round=4,
            controls_per_episode=300,
            adam_updates_per_round=20,
            total_adam_updates=240,
            total_training_controls=14400,
            total_training_physics_substeps=1152000,
            learning_rate=3e-4,
            clip_param=0.2,
            residual_scale_rad=0.03,
            seed=260100,
        ),
        evaluation_families=FAMILIES,
        evaluation_controls=1700,
        evaluation_duration_s=34.0,
    )


def execute(evidence_dir: Path, work_dir: Path) -> None:
    t0 = time.monotonic()
    cfg = build_config()
    DATA.mkdir(parents=True, exist_ok=True)
    cfg_path = DATA / "config.json"
    write_json(cfg_path, cfg)

    warm = torch.load(WARM_FB_CKPT_PATH, map_location="cpu", weights_only=False)
    actor_sd = {k: v.contiguous().clone() for k, v in warm["actor_state_dict"].items()}
    critic_sd = {k: v.contiguous().clone() for k, v in warm["critic_state_dict"].items()}

    trainable_keys = [k for k in actor_sd if k.startswith("mlp.")]
    params = {k: torch.nn.Parameter(actor_sd[k].clone()) for k in trainable_keys}
    actor_sd.update(params)

    critic_keys = [k for k in critic_sd if k.startswith("mlp.")]
    c_params = {k: torch.nn.Parameter(critic_sd[k].clone()) for k in critic_keys}
    critic_sd.update(c_params)

    optimizer = torch.optim.Adam(
        list(params.values()) + list(c_params.values()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8
    )
    torch.manual_seed(cfg["training"]["seed"])

    ppo_rounds = cfg["training"]["ppo_rounds"]
    updates_per_round = cfg["training"]["adam_updates_per_round"]
    round_logs = []
    total_train_controls = 0
    total_train_physics = 0
    total_adam_updates = 0

    with cf.ProcessPoolExecutor(max_workers=4, initializer=_init_worker) as pool:
        for r_idx in range(ppo_rounds):
            serial_sd = {k: v.detach().cpu().numpy().tolist() for k, v in actor_sd.items()}
            specs = []
            for fam in FAMILIES:
                specs.append(
                    dict(
                        id=f"train-r{r_idx:02d}-{fam['seed']}",
                        seed=fam["seed"],
                        label=fam["label"],
                        round_idx=r_idx,
                        controls=300,
                        compressed_train=True,
                        controller="coordinated-transition-feedback",
                        joint_sigma_rad=fam["joint_sigma_rad"],
                        rpy_envelope_rad=fam["rpy_envelope_rad"],
                        joint_velocity_sigma_rad_s=fam["joint_velocity_sigma_rad_s"],
                        actor_sd=serial_sd,
                        explore_sigma=max(0.02, 0.08 * (1.0 - r_idx / ppo_rounds)),
                        collect_training=True,
                    )
                )
            results = list(pool.map(run_transition_episode, specs))
            total_train_controls += sum(r["controls"] for r in results)
            total_train_physics += sum(r["native_physics_steps"] for r in results)

            obs_batch = torch.tensor(
                np.concatenate([r["train_data"]["obs"] for r in results], axis=0), dtype=torch.float32
            )
            act_batch = torch.tensor(
                np.concatenate([r["train_data"]["act"] for r in results], axis=0), dtype=torch.float32
            )
            ustar_batch = torch.tensor(
                np.concatenate([r["train_data"]["target_raw"] for r in results], axis=0), dtype=torch.float32
            )
            rew_batch = torch.tensor(
                np.concatenate([r["train_data"]["rew"] for r in results], axis=0), dtype=torch.float32
            )

            with torch.no_grad():
                n_new = obs_batch.shape[0]
                old_count = int(actor_sd["obs_normalizer.count"].item())
                new_count = old_count + n_new
                rate = n_new / max(1, new_count)
                mean_x = obs_batch.mean(dim=0, keepdim=True)
                var_x = obs_batch.var(dim=0, unbiased=False, keepdim=True)
                delta_m = mean_x - actor_sd["obs_normalizer._mean"]
                actor_sd["obs_normalizer._mean"] = actor_sd["obs_normalizer._mean"] + rate * delta_m
                actor_sd["obs_normalizer._var"] = actor_sd["obs_normalizer._var"] + rate * (
                    var_x - actor_sd["obs_normalizer._var"] + delta_m * (mean_x - actor_sd["obs_normalizer._mean"])
                )
                actor_sd["obs_normalizer._std"] = torch.sqrt(torch.clamp(actor_sd["obs_normalizer._var"], min=1e-6))
                actor_sd["obs_normalizer.count"] = torch.tensor(new_count, dtype=torch.long)

                old_mean = actor_forward(actor_sd, obs_batch).detach()
                returns = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-5)
                adv = torch.clamp(returns, -2.0, 2.0).unsqueeze(1)

            sigma = 0.15
            gen = torch.Generator().manual_seed(cfg["training"]["seed"] + r_idx)
            losses = []
            for _ in range(updates_per_round):
                idx = torch.randint(0, obs_batch.shape[0], (256,), generator=gen)
                b_obs = obs_batch[idx]
                b_act = act_batch[idx]
                b_old = old_mean[idx]
                b_ustar = ustar_batch[idx]
                b_adv = adv[idx]
                b_ret = returns[idx].unsqueeze(1)

                pred_mean = actor_forward(actor_sd, b_obs)
                logp_new = -0.5 * ((b_act - pred_mean) / sigma).square().mean(dim=1, keepdim=True)
                logp_old = -0.5 * ((b_act - b_old) / sigma).square().mean(dim=1, keepdim=True)
                ratio = torch.exp(torch.clamp(logp_new - logp_old, -5.0, 5.0))
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                ppo_loss = -torch.min(surr1, surr2).mean()

                val_pred = critic_forward(critic_sd, b_obs)
                val_loss = F.mse_loss(val_pred, b_ret)
                stabilize_loss = F.mse_loss(pred_mean, b_ustar)

                loss = 0.15 * ppo_loss + 0.15 * val_loss + 1.0 * stabilize_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(params.values()) + list(c_params.values()), 1.0)
                optimizer.step()
                total_adam_updates += 1
                losses.append(float(loss.detach().item()))

            round_summary = dict(
                ppo_round=r_idx + 1,
                cumulative_adam_updates=total_adam_updates,
                mean_tilt_rms_rad=float(np.mean([r["torso_tilt_rms_rad"] for r in results])),
                mean_pose_rms_rad=float(np.mean([r["whole_pose_rms_rad"] for r in results])),
                mean_reward=float(rew_batch.mean().item()),
                final_loss=losses[-1],
            )
            round_logs.append(round_summary)
            print(json.dumps(round_summary), flush=True)

        # Save trained checkpoint
        train_dir = evidence_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = train_dir / "candidate.pt"
        final_actor_sd = {k: v.detach().cpu().clone() for k, v in actor_sd.items()}
        final_critic_sd = {k: v.detach().cpu().clone() for k, v in critic_sd.items()}
        torch.save(
            dict(
                actor_state_dict=final_actor_sd,
                critic_state_dict=final_critic_sd,
                optimizer_state_dict=optimizer.state_dict(),
                config=ref(cfg_path),
                warm_start=ref(WARM_FB_CKPT_PATH),
                walker_donor=ref(WALKER_CKPT_PATH),
                posture_donor=ref(POSTURE_CKPT_PATH),
                recurrent_arm_donor=ref(GRU_CKPT_PATH),
                layout=dict(actor=120, critic=174, raw=22),
                counts=dict(
                    simulator_controls=total_train_controls,
                    physics_substeps=total_train_physics,
                    PPO_rounds=ppo_rounds,
                    optimizer_updates=total_adam_updates,
                    new_weight_sets=1,
                ),
                round_logs=round_logs,
            ),
            ckpt_path,
        )

        # Full 34.0 s (1,700-control) evaluation across all 4 start families (before vs after)
        eval_dir = evidence_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        serial_final_sd = {k: v.numpy().tolist() for k, v in final_actor_sd.items()}
        eval_specs = []
        for fam in FAMILIES:
            for branch, ctrl_name in [
                ("before", "open-loop-handoff"),
                ("after", "coordinated-transition-feedback"),
            ]:
                ep_id = f"eval-{branch}-transition-{fam['seed']}"
                eval_specs.append(
                    dict(
                        id=ep_id,
                        branch=branch,
                        seed=fam["seed"],
                        label=fam["label"],
                        controls=1700,
                        compressed_train=False,
                        controller=ctrl_name,
                        joint_sigma_rad=fam["joint_sigma_rad"],
                        rpy_envelope_rad=fam["rpy_envelope_rad"],
                        joint_velocity_sigma_rad_s=fam["joint_velocity_sigma_rad_s"],
                        actor_sd=serial_final_sd if branch == "after" else None,
                        explore_sigma=0.0,
                        collect_training=False,
                        save_trace_path=str(eval_dir / f"{ep_id}.npz"),
                    )
                )
        eval_results = list(pool.map(run_transition_episode, eval_specs))

    for r in eval_results:
        write_json(eval_dir / f"{r['id']}.json", r)

    paired = []
    for fam in FAMILIES:
        sid = fam["seed"]
        b_ep = next(r for r in eval_results if r["id"] == f"eval-before-transition-{sid}")
        a_ep = next(r for r in eval_results if r["id"] == f"eval-after-transition-{sid}")
        paired.append(
            dict(
                seed=sid,
                label=fam["label"],
                before_completed_upright=b_ep["completed_upright"],
                after_completed_upright=a_ep["completed_upright"],
                before_first_event_s=b_ep["first_evaluator_physical_event_s"],
                after_first_event_s=a_ep["first_evaluator_physical_event_s"],
                before_tilt_rms_rad=b_ep["torso_tilt_rms_rad"],
                after_tilt_rms_rad=a_ep["torso_tilt_rms_rad"],
                tilt_rms_reduction_pct=100.0
                * (b_ep["torso_tilt_rms_rad"] - a_ep["torso_tilt_rms_rad"])
                / max(1e-9, b_ep["torso_tilt_rms_rad"]),
                before_max_tilt_rad=b_ep["max_torso_tilt_rad"],
                after_max_tilt_rad=a_ep["max_torso_tilt_rad"],
                before_pose_rms_rad=b_ep["whole_pose_rms_rad"],
                after_pose_rms_rad=a_ep["whole_pose_rms_rad"],
                after_both_feet_fraction=a_ep["both_feet_contact_control_fraction"],
                after_crouch_lowering_mm=a_ep["max_crouch_lowering_mm"],
                after_left_shoulder_rad=a_ep["left_shoulder_pitch_excursion_rad"],
                after_right_shoulder_rad=a_ep["right_shoulder_pitch_excursion_rad"],
            )
        )

    eval_controls = sum(r["controls"] for r in eval_results)
    eval_physics = sum(r["native_physics_steps"] for r in eval_results)
    report = dict(
        schema="cove50.supported-skill-transitions.report.v1",
        status="completed",
        config=ref(cfg_path),
        candidate=ref(ckpt_path),
        warm_start=ref(WARM_FB_CKPT_PATH),
        walker_donor=ref(WALKER_CKPT_PATH),
        posture_donor=ref(POSTURE_CKPT_PATH),
        recurrent_arm_donor=ref(GRU_CKPT_PATH),
        accounting=dict(
            parallel_world_slots=4,
            training_episodes=ppo_rounds * len(FAMILIES),
            evaluation_episodes=len(eval_results),
            total_episodes=ppo_rounds * len(FAMILIES) + len(eval_results),
            independent_start_families=4,
            training_simulator_controls=total_train_controls,
            evaluation_simulator_controls=eval_controls,
            total_simulator_controls=total_train_controls + eval_controls,
            training_physics_substeps=total_train_physics,
            evaluation_physics_substeps=eval_physics,
            total_physics_substeps=total_train_physics + eval_physics,
            PPO_rounds=ppo_rounds,
            optimizer_updates=total_adam_updates,
            new_weight_sets=1,
        ),
        training_rounds=round_logs,
        paired_comparisons=paired,
        evaluation_episodes=eval_results,
        wall_s=time.monotonic() - t0,
    )
    write_json(evidence_dir / "report.json", report)
    write_json(
        DATA / "summary.json",
        dict(
            report=ref(evidence_dir / "report.json"),
            candidate=ref(ckpt_path),
            accounting=report["accounting"],
            paired_comparisons=paired,
        ),
    )
    print(
        json.dumps(
            dict(
                status="completed",
                candidate=ref(ckpt_path),
                accounting=report["accounting"],
                paired_comparisons=paired,
            )
        ),
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--evidence", type=Path, default=os.environ.get("LAB_EVIDENCE_DIR"))
    p.add_argument("--work", type=Path, default=os.environ.get("LAB_WORK_DIR"))
    args = p.parse_args()
    if not args.execute:
        cfg = build_config()
        DATA.mkdir(parents=True, exist_ok=True)
        write_json(DATA / "config.json", cfg)
        print(json.dumps(dict(config=ref(DATA / "config.json"), activity="config frozen only")))
        return
    if not args.evidence or not args.work:
        p.error("Requires --evidence and --work (or LAB_EVIDENCE_DIR / LAB_WORK_DIR)")
    execute(Path(args.evidence).resolve(), Path(args.work).resolve())


if __name__ == "__main__":
    main()
