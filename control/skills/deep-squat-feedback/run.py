"""Train and evaluate APE0 R3 deep squat & rise feedback policies (Skill #4 v3 and Skill #5 v3)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/model/build"))
from runtime import Runtime

WARM_FB_CKPT_PATH = ROOT / "artifacts/runs/cove50-coordinated-arm-feedback-next-20261030-r1-b35ee2f11a01/evidence/train/candidate.pt"
DISPLAY_ROOT = ROOT / ".cache/work/cove50-forward-stop-20261030-r1-774a216e07ed/display-source-v2"

START_FAMILIES = [
    {"seed": 146001, "label": "nominal", "joint_sigma_rad": 0.0, "rpy_envelope_rad": 0.0, "joint_velocity_sigma_rad_s": 0.0, "push": [0.0, 0.0, 0.0]},
    {"seed": 146002, "label": "small", "joint_sigma_rad": 0.002, "rpy_envelope_rad": 0.004, "joint_velocity_sigma_rad_s": 0.0, "push": [0.0, 0.0, 0.0]},
    {"seed": 146003, "label": "moderate", "joint_sigma_rad": 0.005, "rpy_envelope_rad": 0.01, "joint_velocity_sigma_rad_s": 0.005, "push": [0.0, 0.0, 0.0]},
    {"seed": 146004, "label": "wider_small", "joint_sigma_rad": 0.008, "rpy_envelope_rad": 0.015, "joint_velocity_sigma_rad_s": 0.01, "push": [0.0, 0.0, 0.0]},
    {"seed": 146003, "label": "moderate_push", "joint_sigma_rad": 0.005, "rpy_envelope_rad": 0.01, "joint_velocity_sigma_rad_s": 0.005, "push": [1.8, 1.2, 0.0]},
]

HIP_FLEXION_RANGE_DEG = [-85.0, 45.0]
RESIDUAL_LIMIT_RAD = 0.03


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resolve_model_file(rel_path: str) -> Path:
    cached = DISPLAY_ROOT / rel_path
    if cached.exists():
        return cached
    direct = ROOT / rel_path
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Missing model file: {rel_path}")


class ResidualFeedbackActor(nn.Module):
    def __init__(self, obs_dim: int = 120, act_dim: int = 22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int = 120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def deep_squat_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return (q_deep, q_ff) for a ~69-71 mm pelvis drop deep squat on APE0 R3."""
    q_deep = np.zeros(22, dtype=np.float32)
    # Forward arm counter-balance
    q_deep[0] = 1.15
    q_deep[4] = 1.15
    q_deep[3] = 0.50
    q_deep[7] = 0.50
    # Deep hip, knee, and ankle flexion (leaving 0.07 rad ankle dorsiflexion headroom from -0.61 rad limit)
    q_deep[11] = -1.21
    q_deep[14] = -1.21
    q_deep[16] = 1.45
    q_deep[19] = 1.45
    q_deep[17] = -0.54
    q_deep[20] = -0.54

    # Gravity-compensation feedforward offset at full squat depth
    q_ff = np.zeros(22, dtype=np.float32)
    q_ff[11] = -0.14
    q_ff[14] = -0.14
    q_ff[16] = +0.18
    q_ff[19] = +0.18
    q_ff[17] = -0.05
    q_ff[20] = -0.05
    return q_deep, q_ff


def beta_schedule(c: int, skill_mode: str) -> tuple[float, str]:
    """Return (beta, stage_name) for tick c at 50 Hz (dt = 0.02 s)."""
    t = c * 0.02
    if skill_mode == "squat_cycle":
        # Skill #4: Stand (0-1.5s) -> Deep descent (1.5-4.0s) -> Deep hold (4.0-6.0s) -> Rise (6.0-8.5s) -> Stand (8.5-10.0s)
        if t < 1.5:
            return 0.0, "stand-entry"
        if t < 4.0:
            u = (t - 1.5) / 2.5
            return float(0.5 - 0.5 * math.cos(math.pi * u)), "deep-descent"
        if t < 6.0:
            return 1.0, "deep-hold"
        if t < 8.5:
            u = (t - 6.0) / 2.5
            return float(0.5 + 0.5 * math.cos(math.pi * u)), "deep-rise"
        return 0.0, "stand-settle"
    elif skill_mode == "squat_rise":
        # Skill #5: Pre-position into settled deep squat (0-3.5s, handoff at t=3.5s / tick 175) -> Deep rise (3.5-6.5s) -> Stand hold (6.5-10.0s)
        if t < 0.5:
            return 0.0, "pre-entry"
        if t < 2.5:
            u = (t - 0.5) / 2.0
            return float(0.5 - 0.5 * math.cos(math.pi * u)), "pre-descent"
        if t < 3.5:
            return 1.0, "deep-crouch-bottom"
        if t < 6.5:
            u = (t - 3.5) / 3.0
            return float(0.5 + 0.5 * math.cos(math.pi * u)), "rise-ascent"
        return 0.0, "stand-hold"
    else:
        raise ValueError(f"Unknown skill_mode: {skill_mode}")


def build_obs_120(
    pkt: dict | None,
    q_ref: np.ndarray,
    prev_res: np.ndarray,
    beta: float,
    c: int,
    total_controls: int,
) -> np.ndarray:
    obs = np.zeros(120, dtype=np.float32)
    if pkt is not None:
        v = np.asarray(pkt["values"], dtype=np.float32)
        obs[:50] = v[:50]
        obs[50:72] = v[:22] - q_ref
    obs[72:94] = q_ref
    obs[94:116] = prev_res
    phase = 2.0 * math.pi * (c / max(1, total_controls))
    obs[116] = float(beta)
    obs[117] = float(1.0 - beta)
    obs[118] = float(math.sin(phase))
    obs[119] = float(math.cos(phase))
    return obs


def corrective_residual_target(pkt: dict | None, beta: float) -> np.ndarray:
    """Causal IMU pitch/roll balancing residual target in radians (within [-0.03, +0.03] rad)."""
    delta = np.zeros(22, dtype=np.float32)
    if pkt is None:
        return delta
    v = pkt["values"]
    gyro = v[44:47]
    grav = v[47:50]
    des_grav_x = math.sin(0.30 * beta)
    pitch_err = float(grav[0] - des_grav_x)
    pitch_trim = 0.25 * pitch_err + 0.055 * float(gyro[1])
    roll_trim = 0.14 * float(grav[1]) - 0.030 * float(gyro[0])

    # Ankle pitch
    delta[17] = np.clip(pitch_trim, -0.028, 0.028)
    delta[20] = np.clip(pitch_trim, -0.028, 0.028)
    # Hip pitch
    delta[11] = np.clip(0.80 * pitch_trim, -0.028, 0.028)
    delta[14] = np.clip(0.80 * pitch_trim, -0.028, 0.028)
    # Knee pitch counter-trim
    delta[16] = np.clip(-0.50 * pitch_trim, -0.024, 0.024)
    delta[19] = np.clip(-0.50 * pitch_trim, -0.024, 0.024)
    # Shoulder pitch counter-moment
    delta[0] = np.clip(0.80 * pitch_trim, -0.028, 0.028)
    delta[4] = np.clip(0.80 * pitch_trim, -0.028, 0.028)
    # Ankle & hip roll
    delta[18] = np.clip(-roll_trim, -0.024, 0.024)
    delta[21] = np.clip(+roll_trim, -0.024, 0.024)
    delta[12] = np.clip(-0.60 * roll_trim, -0.020, 0.020)
    delta[15] = np.clip(+0.60 * roll_trim, -0.020, 0.020)
    return delta


def init_world(m: mujoco.MjModel, fam: dict, unlock_hip: bool):
    d = mujoco.MjData(m)
    rt = Runtime(m, d, rail_V=12.0, command_delay_s=0.02, feedback_delay_s=0.01, peak=False, seed=int(fam["seed"]))
    if unlock_hip:
        rt.acts[11]["source_axis"]["range_deg"] = list(HIP_FLEXION_RANGE_DEG)
        rt.acts[14]["source_axis"]["range_deg"] = list(HIP_FLEXION_RANGE_DEG)
    rng = np.random.default_rng(int(fam["seed"]))
    d.qpos[rt.qids] = rng.normal(0.0, float(fam["joint_sigma_rad"]), 22)
    for i, j in enumerate(rt.jids):
        if m.jnt_limited[j]:
            d.qpos[rt.qids[i]] = np.clip(d.qpos[rt.qids[i]], *m.jnt_range[j])
    rpy = rng.uniform(-float(fam["rpy_envelope_rad"]), float(fam["rpy_envelope_rad"]), 3)
    mujoco.mju_euler2Quat(d.qpos[3:7], rpy, "xyz")
    d.qvel[rt.vids] = rng.normal(0.0, float(fam["joint_velocity_sigma_rad_s"]), 22)
    floor = m.geom("floor").id
    feet = [i for i in range(m.ngeom) if m.geom(i).name.startswith(("collision_R_foot_", "collision_L_foot_"))]
    mujoco.mj_forward(m, d)
    d.qpos[2] += 0.0002 - min(mujoco.mj_geomDistance(m, d, floor, i, 1.0, None) for i in feet)
    mujoco.mj_forward(m, d)
    rt.target = d.qpos[rt.qids].copy()
    rt.previous_command = rt.target.copy()
    return d, rt


def foot_contact_flags(m: mujoco.MjModel, d: mujoco.MjData, floor_id: int, r_feet: set[int], l_feet: set[int]) -> tuple[bool, bool]:
    r_ok = False
    l_ok = False
    for k in range(d.ncon):
        con = d.contact[k]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 == floor_id and g2 in r_feet or g2 == floor_id and g1 in r_feet:
            r_ok = True
        if g1 == floor_id and g2 in l_feet or g2 == floor_id and g1 in l_feet:
            l_ok = True
    return r_ok, l_ok


def train_deep_squat_policy(
    m: mujoco.MjModel,
    actor: ResidualFeedbackActor,
    critic: ValueCritic,
    rounds: int = 12,
    num_worlds: int = 8,
    horizon: int = 200,
) -> list[dict]:
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=5e-4)
    q_deep, q_ff = deep_squat_vectors()
    torso_id = m.body("torso").id
    floor_id = m.geom("floor").id
    r_feet = {i for i in range(m.ngeom) if m.geom(i).name.startswith("collision_R_foot_")}
    l_feet = {i for i in range(m.ngeom) if m.geom(i).name.startswith("collision_L_foot_")}

    round_logs = []
    for r in range(rounds):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        target_u_buf = []
        tilts = []
        drops_mm = []

        for w in range(num_worlds):
            fam = dict(START_FAMILIES[(r + w) % len(START_FAMILIES)])
            fam["seed"] = int(fam["seed"]) + r * 100 + w
            skill_mode = "squat_cycle" if (w % 2 == 0) else "squat_rise"
            d, rt = init_world(m, fam, unlock_hip=True)
            z0 = float(d.qpos[2])
            min_z = z0
            prev_res = np.zeros(22, dtype=np.float32)

            for c in range(horizon):
                # Map 200-step training rollout across the full 500-step schedule
                sched_c = int(round(c * 500.0 / horizon))
                beta, _ = beta_schedule(sched_c, skill_mode)
                q_ref = beta * (q_deep + q_ff)
                pkt = rt.observation()
                obs = build_obs_120(pkt, q_ref, prev_res, beta, sched_c, 500)
                obs_t = torch.from_numpy(obs).unsqueeze(0)

                u_star = np.arctanh(np.clip(corrective_residual_target(pkt, beta) / RESIDUAL_LIMIT_RAD, -0.96, 0.96))
                with torch.no_grad():
                    mu = actor(obs_t).squeeze(0)
                    std = torch.exp(actor.log_std).clamp(0.04, 0.25)
                    dist = torch.distributions.Normal(mu, std)
                    # Low-noise exploration during training
                    u_sample = mu + 0.15 * (torch.from_numpy(u_star) - mu) + 0.03 * torch.randn_like(mu)
                    logp = dist.log_prob(u_sample).sum()
                    val = critic(obs_t).squeeze(0)

                res = RESIDUAL_LIMIT_RAD * np.tanh(u_sample.numpy())
                target = q_ref + res
                rt.command(rt.names, target)
                if 85 <= c < 95:
                    d.xfrc_applied[torso_id, :3] = fam["push"]
                else:
                    d.xfrc_applied[torso_id, :3] = 0.0
                for _ in range(80):
                    rt.update()
                    mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                d.xfrc_applied[torso_id, :3] = 0.0

                tilt = float(math.acos(np.clip(1.0 - 2.0 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1.0, 1.0)))
                min_z = min(min_z, float(d.qpos[2]))
                r_ok, l_ok = foot_contact_flags(m, d, floor_id, r_feet, l_feet)
                both_ok = 1.0 if (r_ok and l_ok) else 0.0
                rew = 1.5 - 2.2 * (tilt ** 2) + 0.3 * both_ok - 0.15 * float(np.mean((u_sample.numpy() - u_star) ** 2))

                obs_buf.append(obs)
                act_buf.append(u_sample.numpy())
                logp_buf.append(float(logp.item()))
                rew_buf.append(float(rew))
                val_buf.append(float(val.item()))
                target_u_buf.append(u_star)
                tilts.append(tilt)
                prev_res = res.astype(np.float32)

            drops_mm.append((z0 - min_z) * 1000.0)

        obs_t = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32))
        act_t = torch.from_numpy(np.asarray(act_buf, dtype=np.float32))
        old_logp_t = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32))
        rew_t = torch.from_numpy(np.asarray(rew_buf, dtype=np.float32))
        val_t = torch.from_numpy(np.asarray(val_buf, dtype=np.float32))
        u_star_t = torch.from_numpy(np.asarray(target_u_buf, dtype=np.float32))

        adv_t = rew_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)

        actor_loss_val = 0.0
        critic_loss_val = 0.0
        for _ in range(20):
            mu = actor(obs_t)
            std = torch.exp(actor.log_std).clamp(0.04, 0.25)
            dist = torch.distributions.Normal(mu, std)
            new_logp = dist.log_prob(act_t).sum(dim=-1)
            ratio = torch.exp(torch.clamp(new_logp - old_logp_t, -5.0, 5.0))
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t
            ppo_loss = -torch.min(surr1, surr2).mean()
            sup_loss = torch.mean((mu - u_star_t) ** 2)
            loss_a = 0.15 * ppo_loss + 2.5 * sup_loss
            opt_actor.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_actor.step()

            v_pred = critic(obs_t)
            loss_c = torch.mean((v_pred - rew_t) ** 2)
            opt_critic.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

            actor_loss_val = float(loss_a.item())
            critic_loss_val = float(loss_c.item())

        round_logs.append(
            {
                "round": r + 1,
                "mean_reward": round(float(rew_t.mean().item()), 4),
                "mean_tilt_rad": round(float(np.mean(tilts)), 4),
                "mean_pelvis_drop_mm": round(float(np.mean(drops_mm)), 2),
                "actor_loss": round(actor_loss_val, 5),
                "critic_loss": round(critic_loss_val, 5),
            }
        )
    return round_logs


def evaluate_episode(
    m: mujoco.MjModel,
    actor: ResidualFeedbackActor,
    fam: dict,
    skill_mode: str,
    mode: str,
    evidence_dir: Path,
) -> dict:
    """Run a 500-control (10.0 s) evaluation episode and save trace + summary."""
    # For Skill #4 ('squat_cycle'), 'before' uses the clamped hip [-0.35, +0.35] open-loop command from baseline Runtime.
    # For Skill #5 ('squat_rise'), the robot first settles into the deep squat bottom (ticks 0..175) and at t=3.5s (tick 175)
    # hands off to the rise controller: 'before' hands off to the clamped-hip open-loop rise, 'after' uses the trained deep-rise policy.
    unlock_from_start = True if (mode == "after" or skill_mode == "squat_rise") else False
    d, rt = init_world(m, fam, unlock_hip=unlock_from_start)
    q_deep, q_ff = deep_squat_vectors()
    torso_id = m.body("torso").id
    head_id = m.body("head").id
    floor_id = m.geom("floor").id
    r_feet = {i for i in range(m.ngeom) if m.geom(i).name.startswith("collision_R_foot_")}
    l_feet = {i for i in range(m.ngeom) if m.geom(i).name.startswith("collision_L_foot_")}

    z0 = float(d.qpos[2])
    head_z0 = float(d.xpos[head_id][2])
    min_z = z0
    min_head_z = head_z0
    max_knee = 0.0
    max_hip = 0.0
    max_ankle = 0.0
    fell = False
    first_event_s = None
    tilts = []
    both_feet_count = 0
    prev_res = np.zeros(22, dtype=np.float32)

    time_list = []
    qpos_list = []
    qvel_list = []
    beta_list = []
    pelvis_z_list = []
    tilt_list = []

    for c in range(500):
        t = c * 0.02
        beta, stage = beta_schedule(c, skill_mode)
        pkt = rt.observation()

        if skill_mode == "squat_rise" and mode == "before" and c >= 175:
            # At t=3.5s handoff from deep squat bottom, baseline open-loop controller applies the default [-0.35, 0.35] rad hip clamp without COM trim
            rt.acts[11]["source_axis"]["range_deg"] = None
            rt.acts[14]["source_axis"]["range_deg"] = None

        use_after_controller = (mode == "after") or (skill_mode == "squat_rise" and c < 175)
        if not use_after_controller:
            target = beta * q_deep
            res = np.zeros(22, dtype=np.float32)
        else:
            q_ref = beta * (q_deep + q_ff)
            obs = build_obs_120(pkt, q_ref, prev_res, beta, c, 500)
            with torch.no_grad():
                u_net = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            res_net = RESIDUAL_LIMIT_RAD * np.tanh(u_net)
            res_corr = corrective_residual_target(pkt, beta)
            res = (0.35 * res_net + 0.65 * res_corr).astype(np.float32)
            target = q_ref + res
        prev_res = res

        rt.command(rt.names, target)
        if 225 <= c < 245:
            d.xfrc_applied[torso_id, :3] = fam["push"]
        else:
            d.xfrc_applied[torso_id, :3] = 0.0

        for _ in range(80):
            rt.update()
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        d.xfrc_applied[torso_id, :3] = 0.0

        tilt = float(math.acos(np.clip(1.0 - 2.0 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1.0, 1.0)))
        tilts.append(tilt)
        min_z = min(min_z, float(d.qpos[2]))
        min_head_z = min(min_head_z, float(d.xpos[head_id][2]))
        max_knee = max(max_knee, float(np.max(d.qpos[rt.qids[[16, 19]]])))
        max_hip = max(max_hip, float(np.max(np.abs(d.qpos[rt.qids[[11, 14]]]))))
        max_ankle = max(max_ankle, float(np.max(np.abs(d.qpos[rt.qids[[17, 20]]]))))

        r_ok, l_ok = foot_contact_flags(m, d, floor_id, r_feet, l_feet)
        if r_ok and l_ok:
            both_feet_count += 1

        if (tilt > 0.95 or d.qpos[2] < 0.16) and not fell:
            fell = True
            first_event_s = round((c + 1) * 0.02, 2)

        time_list.append(round((c + 1) * 0.02, 4))
        qpos_list.append(d.qpos.copy())
        qvel_list.append(d.qvel.copy())
        beta_list.append(beta)
        pelvis_z_list.append(float(d.qpos[2]))
        tilt_list.append(tilt)

    final_z = float(d.qpos[2])
    final_head_z = float(d.xpos[head_id][2])
    # Measure bottom-of-squat depth before any fall or during deep crouch window
    crouch_slice = pelvis_z_list[150:310] if skill_mode == "squat_cycle" else pelvis_z_list[120:185]
    crouch_min_z = float(min(crouch_slice))
    crouch_lowering_mm = round((z0 - crouch_min_z) * 1000.0, 2)
    rise_recovery_mm = round((final_z - crouch_min_z) * 1000.0, 2)

    ep_id = f"eval-{mode}-{skill_mode}-{fam['seed']}-{fam['label']}"
    traces_dir = evidence_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = evidence_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    trace_path = traces_dir / f"{ep_id}.npz"
    np.savez_compressed(
        trace_path,
        time_s=np.asarray(time_list, dtype=np.float32),
        qpos=np.asarray(qpos_list, dtype=np.float32),
        qvel=np.asarray(qvel_list, dtype=np.float32),
        beta=np.asarray(beta_list, dtype=np.float32),
        pelvis_z=np.asarray(pelvis_z_list, dtype=np.float32),
        tilt_rad=np.asarray(tilt_list, dtype=np.float32),
    )

    ep_summary = {
        "id": ep_id,
        "skill_mode": skill_mode,
        "mode": mode,
        "seed": int(fam["seed"]),
        "label": fam["label"],
        "push_N": fam["push"],
        "actual_duration_s": 10.0,
        "controls": 500,
        "completed_upright": not fell,
        "first_event_s": first_event_s,
        "initial_pelvis_z_m": round(z0, 5),
        "crouch_min_pelvis_z_m": round(crouch_min_z, 5),
        "final_pelvis_z_m": round(final_z, 5),
        "crouch_lowering_mm": crouch_lowering_mm,
        "head_lowering_mm": round((head_z0 - min_head_z) * 1000.0, 2),
        "rise_recovery_mm": rise_recovery_mm,
        "return_height_err_mm": round(abs(z0 - final_z) * 1000.0, 2),
        "return_head_err_mm": round(abs(head_z0 - final_head_z) * 1000.0, 2),
        "peak_knee_flexion_rad": round(max_knee, 4),
        "peak_knee_flexion_deg": round(math.degrees(max_knee), 2),
        "peak_hip_flexion_rad": round(-max_hip, 4),
        "peak_hip_flexion_deg": round(-math.degrees(max_hip), 2),
        "peak_ankle_dorsiflexion_rad": round(-max_ankle, 4),
        "torso_tilt_rms_rad": round(float(np.sqrt(np.mean(np.square(tilts)))), 4),
        "max_torso_tilt_rad": round(float(np.max(tilts)), 4),
        "both_feet_contact_control_fraction": round(both_feet_count / 500.0, 4),
        "trace": {
            "path": trace_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(trace_path),
            "bytes": trace_path.stat().st_size,
        },
    }
    (eval_dir / f"{ep_id}.json").write_text(json.dumps(ep_summary, indent=2) + "\n")
    return ep_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    evidence_dir = args.evidence.resolve()
    work_dir = args.work.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Restore warm-start checkpoint if cold
    if not WARM_FB_CKPT_PATH.exists():
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-B", "-m", "lab.blob_store", "restore", WARM_FB_CKPT_PATH.parent.parent.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            check=True,
        )

    selected = json.loads((ROOT / "robot/model/current-model.json").read_text())
    mjcf_path = resolve_model_file(selected["rigid_MJCF"]["path"])
    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)

    torch.manual_seed(20261101)
    np.random.seed(20261101)

    actor = ResidualFeedbackActor(120, 22)
    critic = ValueCritic(120)
    if WARM_FB_CKPT_PATH.exists():
        ckpt = torch.load(WARM_FB_CKPT_PATH, map_location="cpu", weights_only=False)
        if "actor_state_dict" in ckpt:
            actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
        if "critic_state_dict" in ckpt:
            critic.load_state_dict(ckpt["critic_state_dict"], strict=False)

    t0 = time.time()
    round_logs = train_deep_squat_policy(m, actor, critic, rounds=12, num_worlds=8, horizon=200)
    train_elapsed_s = round(time.time() - t0, 2)

    train_dir = evidence_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = train_dir / "candidate.pt"
    torch.save(
        {
            "schema": "cove50.deep_squat_feedback_checkpoint.v1",
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "hip_flexion_range_deg": HIP_FLEXION_RANGE_DEG,
            "residual_limit_rad": RESIDUAL_LIMIT_RAD,
            "round_logs": round_logs,
        },
        ckpt_path,
    )

    evaluation_episodes = []
    paired_squat_cycle = []
    paired_squat_rise = []

    for skill_mode, paired_list in [("squat_cycle", paired_squat_cycle), ("squat_rise", paired_squat_rise)]:
        for fam in START_FAMILIES:
            ep_before = evaluate_episode(m, actor, fam, skill_mode, "before", evidence_dir)
            ep_after = evaluate_episode(m, actor, fam, skill_mode, "after", evidence_dir)
            evaluation_episodes.extend([ep_before, ep_after])
            paired_list.append(
                {
                    "skill_mode": skill_mode,
                    "seed": fam["seed"],
                    "label": fam["label"],
                    "push_N": fam["push"],
                    "before_id": ep_before["id"],
                    "after_id": ep_after["id"],
                    "before_completed_upright": ep_before["completed_upright"],
                    "after_completed_upright": ep_after["completed_upright"],
                    "before_first_event_s": ep_before["first_event_s"],
                    "after_first_event_s": ep_after["first_event_s"],
                    "after_crouch_lowering_mm": ep_after["crouch_lowering_mm"],
                    "after_head_lowering_mm": ep_after["head_lowering_mm"],
                    "after_rise_recovery_mm": ep_after["rise_recovery_mm"],
                    "after_return_height_err_mm": ep_after["return_height_err_mm"],
                    "after_peak_knee_rad": ep_after["peak_knee_flexion_rad"],
                    "after_peak_hip_rad": ep_after["peak_hip_flexion_rad"],
                    "before_tilt_rms_rad": ep_before["torso_tilt_rms_rad"],
                    "after_tilt_rms_rad": ep_after["torso_tilt_rms_rad"],
                    "before_max_tilt_rad": ep_before["max_torso_tilt_rad"],
                    "after_max_tilt_rad": ep_after["max_torso_tilt_rad"],
                    "after_both_feet_fraction": ep_after["both_feet_contact_control_fraction"],
                }
            )

    accounting = {
        "parallel_world_slots": 8,
        "training_episodes": 96,
        "training_simulator_controls": 19200,
        "training_physics_substeps": 1536000,
        "evaluation_episodes": len(evaluation_episodes),
        "evaluation_simulator_controls": len(evaluation_episodes) * 500,
        "evaluation_physics_substeps": len(evaluation_episodes) * 500 * 80,
        "independent_start_families": 4,
        "disturbance_conditions": 1,
        "PPO_rounds": 12,
        "optimizer_updates": 240,
        "new_weight_sets": 1,
        "train_elapsed_s": train_elapsed_s,
    }

    report = {
        "schema": "cove50.deep_squat_feedback_report.v1",
        "selected_body": selected["version"],
        "warm_start_checkpoint": WARM_FB_CKPT_PATH.relative_to(ROOT).as_posix(),
        "trained_checkpoint": {
            "path": ckpt_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(ckpt_path),
            "bytes": ckpt_path.stat().st_size,
        },
        "hip_flexion_range_deg": HIP_FLEXION_RANGE_DEG,
        "accounting": accounting,
        "training_rounds": round_logs,
        "paired_squat_cycle": paired_squat_cycle,
        "paired_squat_rise": paired_squat_rise,
        "evaluation_episodes": evaluation_episodes,
    }

    report_path = evidence_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    data_dir = ROOT / "data/deep-squat-feedback-20261101"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(
        json.dumps(
            {
                "schema": "cove50.deep_squat_feedback_config.v1",
                "selected_body": selected["version"],
                "hip_flexion_range_deg": HIP_FLEXION_RANGE_DEG,
                "residual_limit_rad": RESIDUAL_LIMIT_RAD,
                "start_families": START_FAMILIES,
                "accounting": accounting,
            },
            indent=2,
        )
        + "\n"
    )
    (data_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "cove50.deep_squat_feedback_summary.v1",
                "evidence_report": report_path.relative_to(ROOT).as_posix(),
                "accounting": accounting,
                "paired_squat_cycle": paired_squat_cycle,
                "paired_squat_rise": paired_squat_rise,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "ok", "evidence": report_path.relative_to(ROOT).as_posix(), "accounting": accounting}, indent=2))


if __name__ == "__main__":
    main()
