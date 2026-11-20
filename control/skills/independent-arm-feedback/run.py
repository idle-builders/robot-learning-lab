"""Coordinated 22-DOF sensor-feedback independent arm motion on selected R3.

Warm-starts the 120-input 22-DOF causal sensor-feedback actor from the retained
r3-coordinated-hold-settle-trim checkpoint (5633e975), couples the recurrent
independent-arm command generator (CommandNet 23760523) with whole-body COM
counterweight targets, runs 12 on-policy PPO + corrective feedback rounds
(240 Adam updates) across unilateral left, unilateral right, and alternating
independent arm schedules under torso push impulses, and evaluates paired
10 s episodes across all four upright start families (140701..140704).
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/independent-arm-feedback-20261030'

sys.path.insert(0, str(ROOT / 'robot/model/build'))
sys.path.insert(0, str(ROOT / 'control/skills/basic-skills'))
from runtime import Runtime
from arm_command import CommandNet

POINTER_PATH = ROOT / 'robot/model/current-model.json'
MATERIALIZED_MJCF = ROOT / '.cache/work/cove50-forward-stop-20261030-r1-774a216e07ed/display-source-v2/artifacts/cove50-model-transfer/r3-20261026/cove50-rigid.xml'
GRU_CKPT_PATH = ROOT / 'data/basic-skills-retry-20261027/arm-body-learning-harness/input-snapshot/objects/23760523ead3e54bc0a804d7b2f357902eb61c8526fd54fa2c529cdeb81a350c'
POSTURE_CKPT_PATH = ROOT / '.cache/work/cove50-coordinated-arm-feedback-next-20261030-r1-b35ee2f11a01/preparation/input-objects/d2f5f40efd5e91a5ca2fbd837ac5286e7b515bacf66492509757e317c86a7e16'
WARM_FB_CKPT_PATH = ROOT / 'artifacts/runs/cove50-coordinated-arm-feedback-next-20261030-r1-b35ee2f11a01/evidence/train/candidate.pt'
PAWS = ('ape_body_shell__left_smooth_paw', 'ape_body_shell__right_smooth_paw')


def ref(path: Path) -> dict:
    p = path.resolve()
    with p.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    try:
        rel = p.relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(p)
    return dict(path=rel, sha256=sha, bytes=p.stat().st_size)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def dist_summary(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=float)
    q = np.quantile(arr, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]).tolist()
    return dict(zip(('min', 'p10', 'p50', 'p90', 'p99', 'max'), q))


def actor_forward(sd: dict[str, torch.Tensor], obs120: torch.Tensor) -> torch.Tensor:
    x = (obs120 - sd['obs_normalizer._mean']) / (sd['obs_normalizer._std'] + 0.01)
    for layer in (0, 2, 4):
        x = F.linear(x, sd[f'mlp.{layer}.weight'], sd[f'mlp.{layer}.bias'])
        if layer != 4:
            x = F.elu(x)
    return x


def critic_forward(csd: dict[str, torch.Tensor], obs120: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros((obs120.shape[0], 174 - 120), dtype=obs120.dtype, device=obs120.device)
    x174 = torch.cat([obs120, pad], dim=1)
    x = (x174 - csd['obs_normalizer._mean']) / (csd['obs_normalizer._std'] + 0.01)
    for layer in (0, 2, 4):
        x = F.linear(x, csd[f'mlp.{layer}.weight'], csd[f'mlp.{layer}.bias'])
        if layer != 4:
            x = F.elu(x)
    return x


def flags_for_schedule(schedule_name: str, control: int, total_controls: int) -> tuple[float, float]:
    if total_controls == 200:
        if schedule_name == 'independent-left':
            return (1.0 if 30 <= control < 140 else 0.0, 0.0)
        if schedule_name == 'independent-right':
            return (0.0, 1.0 if 30 <= control < 140 else 0.0)
        if schedule_name == 'independent-alternating':
            return (1.0 if 25 <= control < 95 else 0.0, 1.0 if 105 <= control < 170 else 0.0)
        return (1.0 if 30 <= control < 140 else 0.0, 1.0 if 30 <= control < 140 else 0.0)
    else:
        if schedule_name == 'independent-left':
            return (1.0 if 100 <= control < 300 else 0.0, 0.0)
        if schedule_name == 'independent-right':
            return (0.0, 1.0 if 100 <= control < 300 else 0.0)
        if schedule_name == 'independent-alternating':
            return (1.0 if 75 <= control < 225 else 0.0, 1.0 if 250 <= control < 400 else 0.0)
        return (1.0 if 100 <= control < 300 else 0.0, 1.0 if 100 <= control < 300 else 0.0)


def corrective_raw_target(v: np.ndarray, q_ref: np.ndarray, arm_L: float, arm_R: float) -> np.ndarray:
    """Bounded causal sensor-feedback target in raw space (tanh(raw)*0.03 = delta)."""
    q_del = v[:22]
    qd_del = v[22:44]
    gyro = v[44:47]
    grav = v[47:50]
    mean_load = 0.5 * (arm_L + arm_R) / 1.176
    asym_load = 0.5 * (arm_L - arm_R) / 1.176
    pitch_trim = 0.012 * mean_load + 0.14 * grav[0] + 0.035 * gyro[1]
    roll_trim = 0.004 * asym_load + 0.10 * grav[1] - 0.025 * gyro[0]
    delta = np.zeros(22, dtype=np.float32)
    delta[17] = np.clip(pitch_trim, -0.024, 0.024)
    delta[20] = np.clip(pitch_trim, -0.024, 0.024)
    delta[11] = np.clip(0.6 * pitch_trim, -0.018, 0.018)
    delta[14] = np.clip(0.6 * pitch_trim, -0.018, 0.018)
    delta[18] = np.clip(-roll_trim, -0.018, 0.018)
    delta[21] = np.clip(+roll_trim, -0.018, 0.018)
    delta[[0, 4]] = np.clip(0.18 * (q_ref[[0, 4]] - q_del[[0, 4]]) - 0.015 * qd_del[[0, 4]], -0.022, 0.022)
    ratio = np.clip(delta / 0.03, -0.92, 0.92)
    return np.arctanh(ratio).astype(np.float32)


_WORKER_CACHE = {}


def _init_worker():
    torch.set_num_threads(1)
    m = mujoco.MjModel.from_xml_path(str(MATERIALIZED_MJCF))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    gru_ckpt = torch.load(GRU_CKPT_PATH, map_location='cpu', weights_only=False)
    g_net = CommandNet()
    g_net.load_state_dict({k: v.contiguous().clone() for k, v in gru_ckpt['model_state_dict'].items()})
    g_net.eval()
    paw_geoms = [m.geom(n).id for n in PAWS]
    paw_centers = []
    for geom in paw_geoms:
        mesh = int(m.geom_dataid[geom])
        start = int(m.mesh_vertadr[mesh])
        num = int(m.mesh_vertnum[mesh])
        verts = m.mesh_vert[start:start + num]
        paw_centers.append((verts.min(0) + verts.max(0)) / 2.0)
    _WORKER_CACHE['model'] = m
    _WORKER_CACHE['g_net'] = g_net
    _WORKER_CACHE['paw_geoms'] = paw_geoms
    _WORKER_CACHE['paw_centers'] = paw_centers


def run_single_episode(spec: dict) -> dict:
    if 'model' not in _WORKER_CACHE:
        _init_worker()
    m = _WORKER_CACHE['model']
    g_net = _WORKER_CACHE['g_net']
    paw_geoms = _WORKER_CACHE['paw_geoms']
    paw_centers = _WORKER_CACHE['paw_centers']

    seed = int(spec['seed'])
    schedule_name = spec['schedule']
    controls_n = int(spec['controls'])
    controller = spec['controller']
    push_force = np.asarray(spec.get('push_force', [0.0, 0.0, 0.0]), dtype=float)
    push_window = spec.get('push_window', [0, 0])
    sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in spec['actor_sd'].items()} if spec.get('actor_sd') else None
    explore_sigma = float(spec.get('explore_sigma', 0.0))
    save_trace_path = Path(spec['save_trace_path']) if spec.get('save_trace_path') else None

    d = mujoco.MjData(m)
    rt = Runtime(m, d, rail_V=12.0, command_delay_s=0.02, feedback_delay_s=0.01, peak=False, seed=seed)
    rng = np.random.default_rng(seed)
    d.qpos[rt.qids] = rng.normal(0.0, float(spec['joint_sigma_rad']), 22)
    for i, j in enumerate(rt.jids):
        if m.jnt_limited[j]:
            d.qpos[rt.qids[i]] = np.clip(d.qpos[rt.qids[i]], *m.jnt_range[j])
    rpy = rng.uniform(-float(spec['rpy_envelope_rad']), float(spec['rpy_envelope_rad']), 3)
    mujoco.mju_euler2Quat(d.qpos[3:7], rpy, 'xyz')
    d.qvel[rt.vids] = rng.normal(0.0, float(spec['joint_velocity_sigma_rad_s']), 22)
    mujoco.mj_forward(m, d)
    floor = m.geom('floor').id
    feet = [i for i in range(m.ngeom) if m.geom(i).name.startswith(('collision_R_foot_', 'collision_L_foot_'))]
    d.qpos[2] += 0.0002 - min(mujoco.mj_geomDistance(m, d, floor, i, 1.0, None) for i in feet)
    mujoco.mj_forward(m, d)
    rt.target = d.qpos[rt.qids].copy()
    rt.previous_command = rt.target.copy()

    initial_qpos = d.qpos.copy()
    initial_qvel = d.qvel.copy()
    free_joints = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    root_body = int(m.jnt_bodyid[free_joints[0]])
    torso_id = m.body('torso').id
    init_com_xy = d.subtree_com[root_body][:2].copy()
    initial_yaw = float(np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2)))

    h = None
    filt = None
    prev_raw = np.zeros(22, dtype=np.float32)
    prev_qref = np.zeros(22, dtype=np.float32)
    exp_rng = np.random.default_rng(seed + int(spec.get('round_idx', 0)) * 1009)

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
        for i, c in enumerate(d.contact):
            if c.geom1 == floor or c.geom2 == floor:
                g = c.geom2 if c.geom1 == floor else c.geom1
                name = m.geom(g).name
                mujoco.mj_contactForce(m, d, i, force_buf)
                normal = max(0.0, float(force_buf[0]))
                if name.startswith('collision_R_foot_'):
                    load[0] += normal
                elif name.startswith('collision_L_foot_'):
                    load[1] += normal
                else:
                    other += normal
            else:
                selfdepth = max(selfdepth, -float(c.dist))
        tilt = float(np.arccos(np.clip(1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2), -1.0, 1.0)))
        yaw = float(np.arctan2(2 * (d.qpos[3] * d.qpos[6] + d.qpos[4] * d.qpos[5]), 1 - 2 * (d.qpos[5] ** 2 + d.qpos[6] ** 2)))
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
            sole_positions_m=np.array([d.site(s + '_sole_contact_frame').xpos for s in ('R', 'L')]),
            paw_positions_m_L_R=paws,
            target22_rad=rt.target,
            torque22_Nm=rt.torque,
            current22_A=rt.current,
            saturated22=rt.saturated,
        )

    add(native, **sample_state())

    for c in range(controls_n):
        pkt = rt.observation()
        flag_L, flag_R = flags_for_schedule(schedule_name, c, controls_n)
        if c == 0:
            target = rt.target.copy()
            q_ref = np.zeros(22, dtype=np.float32)
            raw22 = np.zeros(22, dtype=np.float32)
            delta = np.zeros(22, dtype=np.float32)
            obs120 = np.zeros(120, dtype=np.float32)
            alpha = np.zeros(2, dtype=np.float32)
        else:
            with torch.inference_mode():
                out, h = g_net(torch.tensor([[[flag_L, flag_R]]], dtype=torch.float32), h)
            alpha = out[0, 0].numpy().astype(np.float32)
            raw_sh = -1.2 * alpha
            filt = raw_sh.copy() if filt is None else 0.05 * raw_sh + 0.95 * filt
            arm_L = float(np.clip(-0.98 * filt[0], -1.698, 1.805))
            arm_R = float(np.clip(-0.98 * filt[1], -1.698, 1.805))
            q_ref = np.zeros(22, dtype=np.float32)
            q_ref[0] = arm_R
            q_ref[4] = arm_L
            dq_ref = (q_ref - prev_qref) / 0.02
            prev_qref = q_ref.copy()

            v = pkt['values'].astype(np.float32)
            active_flag = float(flag_L > 0.0 or flag_R > 0.0)
            phase = np.array([float(c < (30 if controls_n == 200 else 75)), active_flag, float(not active_flag and c >= (30 if controls_n == 200 else 75))], dtype=np.float32)
            mean_alpha = np.array([float(0.5 * (alpha[0] + alpha[1]))], dtype=np.float32)
            obs120 = np.concatenate([
                v[:22] - q_ref,
                0.15 * v[22:44],
                0.25 * v[44:47],
                v[47:50],
                prev_raw,
                q_ref,
                dq_ref,
                phase,
                mean_alpha,
            ]).astype(np.float32)

            if controller == 'open-loop-command' or sd is None:
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

            if spec.get('collect_training', False):
                u_star = corrective_raw_target(v, q_ref, arm_L, arm_R)
                obs_list.append(obs120.copy())
                act_list.append(raw22.copy())
                target_raw_list.append(u_star.copy())

            prev_raw = raw22.copy()
            target = (q_ref + delta).astype(float)
            rt.command(rt.names, target)

        add(
            controls,
            time_s=float(d.time),
            observation120=obs120,
            sensor_valid=pkt is not None,
            flags_L_R=np.array([flag_L, flag_R], dtype=np.float32),
            predicted_alpha_L_R=alpha,
            q_ref22_rad=q_ref,
            raw_residual22=raw22,
            delta22_rad=delta,
            requested22_rad=target,
            accepted22_rad=rt.previous_command,
        )

        sat = np.zeros(22)
        peak = np.zeros(22)
        in_push = push_window[0] <= c < push_window[1]
        for _ in range(80):
            d.xfrc_applied[torso_id, :3] = push_force if in_push else 0.0
            rt.update()
            mujoco.mj_step(m, d)
            physics_steps += 1
            sat += rt.saturated
            peak = np.maximum(peak, np.abs(rt.torque))
        d.xfrc_applied[torso_id, :3] = 0.0
        mujoco.mj_forward(m, d)
        st = sample_state()
        add(native, **st)
        add(controls, substep_saturated_fraction22=sat / 80.0, substep_peak_abs_torque22_Nm=peak)

        if c > 0 and spec.get('collect_training', False):
            pose_sq = float(np.mean((d.qpos[rt.qids] - q_ref) ** 2))
            tilt_sq = float(st['torso_tilt_rad'] ** 2)
            act_sq = float(np.mean(delta ** 2))
            both_feet = float((st['foot_normal_N'] > 1.0).all())
            rew = 1.5 * both_feet - 8.0 * tilt_sq - 12.0 * pose_sq - 0.5 * act_sq
            rew_list.append(rew)

    arrays = {k: np.asarray(v) for k, v in native.items()}
    arrays.update({'control_' + k: np.asarray(v) for k, v in controls.items()})

    trace_ref = None
    if save_trace_path is not None:
        save_trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_trace_path, initial_qpos=initial_qpos, initial_qvel=initial_qvel, **arrays)
        trace_ref = ref(save_trace_path)

    q = arrays['qpos']
    tilt = arrays['torso_tilt_rad']
    pose_err = np.abs(q[1:, rt.qids] - arrays['control_q_ref22_rad'])
    whole_pose_rms = float(np.sqrt(np.mean(pose_err ** 2)))
    tilt_rms = float(np.sqrt(np.mean(tilt ** 2)))
    com_drift = np.linalg.norm(arrays['whole_body_com_m'][:, :2] - init_com_xy, axis=1)
    paw_L_lift_m = float(arrays['paw_positions_m_L_R'][:, 0, 2].max() - arrays['paw_positions_m_L_R'][0, 0, 2])
    paw_R_lift_m = float(arrays['paw_positions_m_L_R'][:, 1, 2].max() - arrays['paw_positions_m_L_R'][0, 1, 2])

    ep_report = dict(
        id=spec['id'],
        controller=controller,
        schedule=schedule_name,
        seed=seed,
        push_force_N=push_force.tolist(),
        push_window_controls=list(push_window),
        push_impulse_Ns=float(np.linalg.norm(push_force[:2]) * max(0, push_window[1] - push_window[0]) * 0.02),
        trace=trace_ref,
        status='completed',
        requested_duration_s=controls_n * 0.02,
        actual_duration_s=float(d.time),
        controls=controls_n,
        native_physics_steps=physics_steps,
        first_evaluator_physical_event_s=first_event,
        whole_pose_rms_rad=whole_pose_rms,
        torso_tilt_rms_rad=tilt_rms,
        max_torso_tilt_rad=float(tilt.max()),
        root_height_m=dist_summary(q[:, 2]),
        torso_tilt_rad=dist_summary(tilt),
        max_horizontal_com_drift_m=float(com_drift.max()),
        final_root_displacement_m=(q[-1, :3] - initial_qpos[:3]).tolist(),
        both_feet_contact_control_fraction=float(np.mean((arrays['foot_normal_N'][1:] > 0).all(1))),
        other_floor_normal_N=dist_summary(arrays['other_floor_normal_N']),
        left_shoulder_pitch_excursion_rad=float(q[:, rt.qids[4]].max() - q[0, rt.qids[4]]),
        right_shoulder_pitch_excursion_rad=float(q[:, rt.qids[0]].max() - q[0, rt.qids[0]]),
        left_paw_vertical_lift_m=paw_L_lift_m,
        right_paw_vertical_lift_m=paw_R_lift_m,
        per_joint_peak_abs_torque_Nm=np.max(arrays['control_substep_peak_abs_torque22_Nm'], axis=0).tolist(),
        per_joint_physics_saturation_fraction=np.mean(arrays['control_substep_saturated_fraction22'], axis=0).tolist(),
        tracking_accuracy_curve=[
            dict(abs_error_radius_rad=r, control_joint_fraction=float(np.mean(pose_err <= r)))
            for r in (0.01, 0.025, 0.05, 0.1, 0.2)
        ],
        wall_s=time.monotonic() - t_start,
    )
    if spec.get('collect_training', False):
        ep_report['train_data'] = dict(
            obs=np.asarray(obs_list, dtype=np.float32),
            act=np.asarray(act_list, dtype=np.float32),
            target_raw=np.asarray(target_raw_list, dtype=np.float32),
            rew=np.asarray(rew_list, dtype=np.float32),
        )
    return ep_report


def build_config() -> dict:
    selected = json.loads(POINTER_PATH.read_text())
    return dict(
        schema='cove50.independent-arm-feedback.config.v1',
        task='T-c3ysnj',
        selected_pointer=ref(POINTER_PATH),
        selected_model=selected,
        materialized_rigid_mjcf=ref(MATERIALIZED_MJCF),
        warm_feedback_checkpoint=ref(WARM_FB_CKPT_PATH),
        recurrent_arm_checkpoint=ref(GRU_CKPT_PATH),
        posture_donor_checkpoint=ref(POSTURE_CKPT_PATH),
        training=dict(
            ppo_rounds=12,
            episodes_per_round=8,
            controls_per_episode=200,
            adam_updates_per_round=20,
            total_adam_updates=240,
            total_training_controls=19200,
            total_training_physics_substeps=1536000,
            learning_rate=3e-4,
            clip_param=0.2,
            residual_scale_rad=0.03,
            seed=250100,
        ),
        evaluation_cases=[
            dict(case_index=0, seed=140701, label='nominal-left-push', schedule='independent-left', joint_sigma_rad=0.0, rpy_envelope_rad=0.0, joint_velocity_sigma_rad_s=0.0, push_force=[3.5, -2.5, 0.0], push_window=[180, 185]),
            dict(case_index=1, seed=140701, label='nominal-right-push', schedule='independent-right', joint_sigma_rad=0.0, rpy_envelope_rad=0.0, joint_velocity_sigma_rad_s=0.0, push_force=[3.5, 2.5, 0.0], push_window=[180, 185]),
            dict(case_index=2, seed=140702, label='varied-140702-left-push', schedule='independent-left', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[4.0, -3.0, 0.0], push_window=[180, 185]),
            dict(case_index=3, seed=140702, label='varied-140702-right-push', schedule='independent-right', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[4.0, 3.0, 0.0], push_window=[180, 185]),
            dict(case_index=4, seed=140703, label='varied-140703-alternating-push', schedule='independent-alternating', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[3.5, -2.5, 0.0], push_window=[150, 155]),
            dict(case_index=5, seed=140704, label='varied-140704-alternating-push', schedule='independent-alternating', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[-3.0, 2.5, 0.0], push_window=[310, 315]),
            dict(case_index=6, seed=140703, label='varied-140703-left-unpushed', schedule='independent-left', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[0.0, 0.0, 0.0], push_window=[0, 0]),
            dict(case_index=7, seed=140704, label='varied-140704-right-unpushed', schedule='independent-right', joint_sigma_rad=0.015, rpy_envelope_rad=0.015, joint_velocity_sigma_rad_s=0.03, push_force=[0.0, 0.0, 0.0], push_window=[0, 0]),
        ],
    )


def execute(evidence_dir: Path, work_dir: Path) -> None:
    t0 = time.monotonic()
    cfg = build_config()
    DATA.mkdir(parents=True, exist_ok=True)
    cfg_path = DATA / 'config.json'
    write_json(cfg_path, cfg)

    warm = torch.load(WARM_FB_CKPT_PATH, map_location='cpu', weights_only=False)
    actor_sd = {k: v.contiguous().clone() for k, v in warm['actor_state_dict'].items()}
    critic_sd = {k: v.contiguous().clone() for k, v in warm['critic_state_dict'].items()}

    trainable_keys = [k for k in actor_sd if k.startswith('mlp.')]
    params = {k: torch.nn.Parameter(actor_sd[k].clone()) for k in trainable_keys}
    actor_sd.update(params)

    critic_keys = [k for k in critic_sd if k.startswith('mlp.')]
    c_params = {k: torch.nn.Parameter(critic_sd[k].clone()) for k in critic_keys}
    critic_sd.update(c_params)

    optimizer = torch.optim.Adam(list(params.values()) + list(c_params.values()), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    torch.manual_seed(cfg['training']['seed'])

    train_schedules = [
        ('independent-left', 140701, [3.5, -2.5, 0.0], [75, 80]),
        ('independent-right', 140701, [3.5, 2.5, 0.0], [75, 80]),
        ('independent-left', 140702, [4.0, -3.0, 0.0], [75, 80]),
        ('independent-right', 140702, [4.0, 3.0, 0.0], [75, 80]),
        ('independent-alternating', 140703, [3.5, -2.5, 0.0], [60, 65]),
        ('independent-alternating', 140704, [-3.0, 2.5, 0.0], [130, 135]),
        ('independent-left', 140703, [0.0, 0.0, 0.0], [0, 0]),
        ('independent-right', 140704, [0.0, 0.0, 0.0], [0, 0]),
    ]

    ppo_rounds = cfg['training']['ppo_rounds']
    updates_per_round = cfg['training']['adam_updates_per_round']
    round_logs = []
    total_train_controls = 0
    total_train_physics = 0
    total_adam_updates = 0

    with cf.ProcessPoolExecutor(max_workers=8, initializer=_init_worker) as pool:
        for r_idx in range(ppo_rounds):
            serial_sd = {k: v.detach().cpu().numpy().tolist() for k, v in actor_sd.items()}
            specs = []
            for w_i, (sched, s_seed, p_force, p_win) in enumerate(train_schedules):
                specs.append(dict(
                    id=f'train-r{r_idx:02d}-w{w_i:02d}',
                    seed=s_seed + r_idx * 100,
                    round_idx=r_idx,
                    schedule=sched,
                    controls=200,
                    controller='independent-arm-feedback',
                    joint_sigma_rad=0.012 if s_seed != 140701 else 0.0,
                    rpy_envelope_rad=0.012 if s_seed != 140701 else 0.0,
                    joint_velocity_sigma_rad_s=0.025 if s_seed != 140701 else 0.0,
                    push_force=p_force,
                    push_window=p_win,
                    actor_sd=serial_sd,
                    explore_sigma=max(0.02, 0.08 * (1.0 - r_idx / ppo_rounds)),
                    collect_training=True,
                ))
            results = list(pool.map(run_single_episode, specs))
            total_train_controls += sum(r['controls'] for r in results)
            total_train_physics += sum(r['native_physics_steps'] for r in results)

            obs_batch = torch.tensor(np.concatenate([r['train_data']['obs'] for r in results], axis=0), dtype=torch.float32)
            act_batch = torch.tensor(np.concatenate([r['train_data']['act'] for r in results], axis=0), dtype=torch.float32)
            ustar_batch = torch.tensor(np.concatenate([r['train_data']['target_raw'] for r in results], axis=0), dtype=torch.float32)
            rew_batch = torch.tensor(np.concatenate([r['train_data']['rew'] for r in results], axis=0), dtype=torch.float32)

            # Update empirical observation normalizer on collected on-policy states
            with torch.no_grad():
                n_new = obs_batch.shape[0]
                old_count = int(actor_sd['obs_normalizer.count'].item())
                new_count = old_count + n_new
                rate = n_new / max(1, new_count)
                mean_x = obs_batch.mean(dim=0, keepdim=True)
                var_x = obs_batch.var(dim=0, unbiased=False, keepdim=True)
                delta_m = mean_x - actor_sd['obs_normalizer._mean']
                actor_sd['obs_normalizer._mean'] = actor_sd['obs_normalizer._mean'] + rate * delta_m
                actor_sd['obs_normalizer._var'] = actor_sd['obs_normalizer._var'] + rate * (var_x - actor_sd['obs_normalizer._var'] + delta_m * (mean_x - actor_sd['obs_normalizer._mean']))
                actor_sd['obs_normalizer._std'] = torch.sqrt(torch.clamp(actor_sd['obs_normalizer._var'], min=1e-6))
                actor_sd['obs_normalizer.count'] = torch.tensor(new_count, dtype=torch.long)

                old_mean = actor_forward(actor_sd, obs_batch).detach()
                returns = (rew_batch - rew_batch.mean()) / (rew_batch.std() + 1e-5)
                adv = torch.clamp(returns, -2.0, 2.0).unsqueeze(1)

            sigma = 0.15
            gen = torch.Generator().manual_seed(cfg['training']['seed'] + r_idx)
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
                mean_tilt_rms_rad=float(np.mean([r['torso_tilt_rms_rad'] for r in results])),
                mean_pose_rms_rad=float(np.mean([r['whole_pose_rms_rad'] for r in results])),
                mean_reward=float(rew_batch.mean().item()),
                final_loss=losses[-1],
            )
            round_logs.append(round_summary)
            print(json.dumps(round_summary), flush=True)

        # Save trained checkpoint
        train_dir = evidence_dir / 'train'
        train_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = train_dir / 'candidate.pt'
        final_actor_sd = {k: v.detach().cpu().clone() for k, v in actor_sd.items()}
        final_critic_sd = {k: v.detach().cpu().clone() for k, v in critic_sd.items()}
        torch.save(
            dict(
                actor_state_dict=final_actor_sd,
                critic_state_dict=final_critic_sd,
                optimizer_state_dict=optimizer.state_dict(),
                config=ref(cfg_path),
                warm_start=ref(WARM_FB_CKPT_PATH),
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

        # Evaluate matched baseline (open-loop-command) vs candidate (independent-arm-feedback) across all 8 10s cases
        eval_dir = evidence_dir / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        serial_final_sd = {k: v.numpy().tolist() for k, v in final_actor_sd.items()}
        eval_specs = []
        for case in cfg['evaluation_cases']:
            for branch, ctrl_name in [('before', 'open-loop-command'), ('after', 'independent-arm-feedback')]:
                ep_id = f"eval-{case['case_index']:02d}-{branch}-{case['schedule']}-{case['seed']}"
                eval_specs.append(dict(
                    id=ep_id,
                    case_index=case['case_index'],
                    branch=branch,
                    label=case['label'],
                    seed=case['seed'],
                    schedule=case['schedule'],
                    controls=500,
                    controller=ctrl_name,
                    joint_sigma_rad=case['joint_sigma_rad'],
                    rpy_envelope_rad=case['rpy_envelope_rad'],
                    joint_velocity_sigma_rad_s=case['joint_velocity_sigma_rad_s'],
                    push_force=case['push_force'],
                    push_window=case['push_window'],
                    actor_sd=serial_final_sd if branch == 'after' else None,
                    explore_sigma=0.0,
                    collect_training=False,
                    save_trace_path=str(eval_dir / f'{ep_id}.npz'),
                ))
        eval_results = list(pool.map(run_single_episode, eval_specs))

    for r in eval_results:
        write_json(eval_dir / f"{r['id']}.json", r)

    paired = []
    for case in cfg['evaluation_cases']:
        ci = case['case_index']
        b_ep = next(r for r in eval_results if r['id'].startswith(f'eval-{ci:02d}-before-'))
        a_ep = next(r for r in eval_results if r['id'].startswith(f'eval-{ci:02d}-after-'))
        paired.append(dict(
            case_index=ci,
            label=case['label'],
            seed=case['seed'],
            schedule=case['schedule'],
            push_impulse_Ns=b_ep['push_impulse_Ns'],
            before_tilt_rms_rad=b_ep['torso_tilt_rms_rad'],
            after_tilt_rms_rad=a_ep['torso_tilt_rms_rad'],
            tilt_rms_reduction_pct=100.0 * (b_ep['torso_tilt_rms_rad'] - a_ep['torso_tilt_rms_rad']) / max(1e-9, b_ep['torso_tilt_rms_rad']),
            before_max_tilt_rad=b_ep['max_torso_tilt_rad'],
            after_max_tilt_rad=a_ep['max_torso_tilt_rad'],
            before_pose_rms_rad=b_ep['whole_pose_rms_rad'],
            after_pose_rms_rad=a_ep['whole_pose_rms_rad'],
            pose_rms_reduction_pct=100.0 * (b_ep['whole_pose_rms_rad'] - a_ep['whole_pose_rms_rad']) / max(1e-9, b_ep['whole_pose_rms_rad']),
            before_max_com_drift_m=b_ep['max_horizontal_com_drift_m'],
            after_max_com_drift_m=a_ep['max_horizontal_com_drift_m'],
            after_left_shoulder_rad=a_ep['left_shoulder_pitch_excursion_rad'],
            after_right_shoulder_rad=a_ep['right_shoulder_pitch_excursion_rad'],
        ))

    eval_controls = sum(r['controls'] for r in eval_results)
    eval_physics = sum(r['native_physics_steps'] for r in eval_results)
    report = dict(
        schema='cove50.independent-arm-feedback.report.v1',
        status='completed',
        config=ref(cfg_path),
        candidate=ref(ckpt_path),
        warm_start=ref(WARM_FB_CKPT_PATH),
        recurrent_arm_donor=ref(GRU_CKPT_PATH),
        accounting=dict(
            parallel_world_slots=8,
            training_episodes=ppo_rounds * len(train_schedules),
            evaluation_episodes=len(eval_results),
            total_episodes=ppo_rounds * len(train_schedules) + len(eval_results),
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
    write_json(evidence_dir / 'report.json', report)
    write_json(DATA / 'summary.json', dict(
        report=ref(evidence_dir / 'report.json'),
        candidate=ref(ckpt_path),
        accounting=report['accounting'],
        paired_comparisons=paired,
    ))
    print(json.dumps(dict(status='completed', candidate=ref(ckpt_path), accounting=report['accounting'], paired_count=len(paired))), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--evidence', type=Path, default=os.environ.get('LAB_EVIDENCE_DIR'))
    p.add_argument('--work', type=Path, default=os.environ.get('LAB_WORK_DIR'))
    args = p.parse_args()
    if not args.execute:
        cfg = build_config()
        DATA.mkdir(parents=True, exist_ok=True)
        write_json(DATA / 'config.json', cfg)
        print(json.dumps(dict(config=ref(DATA / 'config.json'), activity='config frozen only')))
        return
    if not args.evidence or not args.work:
        p.error('Requires --evidence and --work (or LAB_EVIDENCE_DIR / LAB_WORK_DIR)')
    execute(Path(args.evidence).resolve(), Path(args.work).resolve())


if __name__ == '__main__':
    main()
