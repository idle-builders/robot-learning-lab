"""Upstream MJLab BeyondMimic tracking task on the selected APE0 R3 boundary.

The native task terms, rewards, adaptive sampling and PPO runner are unchanged.
This module supplies only the exact R3 entity and numerical options, the R3
delayed-feedback actor observation, a position action that arms the R3 boundary
once per physics step, R3 body/geom names and the narrowed ranges in RANGES.
Upstream already resets the R3 boundary when the motion command teleports a
world (MotionCommand._write_reference_state_to_sim -> Entity.reset).
"""
from dataclasses import dataclass, fields, replace
from functools import partial
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul
from mjlab.utils.spec_config import CollisionCfg

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'artifacts/robot/model/timestep'
RUNTIME = ROOT / 'control/common/plant'

# Narrow ranges (Idle taste M-mx9gnr): none is identified on hardware yet.
# Lengths/linear speeds use the G1 BeyondMimic value x HEIGHT_RATIO (APE0
# 0.5 m / G1 1.32 m, rounded); dimensionless angles keep the G1 value.
HEIGHT_RATIO = .4
_G1_VELOCITY = dict(x=.5, y=.5, z=.2, roll=.52, pitch=.52, yaw=.78)
VELOCITY = {k: (-v*HEIGHT_RATIO, v*HEIGHT_RATIO) for k, v in _G1_VELOCITY.items()}
RANGES = dict(
    push_velocity=dict(value=VELOCITY, source='G1 BeyondMimic VELOCITY_RANGE x HEIGHT_RATIO; disturbance challenge, not model randomization'),
    reset_velocity=dict(value=VELOCITY, source='same scaled range for start-state variation around the reference'),
    reset_pose=dict(value=dict(x=(-.02, .02), y=(-.02, .02), z=(-.004, .004), roll=(-.1, .1), pitch=(-.1, .1), yaw=(-.2, .2)),
        source='G1 pose_range; lengths x HEIGHT_RATIO, angles unchanged'),
    reset_joint_rad=dict(value=(-.1, .1), source='G1 joint_position_range unchanged'),
    foot_friction=dict(value=(.7, .9), source='R3 collision friction .8 (mjlab_adapter) +-.1; assumption, unmeasured'),
    torso_com_offset_m=dict(value=(-.005, .005), source='assumption: assembly/ballast tolerance; deliberately narrower than scaled G1 (+-1-2 cm)'),
    joint_vel_noise_rad_s=dict(value=(-.1, .1), source='assumption: servo speed-report noise; boundary qd is otherwise exact'),
    removed=dict(value=['encoder_bias'], source='boundary owns the measurement (4096-count quantization, IMU bias/noise, 10 ms delay)'),
)
TRACKED_BODIES = ('torso', 'R_upper_arm', 'R_forearm', 'L_upper_arm', 'L_forearm', 'head',
    'R_roll_link', 'R_shin', 'R_foot', 'L_roll_link', 'L_shin', 'L_foot')
END_EFFECTORS = ('R_foot', 'L_foot', 'R_forearm', 'L_forearm')
# G1 0.25 m anchor/end-effector height error x HEIGHT_RATIO.
TERMINATION_HEIGHT_M = .1
# Tracking reward widths, exp(-error/std**2). 'g1' is the native config. 'r3'
# scales lengths and linear speeds by HEIGHT_RATIO as in RANGES; angles and
# angular velocities keep the G1 value. Over 2 s from any start of the R3
# forward walk, standing still earns 88% of the root position, body position
# and body velocity credit with the G1 widths and 64% with the R3 widths.
_G1_STD = dict(motion_global_root_pos=.3, motion_global_root_ori=.4, motion_body_pos=.3, motion_body_ori=.4,
    motion_body_lin_vel=1., motion_body_ang_vel=3.14)
_LENGTHS = ('motion_global_root_pos', 'motion_body_pos', 'motion_body_lin_vel')
REWARD_STD = dict(g1=_G1_STD, r3={k: round(v * HEIGHT_RATIO, 3) if k in _LENGTHS else v for k, v in _G1_STD.items()})
# Native tracking base scale. The G1 rule .25*effort/stiffness would give only
# 0.03 rad for the R3 servo law (rated .98 N m, kp 8 N m/rad).
ACTION_SCALE = .5
SOFT_LIMIT_FACTOR = 1.
# Servo law on the 20 C047 joints (T-gtxyrn direction check). 'c047' is the
# selected ST-3215-C047. 'sts3250' puts the Feetech STS3250 catalog values on
# the same joints without changing the CAD: same 45.2x24.7x35 mm case and
# 1/345 gear, 4.9 N m stall and 0.133 s/60 deg at 12 V, 74.5 g, 4.2 A stall,
# 24 mA idle (servodatabase.com/servo/feetech/sts3250). No rated torque is
# published, so it keeps C047's stall/3. Simulated kp/kd scale with stall
# torque, as the same firmware duty per count gives proportionally more torque.
_STS3250_STALL = 4.9
ACTUATORS = dict(c047=None, sts3250=dict(mass_kg=.0745, law=dict(stall_torque_Nm=_STS3250_STALL,
    rated_torque_Nm=_STS3250_STALL / 3, no_load_speed_rad_s=math.pi / 3 / .133, idle_current_A=.024,
    stall_current_A=4.2, gain_scale=_STS3250_STALL / 2.941995)),
    # Waveshare ST3025 (brushless serial-bus servo), r4 dynamics only: its law, mass and BAM-identified
    # rotor inertia come from robot/model/dynamics/actuator-dynamics.json.
    st3025=dict(mass_kg=.124, r4_only=True),
    # Mixed sizing, r4 only: each joint's servo, law and mass from walk_resume dynamics.ALLOCATIONS.
    **{'st3025-pitch-roll-sts3250': dict(r4_only=True, mixed=True), 'st3025-pitch-roll-sts3250-ak45knee': dict(r4_only=True, mixed=True),
    'st3025-pitch-roll-sts3250-geared-knee': dict(r4_only=True, mixed=True),
    'qdd-edulite-gim': dict(r4_only=True, mixed=True), 'qdd-damiao-gim': dict(r4_only=True, mixed=True),
    'tier3-st3025-sts3250-c047': dict(r4_only=True, mixed=True)})
# Plant dynamics: 'r3' is the R3 assumed PD law with no rotor inertia (all runs to 2026-11-04);
# 'r4' layers the best-estimate actuator dynamics sidecar (Feetech firmware duty loop, rotor
# inertia, Coulomb friction, backlash) on the same body. DYNAMICS_LEVELS names sensitivity
# parameters for walk_resume dynamics.resolve.
DYNAMICS = ('r3', 'r4')
# Body holding each C047 case (link-inertias.json part names): the parent
# link, except the case-driven elbows and the foot-roll servos in the child.
_CASE_IN_CHILD = ('R_elbow', 'L_elbow', 'R_ankle_roll', 'L_ankle_roll')
SERVO_CASE_M = (.0452, .0247, .035)


def runtime():
    """The walk-resume R3 runtime package, shared under its existing module name."""
    name = 'cove50_r3_runtime'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, RUNTIME / '__init__.py', submodule_search_locations=[str(RUNTIME)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runtime_dynamics():
    """The walk-resume runtime's r4 dynamics module."""
    runtime()
    return importlib.import_module('cove50_r3_runtime.dynamics')


def scale_feet(spec, scale):
    """Scale both feet's collision meshes by scale = (length, width) about the foot body origin (the ankle frame).

    A design lever: longer and wider soles enlarge the support polygon, so the floor can carry hops, quick crouches
    and fast bounces it could not (agility demand check). The feet's mass and visual meshes are unchanged.
    """
    if scale is None:
        return spec
    sx, sy = scale
    for mesh in spec.meshes:
        if mesh.name.startswith(('c_R_foot', 'c_L_foot')):
            mesh.scale = [sx, sy, 1.0]
    return spec


def _scaled_feet_spec(spec_fn, scale):
    return scale_feet(spec_fn(), scale)


def scale_arms(spec, scale):
    """Stretch both arms' upper-arm (shoulder yaw to elbow) and forearm (elbow to knuckle) segments by `scale` along
    the hanging arm (body z), about each segment's joint: the elbow moves down, meshes stretch. `scale` is one factor
    for both segments or (upper arm, forearm).

    A design lever: arm reach 0.22 m becomes 0.22 + 0.135 (scale - 1) m, so the knuckles reach the floor in a
    bent-knee forward lean (scale 1.5: torso 72 deg, hips 102 deg, within the 110 deg hip range). Each segment's
    structure (its mass less the C047 case the forearm carries at the elbow) scales in mass with length, its centre of
    mass moves out with it, and its inertia about the transverse axes grows as mass x length^2 (scale^3); the case
    stays at the elbow. Servo models heavier than the C047 add their mass after this (heavier_servos).
    """
    if not scale or scale == 1:
        return spec
    upper, forearm = (float(scale), float(scale)) if np.isscalar(scale) else map(float, scale)
    case = json.loads((MODEL / 'actuators.json').read_text())['models']['c047']['mass_kg']
    for side in 'RL':
        fore = spec.body(f'{side}_forearm')
        x, y, z = fore.pos
        fore.pos = [x, y, z * upper]
        for part, case_kg, s in (('upper_arm', 0., upper), ('forearm', case, forearm)):
            b = spec.body(f'{side}_{part}')
            names = {g.meshname for g in b.geoms}
            for mesh in spec.meshes:
                if mesh.name in names:
                    mesh.scale = [1.0, 1.0, s]
            m, c, f = b.mass, np.array(b.ipos), b.fullinertia
            inertia = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
            skew = lambda r: r @ r * np.eye(3) - np.outer(r, r)
            ms = m - case_kg  # structure; the case is a point at the joint (body origin)
            cs = m * c / ms
            Is = inertia + m * skew(c) - ms * skew(cs)  # structure about its own centre (the case adds none at the origin)
            # Stretching z by s with mass growing by s: the second moments sum(r r^T dm) become s D S D with
            # D = diag(1, 1, s), and the inertia follows from them (so it stays physically valid).
            D = np.diag([1.0, 1.0, s])
            second = s * D @ (.5 * np.trace(Is) * np.eye(3) - Is) @ D
            Is = np.trace(second) * np.eye(3) - second
            ms2, cs2 = ms * s, cs * np.array([1.0, 1.0, s])
            mass = ms2 + case_kg
            com = ms2 * cs2 / mass
            total = Is + ms2 * skew(cs2 - com) + case_kg * skew(-com)
            b.mass, b.ipos = mass, com
            b.fullinertia = [total[0, 0], total[1, 1], total[2, 2], total[0, 1], total[0, 2], total[1, 2]]
    return spec


def _scaled_arms_spec(spec_fn, scale):
    return scale_arms(spec_fn(), scale)


def add_fists(spec, length, mass_kg, width=.026):
    """A knuckle fist on each forearm: a `width`-square block `length` long under the forearm tip (the lowest point
    of its collision meshes), with collision, a visual copy and `mass_kg` added to the forearm at the block's centre.

    The knuckle-support lever: a light fist adds reach where the arm meets the floor without moving the elbow case or
    the forearm structure outward, and gives knuckle-walking a contact pad. Call after scale_arms.
    """
    if not length:
        return spec
    m = spec.copy().compile()  # compiling the spec itself would hide elements added afterwards from spec.geom(name)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    for side in 'RL':
        b = m.body(f'{side}_forearm').id
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                pts.append(m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + m.geom_pos[g])
        pts = np.concatenate(pts)
        tip_z = pts[:, 2].min()
        cx, cy = pts[pts[:, 2] < tip_z + .003, :2].mean(0)
        h, top = width / 2, tip_z + .004  # overlaps the forearm tip by 4 mm
        verts = [(cx + sx * h, cy + sy * h, z) for z in (top, tip_z - length) for sx in (-1, 1) for sy in (-1, 1)]
        mesh = spec.add_mesh(name=f'{side}_fist', uservert=np.array(verts, float).ravel().tolist())
        body = spec.body(f'{side}_forearm')
        body.add_geom(name=f'collision_{side}_fist', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, contype=1,
            conaffinity=1, group=3, density=0)
        body.add_geom(name=f'{side}_fist_visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, contype=0,
            conaffinity=0, group=1, density=0, rgba=[.86, .82, .7, 1])
        point = np.array([cx, cy, (top + tip_z - length) / 2])
        mb, c, f = body.mass, np.array(body.ipos), body.fullinertia
        inertia = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
        skew = lambda r: r @ r * np.eye(3) - np.outer(r, r)
        mass = mb + mass_kg
        com = (mb * c + mass_kg * point) / mass
        own = mass_kg / 12 * np.diag([width ** 2 + length ** 2, width ** 2 + length ** 2, 2 * width ** 2])
        total = inertia + mb * skew(c - com) + mass_kg * skew(point - com) + own
        body.mass, body.ipos = mass, com
        body.fullinertia = [total[0, 0], total[1, 1], total[2, 2], total[0, 1], total[0, 2], total[1, 2]]
    return spec


def _fisted_spec(spec_fn, fist):
    return add_fists(spec_fn(), *fist)


def heavier_servos(spec_fn, extra_kg):
    """The selected R3 spec with extra_kg per C047 at its joint anchor in the body holding its case.

    extra_kg may be a {joint: kg} map (mixed sizing); joints without an entry keep the C047 mass.

    The case's own offset from the axis and its orientation are not modelled:
    the extra mass is a point at the anchor plus an isotropic box inertia.
    """
    spec = spec_fn()
    per_joint = extra_kg if isinstance(extra_kg, dict) else None
    skew = lambda r: r @ r * np.eye(3) - np.outer(r, r)
    for axis in json.loads((MODEL / 'actuators.json').read_text())['actuators']:
        if axis['model'] != 'c047':
            continue
        if per_joint is not None:
            extra_kg = per_joint.get(axis['joint'], 0.)
            if not extra_kg:
                continue
        box = extra_kg / 12 * 2 * sum(d * d for d in SERVO_CASE_M) / 3
        joint, source = spec.joint(axis['joint']), axis['source_axis']
        child = spec.body(source['child'])
        anchor = np.array(joint.pos)
        if axis['joint'] in _CASE_IN_CHILD:
            body, point = child, anchor
        else:
            body, point = spec.body(source['parent']), np.zeros(3)
            mujoco.mju_rotVecQuat(point, anchor, np.array(child.quat))
            point += np.array(child.pos)
        m, c, f = body.mass, np.array(body.ipos), body.fullinertia
        inertia = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
        mass = m + extra_kg
        com = (m * c + extra_kg * point) / mass
        inertia = inertia + m * skew(c - com) + extra_kg * skew(point - com) + box * np.eye(3)
        body.mass, body.ipos = mass, com
        body.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    return spec


def measured(env, start, stop):
    """Delayed R3 feedback packet [N,50]: q22, qd22, gyro3, gravity estimate3."""
    return runtime().get_runtime(env).last_values[:, start:stop]


def joint_position_tracking(env, std):
    """exp(-mean over joints of the squared joint-angle error to the reference / std^2).

    The upstream terms score body positions and orientations, so a teacher can
    stand tall through a squat at little cost and nothing marks the knees it
    left straight; this term pays for following the reference joint angles.
    """
    command = env.command_manager.get_term('motion')
    error = (command.robot_joint_pos - command.joint_pos).square().mean(-1)
    return torch.exp(-error / std ** 2)


def knee_tracking(env, std):
    """exp(-mean over both knees of the squared knee-angle error to the reference / std^2).

    joint_position_tracking averages all 22 joints at a 0.5 rad width, so a knee held 0.15 rad straighter
    than the reference costs almost nothing there; this narrower term pays for the reference's bent-knee stance.
    """
    command = env.command_manager.get_term('motion')
    ids = [i for i, name in enumerate(env.scene['robot'].joint_names) if name.endswith('_knee')]
    error = (command.robot_joint_pos[:, ids] - command.joint_pos[:, ids]).square().mean(-1)
    return torch.exp(-error / std ** 2)


def reference_preview(env, steps):
    """Reference joint angles `steps` control frames ahead, relative to the current reference frame (rad).

    The actor otherwise sees only the reference's current angles and speeds, so the 20 ms command delay and the
    servo's own response show up as a fixed lag behind fast moves. The motion plan is known in advance on the robot,
    so a short preview is deployable and lets the policy command ahead of the reference.
    """
    c = env.command_manager.get_term('motion')
    last = c.motion.time_step_total - 1
    now = c.motion.joint_pos[c.time_steps]
    return torch.cat([c.motion.joint_pos[(c.time_steps + k).clamp(max=last)] - now for k in steps], -1)


def preview_statistics(joint_pos, steps):
    """Mean and variance over the tape of each reference_preview column (joint_pos: the tape's [T, joints] angles)."""
    last = len(joint_pos) - 1
    t = torch.arange(len(joint_pos), device=joint_pos.device)
    x = torch.cat([joint_pos[(t + k).clamp(max=last)] - joint_pos for k in steps], -1)
    return x.mean(0), x.var(0)


class AdaptiveTrackingExp(ManagerTermBase):
    """exp(-e / s^2) whose width s follows the population's current error (adaptive tracking width, as in PBHC).

    e is the term's squared tracking error per world, for kind joint_pos (mean over joints), joint_max (the worst
    joint's squared error, so one straight knee cannot hide in a 22-joint mean), joint_vel (mean over joints),
    body_pos and body_lin_vel (mean over the tracked bodies of the squared xyz error), feet_pos (the two feet
    only), anchor_z or anchor_vz (the torso's height or vertical speed) or anchor_xy (its ground position). As in PBHC (arXiv 2506.12851, Eq. 7-8), an exponential moving average of the population's mean e
    (`rate` per control step) is kept, and s^2 = min(s^2, max(EMA, min_std^2)): the width only shrinks, from
    init_std down to min_std. With fixed widths far above the remaining error every world sits near the
    term's maximum (the reward-set v1 terms logged 93-99% of their weight), so the error the policy still makes,
    straight knees, low feet, half-speed joints, costs almost nothing; a width that follows the error keeps the
    kernel's slope where the policy is.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.s2 = torch.tensor(float(cfg.params['init_std']) ** 2, device=env.device)
        self.ema = self.s2.clone()
        self.feet = None

    def error(self, env, kind):
        c = env.command_manager.get_term('motion')
        if kind == 'joint_pos':
            return (c.robot_joint_pos - c.joint_pos).square().mean(-1)
        if kind == 'joint_max':
            return (c.robot_joint_pos - c.joint_pos).square().amax(-1)
        if kind == 'joint_vel':
            return (c.robot_joint_vel - c.joint_vel).square().mean(-1)
        if kind == 'body_pos':
            return (c.body_pos_relative_w - c.robot_body_pos_w).square().sum(-1).mean(-1)
        if kind == 'body_lin_vel':
            return (c.body_lin_vel_w - c.robot_body_lin_vel_w).square().sum(-1).mean(-1)
        if kind == 'feet_pos':
            if self.feet is None:
                self.feet = [list(c.cfg.body_names).index(n) for n in ('R_foot', 'L_foot')]
            return (c.body_pos_relative_w[:, self.feet] - c.robot_body_pos_w[:, self.feet]).square().sum(-1).mean(-1)
        if kind == 'anchor_z':  # torso height: the body_pos mean over 12 bodies at a 2 cm floor leaves a 1 cm hop unpaid
            return (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).square()
        if kind == 'anchor_vz':  # torso vertical speed: take-off and landing
            return (c.anchor_lin_vel_w[:, 2] - c.robot_anchor_lin_vel_w[:, 2]).square()
        if kind == 'anchor_xy':  # torso ground position: travel along the reference path (the body terms are relative)
            return (c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2]).square().sum(-1)
        raise ValueError(f'unknown tracking kind {kind}')

    def __call__(self, env, kind, init_std, min_std, rate=1e-3):
        e = self.error(env, kind)
        with torch.no_grad():
            self.ema = (1 - rate) * self.ema + rate * e.mean()
            self.s2 = torch.minimum(self.s2, self.ema.clamp(min=min_std ** 2))
        return torch.exp(-e / self.s2)

    @property
    def std(self):
        return float(self.s2.sqrt())


class AdaptiveJointExp(ManagerTermBase):
    """Per-joint kernels for one joint group, each with its own shrink-only adaptive width (reward set v3).

    reward = exp(-sum_j w_j e_j / s_j^2 / sum_j w_j) over the group's joints (legs: hips, knees, ankles; arms:
    shoulders, elbows). Each joint's error is normalised by its own width, which follows the population's mean error
    of that joint, so every joint counts on its own scale; one exponential over the normalised mean keeps the
    gradient of a joint that is far off (separate per-joint kernels would cap its cost at 1/n and flatten to zero).
    kind 'pos' tracks angles (w_j = 1); kind 'vel' tracks joint speeds with w_j = 1 + |reference speed_j| / speed_scale,
    so the frames and joints where the reference moves fast carry the reward: agility is matching the fast phases.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        names = env.scene['robot'].joint_names
        keys = ('hip', 'knee', 'ankle') if cfg.params['group'] == 'legs' else ('shoulder', 'elbow')
        self.ids = [i for i, n in enumerate(names) if any(k in n for k in keys)]
        s2 = float(cfg.params['init_std']) ** 2
        self.s2 = torch.full((len(self.ids),), s2, device=env.device)
        self.ema = self.s2.clone()

    def __call__(self, env, group, kind, init_std, min_std, speed_scale=2.0, rate=1e-3):
        c = env.command_manager.get_term('motion')
        if kind == 'pos':
            e = (c.robot_joint_pos[:, self.ids] - c.joint_pos[:, self.ids]).square()
            w = torch.ones_like(e)
        else:
            ref = c.joint_vel[:, self.ids]
            e = (c.robot_joint_vel[:, self.ids] - ref).square()
            w = 1 + ref.abs() / speed_scale
        with torch.no_grad():
            self.ema = (1 - rate) * self.ema + rate * e.mean(0)
            self.s2 = torch.minimum(self.s2, self.ema.clamp(min=min_std ** 2))
        return torch.exp(-(w * e / self.s2).sum(-1) / w.sum(-1))

    @property
    def std(self):
        return [round(float(v), 4) for v in self.s2.sqrt()]


# Ground under the reference and under the robot (terrain: ErrorAwareMotionCommand with cfg.terrain; else the plane).
GROUND_SENSOR = 'ground'
GROUND_FRAMES = ('torso', 'R_foot', 'L_foot', 'R_forearm', 'L_forearm')


def _ref_ground(c, bodies):
    """Ground (m, world z) under the reference's `bodies` (indices into body_names), [worlds, len(bodies)]."""
    if hasattr(c, 'reference_ground'):
        return c.reference_ground(bodies)
    return c._env.scene.env_origins[:, 2:3].expand(-1, len(bodies))


def _robot_ground(c, names):
    return c.robot_ground(names) if hasattr(c, 'robot_ground') else c._env.scene.env_origins[:, 2:3].expand(-1, len(names))


class FootContactMatch(ManagerTermBase):
    """1 - mean over both feet of |robot contact - reference contact| (PBHC's contact mask, reward set v3).

    A foot is in contact when its origin is within `band` m of its stance height, the lowest height the reference
    gives it over the whole tape. It pays for keeping planted feet down and for leaving the floor when the
    reference does, so flight phases are rewarded instead of only risked.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        c = env.command_manager.get_term('motion')
        self.feet = [list(c.cfg.body_names).index(n) for n in ('R_foot', 'L_foot')]
        self.h0 = torch.quantile(c.motion.body_pos_w[:, self.feet, 2], .05, dim=0)

    def __call__(self, env, band=.005):
        c = env.command_manager.get_term('motion')
        ref = c.body_pos_w[:, self.feet, 2] - _ref_ground(c, self.feet) < self.h0 + band
        rob = c.robot_body_pos_w[:, self.feet, 2] - _robot_ground(c, ('R_foot', 'L_foot')) < self.h0 + band
        return 1 - (ref != rob).float().mean(-1)


class HandContactMatch(ManagerTermBase):
    """1 - mean over both hands of |robot knuckle contact - reference knuckle contact|, for knuckle-support moves.

    The knuckle point is the mean of the forearm collision mesh's distal end (its lowest 3 mm in the forearm frame,
    as body.knuckle_points), placed by each forearm's pose: the reference's from the tape, the robot's from the
    simulation. A knuckle within `band` m of the floor is in contact. On tapes without knuckle moves both are always
    off and the term is a constant, so it is only used with them (reward set v3k).
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        c = env.command_manager.get_term('motion')
        self.arms = [list(c.cfg.body_names).index(n) for n in ('R_forearm', 'L_forearm')]
        m = env.sim.mj_model
        local = []
        for side in 'RL':
            b = m.body(f'robot/{side}_forearm').id
            pts = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, m.geom_quat[g])
                    pts.append(m.mesh_vert[a:a + n] @ R.reshape(3, 3).T + m.geom_pos[g])
            pts = np.concatenate(pts)
            local.append(pts[pts[:, 2] < pts[:, 2].min() + .003].mean(0))
        self.k = torch.tensor(np.array(local), dtype=torch.float32, device=env.device)  # [2, 3] forearm frame

    def knuckle_z(self, pos, quat):
        return pos[..., 2] + quat_apply(quat, self.k.expand(pos.shape[0], 2, 3))[..., 2]

    def __call__(self, env, band=.006):
        c = env.command_manager.get_term('motion')
        ref = self.knuckle_z(c.body_pos_w[:, self.arms], c.body_quat_w[:, self.arms]) - _ref_ground(c, self.arms) < band
        rob = (self.knuckle_z(c.robot_body_pos_w[:, self.arms], c.robot_body_quat_w[:, self.arms])
               - _robot_ground(c, ('R_forearm', 'L_forearm')) < band)
        return 1 - (ref != rob).float().mean(-1)


class SwingHeight(ManagerTermBase):
    """Swing-foot height tracking, only while the reference lifts that foot more than `lift` m (reward set v3).

    exp(-dz^2 / s^2) per swinging foot with a shrink-only adaptive width, averaged over the swinging feet; 0 when no
    foot swings, so standing never saturates it. It pays for full foot clearance and for the hop's rise.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        c = env.command_manager.get_term('motion')
        self.feet = [list(c.cfg.body_names).index(n) for n in ('R_foot', 'L_foot')]
        self.h0 = torch.quantile(c.motion.body_pos_w[:, self.feet, 2], .05, dim=0)
        self.s2 = torch.tensor(float(cfg.params['init_std']) ** 2, device=env.device)
        self.ema = self.s2.clone()

    def __call__(self, env, init_std, min_std, lift=.008, rate=1e-3):
        c = env.command_manager.get_term('motion')
        rz = c.body_pos_w[:, self.feet, 2] - _ref_ground(c, self.feet)
        z = c.robot_body_pos_w[:, self.feet, 2] - _robot_ground(c, ('R_foot', 'L_foot'))
        swing = (rz > self.h0 + lift).float()
        e = (z - rz).square()
        n = swing.sum()
        with torch.no_grad():
            if n > 0:
                self.ema = (1 - rate) * self.ema + rate * (e * swing).sum() / n
                self.s2 = torch.minimum(self.s2, self.ema.clamp(min=min_std ** 2))
        return (swing * torch.exp(-e / self.s2)).sum(-1) / swing.sum(-1).clamp(min=1)

    @property
    def std(self):
        return float(self.s2.sqrt())


def adaptive_widths(env):
    """{reward term: current width} for the AdaptiveTrackingExp terms of a built environment (for reports)."""
    manager = env.unwrapped.reward_manager
    return {name: (cfg.func.std if isinstance(cfg.func.std, list) else round(cfg.func.std, 5)) for name, cfg in zip(manager._term_names, manager._term_cfgs)
            if isinstance(cfg.func, (AdaptiveTrackingExp, AdaptiveJointExp, SwingHeight))}


def set_adaptive_widths(env, widths):
    """Start the adaptive terms at given widths, e.g. a continued teacher's last ones; they stay shrink-only.

    Without this a continued teacher restarts every width at init_std and spends its first few hundred updates
    re-tightening the reward it was already trained under (2026-11-09: a run on the 3.6x slower obstacle world ended
    mid-way, at about twice its teacher's widths, and tracked looser on every motion). Returns the widths now in use."""
    manager = env.unwrapped.reward_manager
    for name, cfg in zip(manager._term_names, manager._term_cfgs):
        f = cfg.func
        if name in widths and isinstance(f, (AdaptiveTrackingExp, AdaptiveJointExp, SwingHeight)):
            w = torch.as_tensor(widths[name], dtype=f.s2.dtype, device=f.s2.device).reshape(f.s2.shape)
            f.s2 = torch.minimum(f.s2, w.square())
            f.ema = f.s2.clone()
    return adaptive_widths(env)


# Reward set v2: adaptive (shrink-only) widths on the joint, body-position and body-velocity terms, plus worst-joint,
# joint-velocity and foot tracking (PBHC, ASAP and SONIC track these; v1 had none of them). (kind, init_std, min_std,
# weight); init_std is the v1 width (r3 lengths) where v1 had the term, so v2 starts where v1 was.
REWARD_SET_V2 = dict(
    motion_body_pos=('body_pos', .12, .02, 1.0),
    motion_body_lin_vel=('body_lin_vel', .4, .1, 1.0),
    joint_position_tracking=('joint_pos', .5, .05, 1.0),
    joint_max_tracking=('joint_max', .5, .05, 1.0),
    joint_velocity_tracking=('joint_vel', 3.0, .5, .5),
    feet_position_tracking=('feet_pos', .12, .01, 1.0),
)


# Reward set v3: built around what creates agility. Joint terms per group (legs, arms) with per-joint adaptive widths,
# so arms and legs do not trade and one joint cannot hide in a mean; leg joint speed weighted toward the reference's
# fast phases; foot contact phases (planted feet stay down, flight is paid) and swing height. Body-position, body-
# velocity and feet terms as in v2. name: (class, params, weight).
REWARD_SET_V3 = dict(
    motion_body_pos=(AdaptiveTrackingExp, dict(kind='body_pos', init_std=.12, min_std=.02), 1.0),
    motion_body_lin_vel=(AdaptiveTrackingExp, dict(kind='body_lin_vel', init_std=.4, min_std=.1), 1.0),
    feet_position_tracking=(AdaptiveTrackingExp, dict(kind='feet_pos', init_std=.12, min_std=.01), 1.0),
    leg_joint_pos=(AdaptiveJointExp, dict(group='legs', kind='pos', init_std=.5, min_std=.04), 1.0),
    arm_joint_pos=(AdaptiveJointExp, dict(group='arms', kind='pos', init_std=.5, min_std=.05), .5),
    leg_joint_vel=(AdaptiveJointExp, dict(group='legs', kind='vel', init_std=3.0, min_std=.5, speed_scale=2.0), 1.0),
    arm_joint_vel=(AdaptiveJointExp, dict(group='arms', kind='vel', init_std=3.0, min_std=.5, speed_scale=2.0), .3),
    foot_contact_match=(FootContactMatch, dict(band=.005), .5),
    # lift: the reference foot height above stance at which the term turns on. 8 mm missed the knuckle walk's 12 mm
    # swings (its trained feet dragged); 5 mm sees them.
    swing_height=(SwingHeight, dict(init_std=.03, min_std=.005, lift=.005), .5),
)
# v4: v3 plus the torso's height and vertical speed on their own. Under body_pos (a mean over 12 bodies, 2 cm floor)
# the 1 cm hop's torso error is almost free; with the v2 teacher the torso height error is 1.2-1.7 mm rms on bounces,
# squats and the walk but 7.8 mm on the hop, so these terms pay mostly for take-off and flight.
# A 'k' suffix (v3k, v4k, v5k) adds knuckle contact, for tapes with knuckle-support moves (long arms).
HAND_CONTACT_TERM = dict(hand_contact_match=(HandContactMatch, dict(band=.006), .5))
REWARD_SET_V4 = dict(REWARD_SET_V3,
    torso_height=(AdaptiveTrackingExp, dict(kind='anchor_z', init_std=.03, min_std=.005), .5),
    torso_vertical_speed=(AdaptiveTrackingExp, dict(kind='anchor_vz', init_std=.3, min_std=.05), .5),
)
# v5: v4 with the global root position on a shrink-only adaptive width (11.4 cm, the r3 width, down to 3 cm) in place
# of the fixed upstream term. Every body term is relative to the robot's own torso, so only this term pays for
# travel; at 11.4 cm a knuckle walk that steps in place (1.6 of 7.5 cm) scores 0.87 against 0.94-0.98 for still moves,
# at 3 cm 0.36 against 0.77-0.80 (loop2-ape-body-r3-damiao traces). Ground position only; height is in v4.
REWARD_SET_V5 = dict(REWARD_SET_V4,
    motion_global_root_pos=(AdaptiveTrackingExp, dict(kind='anchor_xy', init_std=.114, min_std=.03), .5),
)
# v3r: v3 with v5's adaptive global root position and no torso terms (separates the two v5 changes).
REWARD_SET_V3R = dict(REWARD_SET_V3, motion_global_root_pos=REWARD_SET_V5['motion_global_root_pos'])
# v6: v5 with the global root weight doubled (0.5 -> 1.0). On the ape body v5's torso terms gave full hop flight and
# closer knuckle moves, but doubled the forward walk's drift (16.9 -> 33.8 cm against the root-only run).
REWARD_SET_V6 = dict(REWARD_SET_V5, motion_global_root_pos=(*REWARD_SET_V5['motion_global_root_pos'][:2], 1.0))
REWARD_SETS = dict(v3=REWARD_SET_V3, v3r=REWARD_SET_V3R, v4=REWARD_SET_V4, v5=REWARD_SET_V5, v6=REWARD_SET_V6)


def target_beyond_stop(env):
    """rad by which the commanded joint targets lie past the joint stops, summed over joints.

    The boundary clamps commands to the stops (as servo firmware angle limits
    do), so every target past a stop acts the same and nothing pulls a policy
    back once it drifts there; this penalty keeps commands where they matter.
    """
    return runtime().get_runtime(env).beyond_stop.sum(1)


class R3PositionAction(JointPositionAction):
    """Native joint-position action that drives the R3 boundary.

    residual=True makes the action an offset on the current reference joint
    angles (target = reference + scale x action) instead of on the default
    pose, so a zero action already commands the motion and doing less than
    the reference takes effort (Any2Track-style residual tracking).
    """

    def process_actions(self, actions):
        super().process_actions(actions)
        reference = self._env.command_manager.get_term('motion').joint_pos[:, self._target_ids]
        if self.cfg.residual:
            self._processed_actions[:] = self._processed_actions - self._offset + reference
        # No feedback packet has arrived in the first control period after a
        # reset; hold the reference posture rather than act on zeroed inputs.
        valid = runtime().get_runtime(self._env).valid[:, None]
        self._processed_actions[:] = torch.where(valid, self._processed_actions, reference)

    def apply_actions(self):
        runtime().get_runtime(self._env).arm_physics_step()
        super().apply_actions()


@dataclass(kw_only=True)
class R3PositionActionCfg(JointPositionActionCfg):
    residual: bool = False

    def build(self, env):
        return R3PositionAction(self, env)


class ErrorAwareMotionCommand(MotionCommand):
    """mjlab's adaptive sampling, with a share of the starts drawn by tracking error.

    Upstream counts only terminations per 1 s bin, so a segment the policy
    survives by not performing it (a squat done standing tall) keeps only the
    uniform floor. Here every control step adds the worlds' joint-angle RMS
    error to an exponential average per bin, and error_share of the sampling
    mass follows (error / max error) ** error_power (GMT-style), the rest the
    upstream failure statistic.
    """

    def __init__(self, cfg, env):
        self.offset = torch.zeros(env.num_envs, 3, device=env.device)  # terrain: reference shift per world
        self._recenter_ids = None
        self.ground = None  # terrain with spawn_radius: parcour.ground_map of the world's tiles
        self.fixed_spawn = torch.full((env.num_envs, 2), float('nan'), device=env.device)  # evaluation: start xy per world
        self._placed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)  # steps to keep the start height
        if cfg.terrain and cfg.spawn_radius > 0:
            import parcour
            self.ground = parcour.ground_map(env.scene.terrain.cfg.terrain_generator)
            axis = self.ground['axis']
            valid = parcour.spawn_valid(self.ground['top'], self.ground['low'], self.ground['origins'][:, :, None, None, 2])
            valid &= (np.abs(axis)[:, None] <= cfg.spawn_radius + 1e-9) & (np.abs(axis)[None, :] <= cfg.spawn_radius + 1e-9)
            self._ground_top = torch.tensor(self.ground['top'], dtype=torch.float, device=env.device)
            self._ground_valid = torch.tensor(valid, device=env.device)
            self._ground_axis = torch.tensor(axis, dtype=torch.float, device=env.device)
            self._ground_raw = torch.tensor(self.ground['raw'], dtype=torch.float, device=env.device)
            self._spawn_feet = [cfg.body_names.index(n) for n in ('R_foot', 'L_foot')]
            inner = np.nonzero(np.abs(axis) <= cfg.spawn_radius + 1e-9)[0]
            self._spawn_index_range = (int(inner[0]), int(inner[-1]) + 1)
            self._ground_slope = torch.tensor(self.ground['slope'], dtype=torch.float, device=env.device)
        super().__init__(cfg, env)
        if self.ground is not None and cfg.slope_references:
            self._init_slope_references()
        self.bin_error = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.metrics['sampling_error_top1_bin'] = torch.zeros(self.num_envs, device=self.device)

    # Terrain (cfg.terrain, parcour.py): the tape is flat-ground motion from one long path. Each episode is moved so
    # the reference root starts on the world's own tile spawn point (offset xy), and every step the reference rises
    # or falls with the ground under the robot's torso (offset z), so heights, contacts and terminations are read
    # against the local ground. On a plane both offsets stay zero and nothing changes.
    def _recenter(self):
        ids, self._recenter_ids = self._recenter_ids, None
        start = self.motion.body_pos_w[self._t()[ids], 0, :2]
        self.offset[ids] = 0.
        self.offset[ids, :2] = -start
        if self.ground is not None:
            xy, top = self._spawn(ids)
            self.offset[ids, :2] += xy
            self.offset[ids, 2] = top - self._env.scene.env_origins[ids, 2] + .002
            self._placed[ids] = 2  # the ground sensor reads the old spot for one more step after a reset
            self._carry_objects(ids, torch.cat([xy, (top - self._env.scene.env_origins[ids, 2])[:, None]], -1))

    def _carry_objects(self, ids, shift):
        """Move the ball and the door, which their reset events place around the tile centre, by the robot's start
        offset, so they keep their intended distance from the robot (the events run before the command resets)."""
        for name in ('ball', 'door'):
            if name not in self._env.scene.entities:
                continue
            data = self._env.scene.entities[name].data
            env = data._resolve_env_ids(ids)
            if data.indexing.mocap_id is not None:
                data.data.mocap_pos[env, data.indexing.mocap_id] = data.data.mocap_pos[env, data.indexing.mocap_id] + shift.unsqueeze(1)
            else:
                adr = data.indexing.free_joint_q_adr[:3]
                data.data.qpos[env, adr] = data.data.qpos[env, adr] + shift

    def _spawn(self, ids):
        """Start point per world (xy from the tile centre) and the start height: the highest ground under the two
        reference feet, so neither starts inside the ground and the other hovers by the ground step between them.
        Training draws up to 8 points within cfg.spawn_radius and keeps the first whose ground suits (parcour.spawn_valid,
        and at most SPAWN_MAX_STEP between the feet), else the tile centre; evaluation uses its fixed point (fixed_spawn,
        parcour.place_eval_worlds), else the centre."""
        import parcour
        terrain = self._env.scene.terrain
        rows, cols = terrain.terrain_levels[ids], terrain.terrain_types[ids]
        axis, res, n, count = self._ground_axis, self.ground['res'], len(self._ground_axis), len(ids)
        lo, hi = self._spawn_index_range
        cand = torch.randint(lo, hi, (count, 8, 2), device=self.device)
        fixed = self.fixed_spawn[ids]
        cand = torch.where(torch.isnan(fixed)[:, None, :], cand, ((fixed.nan_to_num(0.) - axis[0]) / res).round().long()[:, None, :])
        cand = torch.cat([cand, torch.full((count, 1, 2), n // 2, device=self.device, dtype=cand.dtype)], 1)  # the centre last
        t = self._t()[ids]
        feet = self.motion.body_pos_w[t][:, self._spawn_feet, :2] - self.motion.body_pos_w[t, 0, :2][:, None]  # from the root
        at = axis[cand][:, :, None, :] + feet[:, None]  # [worlds, candidates, feet, xy] from the tile centre
        cell = ((at - axis[0]) / res).round().long().clamp(0, n - 1)
        ground = self._ground_raw[rows[:, None, None], cols[:, None, None], cell[..., 0], cell[..., 1]]
        top, step = ground.amax(-1), ground.amax(-1) - ground.amin(-1)
        ok = self._ground_valid[rows[:, None], cols[:, None], cand[..., 0], cand[..., 1]] & (step < parcour.SPAWN_MAX_STEP)
        ok[:, -1] = True
        choice, row = ok.float().argmax(1), torch.arange(count, device=self.device)
        return axis[cand[row, choice]], top[row, choice]

    # Slope-aware references (cfg.slope_references, terrain with a ground map). The tape is flat-ground motion; on an
    # inclined tile (hill, hollow, rolling) a flat reference foot asks the ankles to stay level, so the leg-joint and
    # foot-orientation terms pay for tilting the whole body with the ground. Here each reference foot turns with the
    # top face it stands over (parcour.ground_map slope; level boxes such as stairs, mats and sills give zero, so a
    # step edge never tilts a foot), the two ankle joints take that turn (projected on their axes at the reference
    # pose), and feet and forearms rise or fall with the slope between them and the torso. Torso, arms and every other
    # joint keep the tape. On level tiles nothing changes.
    SLOPE_RISE_BODIES = ('R_foot', 'L_foot', 'R_forearm', 'L_forearm')
    SLOPE_TILT_BODIES = ('R_foot', 'L_foot')

    def _init_slope_references(self):
        names = list(self.cfg.body_names)
        self._rise_ids = [names.index(n) for n in self.SLOPE_RISE_BODIES if n in names]
        self._tilt_ids = [names.index(n) for n in self.SLOPE_TILT_BODIES]
        joints, model = list(self.robot.joint_names), self._env.sim.mj_model
        ids, axes = [], []
        for foot in ('R', 'L'):
            for kind in ('pitch', 'roll'):
                name = f'{foot}_ankle_{kind}'
                j = next(k for k in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, k).split('/')[-1] == name)
                ids.append(joints.index(name))
                axes.append(model.jnt_axis[j])  # in the joint's body frame; the foot frame up to the ankle roll
        self._ankle_ids = torch.tensor(ids, device=self.device)
        self._ankle_axes = torch.tensor(np.array(axes), dtype=torch.float, device=self.device).view(2, 2, 3)  # [foot, pitch/roll, xyz]

    def _ground_slope_at(self, xy):
        """Rise per run (dz/dx, dz/dy) of the ground face under world points xy [worlds, k, 2], [worlds, k, 2]."""
        terrain = self._env.scene.terrain
        rows, cols = terrain.terrain_levels, terrain.terrain_types
        n, res, first = len(self._ground_axis), self.ground['res'], self._ground_axis[0]
        cell = ((xy - self._env.scene.env_origins[:, None, :2] - first) / res).round().long().clamp(0, n - 1)
        return self._ground_slope[rows[:, None], cols[:, None], cell[..., 0], cell[..., 1]]

    def _slope_adjust(self, t):
        """(rise [worlds, bodies] m, turn quaternion [worlds, bodies, 4], ankle deltas [worlds, joints] rad) at time t."""
        W, B = self.num_envs, len(self.cfg.body_names)
        rise = torch.zeros(W, B, device=self.device)
        turn = torch.zeros(W, B, 4, device=self.device)
        turn[..., 0] = 1.
        delta = torch.zeros(W, self.motion.joint_pos.shape[-1], device=self.device)
        if self.ground is None or not self.cfg.slope_references:
            return rise, turn, delta
        origin = self._env.scene.env_origins[:, None, :2] + self.offset[:, None, :2]
        pos = self.motion.body_pos_w[t][:, self._rise_ids, :2] + origin
        root = self.motion.body_pos_w[t][:, :1, :2] + origin
        g, g0 = self._ground_slope_at(pos), self._ground_slope_at(root)
        rise[:, self._rise_ids] = (.5 * (g + g0) * (pos - root)).sum(-1)  # trapezoid between torso and body
        gt = self._ground_slope_at(self.motion.body_pos_w[t][:, self._tilt_ids, :2] + origin)  # [W, 2 feet, 2]
        # The rotation taking up (0, 0, 1) to the face normal (-gx, -gy, 1): about (gy, -gx, 0) by atan(|g|).
        norm = gt.norm(dim=-1, keepdim=True)
        axis = torch.stack([gt[..., 1], -gt[..., 0], torch.zeros_like(gt[..., 0])], -1) / norm.clamp(min=1e-9)
        angle = torch.atan(norm[..., 0])
        omega = axis * angle[..., None]  # world rotation vector per foot
        turn[:, self._tilt_ids] = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3)).view(W, -1, 4)
        quat = self.motion.body_quat_w[t][:, self._tilt_ids]  # [W, 2, 4]
        world_axes = quat_apply(quat[:, :, None].expand(-1, -1, 2, -1).reshape(-1, 4),
            self._ankle_axes[None].expand(W, -1, -1, -1).reshape(-1, 3)).view(W, 2, 2, 3)
        delta[:, self._ankle_ids] = (world_axes * omega[:, :, None]).sum(-1).reshape(W, 4)
        return rise, turn, delta

    def _resample_command(self, env_ids):
        if self.cfg.terrain:
            self._recenter_ids = env_ids  # applied when the parent first reads the reference for placement
        super()._resample_command(env_ids)

    def robot_ground(self, names):
        """Ground height (m, world z) under the named robot bodies (GROUND_FRAMES), [worlds, len(names)]."""
        if not self.cfg.terrain:
            return self._env.scene.env_origins[:, 2:3].expand(-1, len(names))
        data = self._env.scene[GROUND_SENSOR].data
        ids = [GROUND_FRAMES.index(n) for n in names]
        return data.frame_pos_w[:, ids, 2] - data.heights[:, ids]

    def reference_ground(self, bodies=None):
        """Ground height (m, world z) the reference stands on, [worlds]; with `bodies` (indices into body_names) the
        ground under each of them, [worlds, len(bodies)] (the torso's, plus the slope rise of slope references)."""
        base = self._env.scene.env_origins[:, 2] + self.offset[:, 2]
        if bodies is None:
            return base
        return base[:, None] + self._slope_adjust(self._t())[0][:, bodies]

    # Reference lead (cfg.lead control frames). mjlab computes rewards after the physics step and before the command
    # advances, so they compare the robot at t + dt with the reference at t and pay for being one frame (20 ms)
    # late; every trained policy settled at that lag, also on a plant with half the delay. With lead=1 the actor's
    # command and every reward term use the reference for the time the robot state represents. Evaluation logs the
    # reference at the true time (reference_now).
    def _t(self):
        return (self.time_steps + self.cfg.lead).clamp(max=self.motion.time_step_total - 1)

    def reference_now(self, field, bodies=None):
        """Reference `field` of the motion file at the current time step, without the lead (for evaluation)."""
        x = getattr(self.motion, field)[self.time_steps]
        if field == 'body_pos_w':
            x = x + self.offset[:, None, :]
            x[..., 2] += self._slope_adjust(self.time_steps)[0]
        elif field == 'body_quat_w':
            x = quat_mul(self._slope_adjust(self.time_steps)[1], x)
        elif field == 'joint_pos':
            x = x + self._slope_adjust(self.time_steps)[2]
        return x if bodies is None else x[:, bodies]

    @property
    def joint_pos(self):
        t = self._t()
        return self.motion.joint_pos[t] + self._slope_adjust(t)[2] if self.cfg.slope_references else self.motion.joint_pos[t]

    @property
    def joint_vel(self):
        return self.motion.joint_vel[self._t()]

    @property
    def body_pos_w(self):
        if self._recenter_ids is not None:
            self._recenter()
        t = self._t()
        x = self.motion.body_pos_w[t] + self._env.scene.env_origins[:, None, :] + self.offset[:, None, :]
        if self.cfg.slope_references:
            x[..., 2] += self._slope_adjust(t)[0]
        return x

    @property
    def body_quat_w(self):
        t = self._t()
        return quat_mul(self._slope_adjust(t)[1], self.motion.body_quat_w[t]) if self.cfg.slope_references else self.motion.body_quat_w[t]

    @property
    def body_lin_vel_w(self):
        return self.motion.body_lin_vel_w[self._t()]

    @property
    def body_ang_vel_w(self):
        return self.motion.body_ang_vel_w[self._t()]

    @property
    def anchor_pos_w(self):
        if self._recenter_ids is not None:
            self._recenter()
        return self.motion.body_pos_w[self._t(), self.motion_anchor_body_index] + self._env.scene.env_origins + self.offset

    @property
    def anchor_quat_w(self):
        return self.motion.body_quat_w[self._t(), self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self.motion.body_lin_vel_w[self._t(), self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self.motion.body_ang_vel_w[self._t(), self.motion_anchor_body_index]

    def _bins(self):
        return torch.clamp((self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1)

    def _update_command(self):
        error = (self.joint_pos - self.robot_joint_pos).square().mean(-1).sqrt()
        bins = self._bins()
        total = torch.bincount(bins, weights=error, minlength=self.bin_count)
        count = torch.bincount(bins, minlength=self.bin_count).float()
        seen = count > 0
        mean = torch.where(seen, total / count.clamp(min=1), self.bin_error)
        self.bin_error = torch.where(seen, self.cfg.error_alpha * mean + (1 - self.cfg.error_alpha) * self.bin_error, self.bin_error)
        if self.cfg.terrain:
            sensed = self.robot_ground(('torso',))[:, 0] - self._env.scene.env_origins[:, 2]
            self.offset[:, 2] = torch.where(self._placed > 0, self.offset[:, 2], sensed)
            self._placed = (self._placed - 1).clamp(min=0)
        super()._update_command()

    def _adaptive_sampling(self, env_ids):
        saved = self.bin_failed_count
        failure = saved + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        failure = failure / failure.sum()
        score = (self.bin_error / self.bin_error.max().clamp(min=1e-6)) ** self.cfg.error_power
        score = score / score.sum() if score.sum() > 0 else failure
        mixed = (1 - self.cfg.error_share) * failure + self.cfg.error_share * score
        # Upstream adds the uniform floor and normalises again; hand it the mixed distribution with no floor.
        ratio, self.cfg.adaptive_uniform_ratio = self.cfg.adaptive_uniform_ratio, 0.
        self.bin_failed_count = mixed
        try:
            super()._adaptive_sampling(env_ids)
        finally:
            self.bin_failed_count, self.cfg.adaptive_uniform_ratio = saved, ratio
        self.metrics['sampling_error_top1_bin'][:] = self.bin_error.argmax().float() / self.bin_count


@dataclass(kw_only=True)
class ErrorAwareMotionCommandCfg(MotionCommandCfg):
    error_share: float = .5
    error_power: float = 2.
    error_alpha: float = .01
    lead: int = 0  # reference lead in control frames (ErrorAwareMotionCommand._t)
    terrain: bool = False  # reference re-centred per episode and following the ground (parcour.py)
    spawn_radius: float = 0.  # terrain: start each episode this far (m, at most) from the tile centre, on the ground
    slope_references: bool = False  # terrain: feet, ankles and forearms follow inclined ground (ErrorAwareMotionCommand)

    def build(self, env):
        return ErrorAwareMotionCommand(self, env)


def native_options(cfg, physics_dt=None):
    """Copy the selected MJCF's solver options and floor contact into the scene (physics_dt: a coarser r4 step)."""
    model = mujoco.MjModel.from_xml_path(str(MODEL / 'cove50-rigid.xml'))
    opt, floor = model.opt, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    cfg.scene.terrain.collisions = (CollisionCfg(geom_names_expr=(r'^terrain(_\d+)?$',),  # plane, or generator tiles terrain_N
        friction=tuple(model.geom_friction[floor]), solref=tuple(model.geom_solref[floor]),
        solimp=tuple(model.geom_solimp[floor]), contype=int(model.geom_contype[floor]),
        conaffinity=int(model.geom_conaffinity[floor]), condim=int(model.geom_condim[floor]),
        priority=int(model.geom_priority[floor]), margin=float(model.geom_margin[floor]),
        gap=float(model.geom_gap[floor]), solmix=float(model.geom_solmix[floor])),)
    m = cfg.sim.mujoco
    m.timestep = physics_dt or runtime().BatchedRuntime.physics_dt_s
    if physics_dt is None and abs(opt.timestep - m.timestep) > 1e-12:
        raise ValueError('Selected MJCF and R3 boundary timesteps differ')
    m.integrator, m.cone = 'implicitfast', 'elliptic'
    if opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST or opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError('Selected MJCF integrator/cone changed')
    m.iterations, m.ls_iterations = int(opt.iterations), int(opt.ls_iterations)
    m.tolerance, m.ls_tolerance, m.impratio = float(opt.tolerance), float(opt.ls_tolerance), float(opt.impratio)
    m.gravity = tuple(opt.gravity)
    cfg.sim.nconmax, cfg.sim.njmax = 512, 2048  # walk-resume R3 task values
    cfg.decimation = round(runtime().BatchedRuntime.policy_dt_s / m.timestep)  # 80 x .25 ms = 50 Hz


def env_cfg(motion_file, *, play=False, num_envs=1, reward_widths='g1', peak=False, actuator='c047', dynamics='r3',
            levels=None, stop_weight=None, joint_weight=None, joint_std=.5, knee_weight=None, knee_std=.1, reward_set='v1', command_delay=None, feedback_delay=None, feet_scale=None, arm_scale=None, fist=None, preview_steps=None, reference_lead=None, residual_action=False,
            anchor_tolerance=None, error_sampling=None, physics_dt=None, terrain=None, slope_references=True):
    """stop_weight: weight of the target_beyond_stop penalty; joint_weight: weight of joint_position_tracking
    (std joint_std rad); knee_weight: weight of knee_tracking (std knee_std rad). None leaves the upstream reward set.
    reward_set: 'v1' (fixed widths), 'v2' (REWARD_SET_V2: adaptive widths, joint-velocity and foot tracking) or 'v3'
    (REWARD_SET_V3: per-group per-joint adaptive terms, speed-weighted leg velocity, foot contact and swing height) or
    'v4' (REWARD_SET_V4: v3 plus torso height and vertical speed) or 'v5' (REWARD_SET_V5: v4 with an adaptive global
    root position); a 'k' suffix adds knuckle contact (HAND_CONTACT_TERM). residual_action: actions are offsets on the
    reference joint angles (R3PositionAction). anchor_tolerance: torso-height termination in m instead of
    TERMINATION_HEIGHT_M (end effectors keep it). arm_scale: upper-arm and forearm length scale (scale_arms). fist: (length m, mass kg) knuckle fists (add_fists).
    terrain: 'parcour' puts each world on a random tile of the obstacle world (parcour.py) with ground-relative reference
    heights and contacts (ErrorAwareMotionCommand terrain mode); None keeps the plane. slope_references: on terrain,
    reference feet, ankles and forearms follow inclined ground (ErrorAwareMotionCommand._slope_adjust; False gives the
    flat tape as before 2026-11-12).
    reference_lead:
    control frames the command and rewards lead the reference by (ErrorAwareMotionCommand). preview_steps: control frames ahead of the reference the actor and
    critic also see (reference_preview). error_sampling: share of adaptive starts drawn by per-bin
    joint-angle error (ErrorAwareMotionCommand). physics_dt: r4 physics step in s (None keeps R3's .25 ms)."""
    cfg = make_tracking_env_cfg()
    cfg.scene.num_envs = num_envs
    for name, std in REWARD_STD[reward_widths].items():
        if cfg.rewards[name].params['std'] != _G1_STD[name]:
            raise ValueError(f'Upstream {name} width changed')
        cfg.rewards[name].params['std'] = std
    if stop_weight:
        cfg.rewards['target_beyond_stop'] = RewardTermCfg(func=target_beyond_stop, weight=-abs(stop_weight))
    if joint_weight:
        cfg.rewards['joint_position_tracking'] = RewardTermCfg(func=joint_position_tracking, weight=joint_weight,
            params=dict(std=joint_std))
    if knee_weight:
        cfg.rewards['knee_tracking'] = RewardTermCfg(func=knee_tracking, weight=knee_weight, params=dict(std=knee_std))
    if reward_set == 'v2':
        for name, (kind, init_std, min_std, weight) in REWARD_SET_V2.items():
            cfg.rewards[name] = RewardTermCfg(func=AdaptiveTrackingExp, weight=weight,
                params=dict(kind=kind, init_std=init_std, min_std=min_std))
    elif reward_set.rstrip('k') in REWARD_SETS:
        cfg.rewards.pop('joint_position_tracking', None)  # replaced by the per-group, per-joint terms
        terms = dict(REWARD_SETS[reward_set.rstrip('k')], **(HAND_CONTACT_TERM if reward_set.endswith('k') else {}))
        for name, (cls, params, weight) in terms.items():
            cfg.rewards[name] = RewardTermCfg(func=cls, weight=weight, params=dict(params))
    elif reward_set != 'v1':
        raise ValueError(f'unknown reward set {reward_set}')
    # peak: the servo's stall-torque curve instead of its rated-torque cap.
    servo = ACTUATORS[actuator]
    if dynamics == 'r4':
        resolved = runtime_dynamics().resolve(actuator, levels)
        delays = {k: v for k, v in (('command_delay_s', command_delay), ('feedback_delay_s', feedback_delay)) if v is not None}
        robot = runtime().robot_cfg(peak=peak, dynamics=resolved, physics_dt_s=physics_dt, **delays)
    elif servo and servo.get('r4_only'):
        raise ValueError(f'{actuator} is defined for r4 dynamics only')
    else:
        robot = runtime().robot_cfg(peak=peak, c047=servo and servo['law'])
    if feet_scale:
        robot.spec_fn = partial(_scaled_feet_spec, robot.spec_fn, tuple(feet_scale))
    if arm_scale:
        robot.spec_fn = partial(_scaled_arms_spec, robot.spec_fn, arm_scale if np.isscalar(arm_scale) else tuple(arm_scale))
    if fist:
        robot.spec_fn = partial(_fisted_spec, robot.spec_fn, tuple(fist))
    if servo:
        c047_kg = json.loads((MODEL / 'actuators.json').read_text())['models']['c047']['mass_kg']
        extra = ({j: v['mass_kg'] - c047_kg for j, v in resolved['joints'].items()} if servo.get('mixed')
            else servo['mass_kg'] - c047_kg)
        robot.spec_fn = partial(heavier_servos, robot.spec_fn, extra)
    cfg.scene.entities = {'robot': robot}
    # R3 straight knees/elbows sit exactly on their hard 0 rad stop. The
    # adapter's .9 soft factor would put every reference ~.11 rad inside the
    # joint_limit penalty and bend straight joints when resets clip to it.
    cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor = SOFT_LIMIT_FACTOR
    cfg.scene.sensors = (ContactSensorCfg(name='self_collision',
        primary=ContactMatch(mode='subtree', pattern='torso', entity='robot'),
        secondary=ContactMatch(mode='subtree', pattern='torso', entity='robot'),
        fields=('found', 'force'), reduce='none', num_slots=1, history_length=4),)
    if physics_dt and dynamics != 'r4':
        raise ValueError('a coarser physics step is defined for r4 dynamics only')
    native_options(cfg, physics_dt)
    if terrain == 'parcour':
        import parcour
        cfg.scene.terrain.terrain_type, cfg.scene.terrain.terrain_generator = 'generator', parcour.PARCOUR_TERRAIN
        cfg.scene.terrain.max_init_terrain_level = None
        # A low floor friction lets the sole's per-world value decide (MuJoCo takes the larger); per-tile solref (a soft
        # cushion) is kept.
        cfg.scene.terrain.collisions = tuple(replace(c, friction=(parcour.FLOOR_FRICTION_BASE, .005, .0001), solref=None, solimp=None)
            for c in cfg.scene.terrain.collisions)
        cfg.scene.sensors += (TerrainHeightSensorCfg(name=GROUND_SENSOR,
            frame=tuple(ObjRef(type='body', name=n, entity='robot') for n in GROUND_FRAMES),
            pattern=RingPatternCfg.single_ring(radius=.03, num_samples=4), ray_alignment='yaw', max_distance=1.,
            exclude_parent_body=True, include_geom_groups=(0,)),)
    elif terrain:
        raise ValueError(f'unknown terrain {terrain}')

    cfg.actions = {'joint_pos': R3PositionActionCfg(entity_name='robot', actuator_names=('.*',),
        scale=ACTION_SCALE, use_default_offset=True, residual=residual_action)}

    motion = cfg.commands['motion']
    assert isinstance(motion, MotionCommandCfg)
    if error_sampling or reference_lead or terrain:
        motion = ErrorAwareMotionCommandCfg(**{f.name: getattr(motion, f.name) for f in fields(motion)}, error_share=error_sampling or 0.,
            lead=reference_lead or 0, terrain=bool(terrain), spawn_radius=parcour.SPAWN_RADIUS if terrain else 0.,
            slope_references=bool(terrain and slope_references))
        cfg.commands['motion'] = motion
    motion.motion_file = str(motion_file)
    motion.anchor_body_name = 'torso'
    motion.body_names = TRACKED_BODIES
    motion.pose_range = RANGES['reset_pose']['value']
    motion.velocity_range = RANGES['reset_velocity']['value']
    motion.joint_position_range = RANGES['reset_joint_rad']['value']

    # Deployable actor: no root position/linear velocity (no state estimation,
    # as the native G1 No-State-Estimation variant) and R3-measured joints/IMU.
    # motion_anchor_ori_b still reads true torso orientation; replacing it with
    # the gravity estimate is later student work.
    actor = cfg.observations['actor'].terms
    for name in ('motion_anchor_pos_b', 'base_lin_vel', 'base_ang_vel', 'joint_pos', 'joint_vel'):
        actor.pop(name)
    actor['base_ang_vel'] = ObservationTermCfg(func=measured, params=dict(start=44, stop=47))
    actor['projected_gravity'] = ObservationTermCfg(func=measured, params=dict(start=47, stop=50))
    actor['joint_pos'] = ObservationTermCfg(func=measured, params=dict(start=0, stop=22))
    noise = RANGES['joint_vel_noise_rad_s']['value']
    actor['joint_vel'] = ObservationTermCfg(func=measured, params=dict(start=22, stop=44), noise=Unoise(n_min=noise[0], n_max=noise[1]))
    actor['actions'] = actor.pop('actions')
    if preview_steps:  # appended last, so a retained teacher widens by zero columns (stage.widen_checkpoint)
        actor['reference_preview'] = ObservationTermCfg(func=reference_preview, params=dict(steps=tuple(preview_steps)))
    cfg.observations['actor'] = ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play)
    # The privileged critic keeps true root velocity; R3 has no velocimeter
    # sensor, so read it from the entity instead of G1's imu_lin/ang_vel.
    critic = cfg.observations['critic'].terms
    critic['base_lin_vel'] = ObservationTermCfg(func=mdp.base_lin_vel)
    critic['base_ang_vel'] = ObservationTermCfg(func=mdp.base_ang_vel)
    if preview_steps:
        critic['reference_preview'] = ObservationTermCfg(func=reference_preview, params=dict(steps=tuple(preview_steps)))

    events = cfg.events
    events.pop('encoder_bias')
    events['push_robot'].params['velocity_range'] = RANGES['push_velocity']['value']
    events['base_com'].params['asset_cfg'].body_names = ('torso',)
    lo, hi = RANGES['torso_com_offset_m']['value']
    events['base_com'].params['ranges'] = {0: (lo, hi), 1: (lo, hi), 2: (lo, hi)}
    events['foot_friction'].params['asset_cfg'].geom_names = r'^collision_(R|L)_foot_\d+$'
    events['foot_friction'].params['ranges'] = RANGES['foot_friction']['value']

    cfg.terminations['anchor_pos'].params['threshold'] = anchor_tolerance or TERMINATION_HEIGHT_M
    cfg.terminations['ee_body_pos'].params['threshold'] = TERMINATION_HEIGHT_M
    cfg.terminations['ee_body_pos'].params['body_names'] = END_EFFECTORS
    cfg.viewer.body_name = 'torso'
    cfg.viewer.distance = 1.2

    if terrain == 'parcour':
        import parcour
        parcour.add_world_objects(cfg)  # ball, door, carried toys, sole friction from smooth wood to carpet
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.events.pop('push_robot')
        motion.pose_range, motion.velocity_range = {}, {}
        motion.sampling_mode = 'start'
    return cfg


def ppo_cfg(**overrides):
    """Native G1 tracking PPO settings; only the experiment name differs."""
    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=.2,
            entropy_coef=.005, num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3,
            schedule='adaptive', gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.0),
        experiment_name='ape0_tracking', save_interval=500, num_steps_per_env=24, max_iterations=30_000)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
