"""Cove-50 R3 whole-body inverse dynamics with optimised floor forces, and the servo envelopes it is compared with.

Each frame's joint torques come from mj_inverse on the selected R3 body with
constraints off, plus floor forces at the active sole corners. Those forces sit
inside a friction pyramid (MU) and are chosen by non-negative least squares:
first the floating base must be carried by them, then among the ways to do that
the one with the least rated-normalised joint torque is taken. Whatever the
floor cannot supply is left as a residual force and moment on the torso, which
marks a reference the body cannot perform as given (or an outside push). The
demand is a lower bound for a perfect controller on this rigid model.

Actuator values (catalog torque-speed line, mass, rotor inertia, Coulomb
friction) come from the r4 best-estimate sidecar,
robot/model/dynamics/actuator-dynamics.json, with its
sources and low/high bounds. Rotor inertia 'none' reproduces the R3 model.
"""
from functools import partial
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tracking  # noqa: E402

tracking.runtime()
from cove50_r3_runtime import mjlab_adapter  # noqa: E402

CONTRACT = json.loads((tracking.MODEL / 'actuators.json').read_text())
C047 = CONTRACT['models']['c047']
JOINTS = [a['joint'] for a in CONTRACT['actuators']]
# The 20 C047 joints the actuator question is about; the two XM335 neck axes stay as they are.
SERVO_JOINTS = [a['joint'] for a in CONTRACT['actuators'] if a['model'] == 'c047']
LEGS = [j for j in SERVO_JOINTS if any(k in j for k in ('hip', 'knee', 'ankle'))]
ARMS = [j for j in SERVO_JOINTS if j not in LEGS]
SIDECAR = json.loads((tracking.ROOT / 'robot/model/dynamics/actuator-dynamics.json').read_text())
BODIES = ('c047', 'sts3250', 'st3025')
MU = .7  # low end of the narrowed 0.7-0.9 foot friction range
G = 9.81


def armature(body, level):
    """Rotor inertia at the output for level none (R3), low, nominal or high."""
    return 0. if level == 'none' else SIDECAR['models'][body]['armature_kg_m2'][level]


def friction(body):
    return SIDECAR['models'][body]['frictionloss_Nm']['nominal']


def _servos():
    out = {}
    for b in [k for k in SIDECAR['models'] if k != 'xm335']:  # every body-joint servo in the sidecar
        m = SIDECAR['models'][b]
        base = dict(body=b, stall=m['stall_torque_Nm'], speed=m['no_load_speed_rad_s'], rated=m['rated_torque_Nm'],
            idle=m['idle_current_A'], stall_current=m['stall_current_A'])
        out[f'{b}-rated'] = dict(base, cap=base['rated'])
        out[f'{b}-stall'] = dict(base, cap=base['stall'])
        # Catalog sensitivity: a servo delivering 20% less stall torque, or 20% less no-load speed, than listed.
        out[f'{b}-stall-20'] = dict(base, stall=.8 * base['stall'], cap=.8 * base['stall'])
        out[f'{b}-speed-20'] = dict(base, speed=.8 * base['speed'], cap=base['stall'])
    return out


SERVOS = _servos()


FEET_SCALE = tuple(float(x) for x in os.environ['COVE50_FEET_SCALE'].split(',')) if os.environ.get('COVE50_FEET_SCALE') else None
ARM_SCALE = (tuple(float(x) for x in os.environ['COVE50_ARM_SCALE'].split(',')) if ',' in os.environ['COVE50_ARM_SCALE']
    else float(os.environ['COVE50_ARM_SCALE'])) if os.environ.get('COVE50_ARM_SCALE') else None  # s, or upper,forearm
FIST = tuple(float(x) for x in os.environ['COVE50_FIST'].split(',')) if os.environ.get('COVE50_FIST') else None  # length,mass


def robot_spec():
    """The selected R3 spec, with COVE50_FEET_SCALE=length,width (tracking.scale_feet), COVE50_ARM_SCALE=s or
    upper,forearm (tracking.scale_arms) and COVE50_FIST=length,mass (tracking.add_fists) when set."""
    spec = tracking.scale_arms(tracking.scale_feet(mjlab_adapter.robot_spec(), FEET_SCALE), ARM_SCALE)
    return tracking.add_fists(spec, *FIST) if FIST else spec


def model(body='c047', rotor=0., constraints=False):
    """The selected R3 body (no floor, no actuators) with `body` servos' mass on the 20 C047 joints, rotor
    inertia `rotor` (kg m2) on them and the neck's nominal r4 rotor inertia when rotor > 0."""
    spec_fn = robot_spec
    if body != 'c047':
        spec_fn = partial(tracking.heavier_servos, spec_fn, SIDECAR['models'][body]['mass_kg'] - C047['mass_kg'])
    m = spec_fn().compile()
    for name in SERVO_JOINTS:
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]] = rotor
    neck = SIDECAR['models']['xm335']['armature_kg_m2']['nominal'] if rotor else 0.
    for name in set(JOINTS) - set(SERVO_JOINTS):
        m.dof_armature[m.jnt_dofadr[m.joint(name).id]] = neck
    if not constraints:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m


def qadr(m, names):
    return np.array([m.jnt_qposadr[m.joint(n).id] for n in names])


def dadr(m, names):
    return np.array([m.jnt_dofadr[m.joint(n).id] for n in names])


def neutral(m):
    d = mujoco.MjData(m)
    d.qpos[:3] = 0, 0, .30201
    d.qpos[3] = 1
    mujoco.mj_kinematics(m, d)
    return d


def sole_corners(m):
    """Four sole corners per foot in the foot body frame, from its collision mesh at the neutral pose."""
    d = neutral(m)
    corners = {}
    for side in 'RL':
        foot = m.body(f'{side}_foot').id
        pts = []
        for g in range(m.ngeom):
            if (m.geom(g).name or '').startswith(f'collision_{side}_foot') and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
                pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        pts = np.concatenate(pts)
        sole = pts[pts[:, 2] < pts[:, 2].min() + .002]
        world = np.array([sole[np.argmax(sole[:, :2] @ s)] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
        world[:, 2] = pts[:, 2].min()
        corners[f'{side}_foot'] = (world - d.xpos[foot]) @ d.xmat[foot].reshape(3, 3)
    return corners


def knuckle_points(m):
    """Four contact points per hand in the forearm body frame: the corners of the forearm collision mesh's distal
    end (its lowest 3 mm with the arm hanging at the neutral pose), where a knuckle pad would touch the floor."""
    d = neutral(m)
    points = {}
    for side in 'RL':
        arm = m.body(f'{side}_forearm').id
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == arm and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
                pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        pts = np.concatenate(pts)
        tip = pts[pts[:, 2] < pts[:, 2].min() + .003]
        world = np.array([tip[np.argmax(tip[:, :2] @ s)] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))])
        world[:, 2] = pts[:, 2].min()
        points[f'{side}_forearm'] = (world - d.xpos[arm]) @ d.xmat[arm].reshape(3, 3)
    return points


def active_contacts(dyn, mo, qpos, t):
    """{body: bool[4]} support points on the floor at frame t of motion arrays mo: the feet from its stored sole
    corners (or from heights when it has none), and the knuckles from its hand_contacts [T, 2] when present."""
    if 'corners' in mo.files:
        active = dict(zip(('R_foot', 'L_foot'), mo['corners'][t]))
    else:
        active = {f: v for f, v in contacts_from_height(dyn, qpos).items() if f.endswith('_foot')}
    if 'hand_contacts' in mo.files:
        for k, side in enumerate('RL'):
            active[f'{side}_forearm'] = np.full(4, bool(mo['hand_contacts'][t, k]))
    return active


EDGES = np.array([(MU, 0, 1), (-MU, 0, 1), (0, MU, 1), (0, -MU, 1)], float)


class Dynamics:
    """Inverse dynamics with floor forces for one body condition."""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.corners = dict(sole_corners(m), **knuckle_points(m))  # knuckles only bear load in hand-contact motions
        self.feet = {f: m.body(f).id for f in self.corners}
        mass = m.body_subtreemass[1]
        self.weight = mass * G
        # Base residual weighted 30x a rated servo; moments over a 0.1 m lever.
        self.wb = np.r_[np.full(3, 30 / self.weight), np.full(3, 30 / (self.weight * .1))]
        rated = {a['joint']: (C047['rated_torque_Nm'] if a['model'] == 'c047' else .2) for a in CONTRACT['actuators']}
        self.dofs = dadr(m, JOINTS)
        self.wj = np.array([1 / rated[j] for j in JOINTS])

    def corner_world(self, foot):
        b = self.feet[foot]
        return self.d.xpos[b] + self.corners[foot] @ self.d.xmat[b].reshape(3, 3).T

    def frame(self, qpos, qvel, qacc, active):
        """active: {foot: bool[4]} corners on the floor. Returns joint torques (JOINTS order), base residual (6), foot normal forces."""
        m, d = self.m, self.d
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos, qvel, qacc
        mujoco.mj_inverse(m, d)
        F = d.qfrc_inverse.copy()
        cols, owner = [], []
        jac = np.zeros((3, m.nv))
        for foot, on in active.items():
            b = self.feet[foot]
            for k, p in enumerate(self.corner_world(foot)):
                if on[k]:
                    mujoco.mj_jac(m, d, jac, None, p, b)
                    cols.append(jac.T @ EDGES.T)  # nv x 4 pyramid edges at this corner
                    owner.append(foot)
        normals = {f: 0. for f in self.feet}
        gen = np.zeros(m.nv)
        if owner:
            A = np.concatenate(cols, axis=1)
            K = A.shape[1]
            lhs = np.vstack([self.wb[:, None] * A[:6], self.wj[:, None] * A[self.dofs], 1e-4 * np.eye(K)])
            rhs = np.r_[self.wb * F[:6], self.wj * F[self.dofs], np.zeros(K)]
            lam, _ = nnls(lhs, rhs, maxiter=50 * K)
            gen = A @ lam
            for foot, f in zip(owner, lam.reshape(-1, 4).sum(1)):
                normals[foot] += float(f)  # EDGES all have unit normal component
        return F[self.dofs] - gen[self.dofs], F[:6] - gen[:6], normals


def contacts_from_height(dyn, qpos, threshold=.004):
    """Corners within threshold of the floor (z=0) count as on it."""
    d = dyn.d
    d.qpos[:] = qpos
    mujoco.mj_kinematics(dyn.m, d)
    return {f: dyn.corner_world(f)[:, 2] < threshold for f in dyn.feet}


def envelope(servo, tau, omega):
    """Fraction of the servo's available torque in the demanded direction at that speed (>1: beyond it), and current."""
    s = SERVOS[servo]
    upper = np.minimum(s['cap'], np.clip(s['stall'] * (1 - omega / s['speed']), 0, None))
    lower = np.minimum(s['cap'], np.clip(s['stall'] * (1 + omega / s['speed']), 0, None))
    bound = np.where(tau >= 0, upper, lower)
    use = np.minimum(np.abs(tau) / np.maximum(bound, 1e-6), 99.)
    current = s['idle'] + (s['stall_current'] - s['idle']) * np.abs(tau) / s['stall']
    return use, current
