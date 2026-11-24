"""The APE0 agility motion set: fixed references across tempos for the servo-demand check and later WBC training.

Families (each at two to four tempos):
- retained: the six published R3 skill rollouts (physics rollouts on this body), the
  learned forward walk replayed 1.25x and 1.5x (same stride, quicker cadence), and
  the four without stepping replayed faster (time-compressed; feet stay planted);
- authored ape moves with planted feet, legs solved by inverse kinematics on a
  designed torso path: chest beat, overhead wave, arm display swing, knee bounce,
  side sway, quick crouch, and repeated hops with ballistic flight;
- stepping from a linear-inverted-pendulum ZMP preview controller (Kajita 2003) with
  leg IK and a heel lift before each swing: turn in place and a stomping march.
  Walking speed comes from the learned R3 walk replayed faster: a flat-footed
  generator on these 0.2 m legs with +-35 deg ankles could not make 7 cm steps at
  0.2 m/s and above, and its 0.1 m/s walk needed more joint speed than the learned
  0.2 m/s walk (its swing design, not the body).
Authored and generated references are kinematic designs, not physics rollouts:
the demand check's base residual shows how far each is from what the floor can
carry. Joint angles are the R3 convention: shoulder pitch + raises the arm
forward, roll + abducts, yaw - turns the forearm inward, elbow + flexes; hip
pitch - flexes, knee + flexes, ankle pitch - dorsiflexes.

    .venv-control/bin/python control/wbc/agility/library.py --output DIR [--only NAME ...]

writes NAME.npz (qpos, qvel, qacc, contact corners, fps) and motions.json with
hashes; cheap and exact to regenerate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import least_squares

import body

FPS = 200
DT = 1 / FPS
STAND_Z = .296  # torso height with knees about 34 deg bent (straight-leg neutral is .302)
# Gaits stand 8 mm lower and keep the ZMP 20 mm inside the sole centre: at
# STAND_Z the lateral COM sway over a stance foot over-reaches the other leg.
GAIT_Z, ZMP_INSET = .296, .025
MAX_STEP_TIME = .45  # quicker steps keep the lateral COM sway (and the far leg's reach) small
# Gait feet sit this much inside their neutral, under the hips as in the learned walk.
GAIT_NARROW = .015
TOE_OFF = np.radians(12.)  # heel lift of the trailing foot before it swings
GAIT_LEAN = 0.  # forward torso lean while stepping (rad); lean tilts a flat swinging foot further against the ankle stop
# Torso roll per metre of lateral COM offset from the feet midpoint. The learned
# R3 forward walk rolls about +-8 deg over the stance foot; without it the far leg over-reaches.
GAIT_ROLL = 0.
ARM_REST = (5., 8., 0., 15.)  # pitch, roll, yaw, elbow (deg)
# Forward torso lean per metre of crouch, capped at 30 deg: an upright torso
# runs the ankle into its -35 deg stop within about 25 mm of crouch.
LEAN_PER_M, LEAN_MAX = np.radians(1.) / .001, np.radians(30.)
BLEND_SPEED = 2.  # rad/s peak joint speed while blending into and out of a move
IK_ACCEPT = 5e-4  # m (and rad) a foot target may be missed where joint stops bind
ROT_WEIGHT = .1  # IK residual metres per radian of foot rotation
SWING_ROT_WEIGHT = .003  # a swinging foot tilts 0.1 rad before its position gives 0.3 mm
SWING_SMOOTH = .002  # and its leg stays near the previous frame's pose (m per rad), so the fit cannot wander
SIDES = ('R', 'L')
FEET = ('R_foot', 'L_foot')
LEG = ('hip_yaw', 'hip_pitch', 'hip_roll', 'knee', 'ankle_pitch', 'ankle_roll')
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')
RETAINED = dict(**{'forward-walk': (1, 1.25, 1.5), 'walk-stop': (1,)},
    **{n: (1, 2, 4) for n in ('deep-squat', 'deep-rise', 'bilateral-arms', 'alternating-arms')})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def minjerk(s):
    s = np.clip(s, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def quintic(p0, v0, a0, p1, v1, a1, T):
    """Coefficients c (c[0] + c[1] t + ...) meeting position/velocity/acceleration at 0 and T."""
    M = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5], [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4], [0, 0, 2, 6*T, 12*T**2, 20*T**3]])
    return np.linalg.solve(M, [p0, v0, a0, p1, v1, a1])


def yaw_mat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def quat(yaw=0., pitch=0., roll=0.):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [roll, pitch, yaw], 'xyz')  # extrinsic x, y, z: R = Rz Ry Rx
    return q


class Body:
    """Kinematics helpers on the c047 R3 body: neutral foot frames, leg IK, whole-body COM."""

    def __init__(self):
        self.m = body.model('c047', 0.)
        self.d = mujoco.MjData(self.m)
        n = body.neutral(self.m)
        self.foot0 = {f: (n.xpos[self.m.body(f).id].copy(), n.xmat[self.m.body(f).id].reshape(3, 3).copy()) for f in FEET}
        corners = body.sole_corners(self.m)
        self.sole0 = {f: self.foot0[f][0] + corners[f].mean(0) @ self.foot0[f][1].T for f in FEET}
        self.leg_q = {f: body.qadr(self.m, [f'{f[0]}_{j}' for j in LEG]) for f in FEET}
        self.leg_d = {f: body.dadr(self.m, [f'{f[0]}_{j}' for j in LEG]) for f in FEET}
        self.arm_q = {s: body.qadr(self.m, [f'{s}_{j}' for j in ARM]) for s in SIDES}
        self.knuckle = {f: pts.mean(0) for f, pts in body.knuckle_points(self.m).items()}
        jr = self.m.jnt_range
        self.lo = np.full(self.m.nq, -np.inf)
        self.hi = np.full(self.m.nq, np.inf)
        for j in range(1, self.m.njnt):
            if self.m.jnt_limited[j]:
                self.lo[self.m.jnt_qposadr[j]], self.hi[self.m.jnt_qposadr[j]] = jr[j]
        # The R3 hips are unlimited in the model; bound the IK to a sane envelope so it cannot flip.
        for s in SIDES:
            for j, (lo, hi) in dict(hip_yaw=(-35, 35), hip_pitch=(-110, 45), hip_roll=(-40, 40)).items():
                a = self.m.jnt_qposadr[self.m.joint(f'{s}_{j}').id]
                self.lo[a], self.hi[a] = np.radians(lo), np.radians(hi)
        self.mass = self.m.body_subtreemass[1]
        self.ik_miss = self.swing_tilt = 0.
        guess = np.zeros(self.m.nq)
        guess[:3], guess[3:7] = (0, 0, STAND_Z), quat()
        for f in FEET:
            guess[self.leg_q[f][[1, 3, 4]]] = -.25, .5, -.25  # bent-knee start away from the straight-knee singularity
        self.q_stand = self.legs(guess, {f: self.foot0[f] for f in FEET})

    def stand(self, z=STAND_Z, arms=(ARM_REST, ARM_REST)):
        q = self.q_stand.copy()
        q[2] = z
        self.set_arms(q, *arms)
        return self.legs(q, {f: self.foot0[f] for f in FEET})

    def seeded(self, q, seed):
        q = q.copy()
        for f in FEET:
            q[self.leg_q[f]] = seed[self.leg_q[f]]
        return q

    def set_arms(self, q, right, left):
        for s, a in zip(SIDES, (right, left)):
            q[self.arm_q[s]] = np.radians(a)

    def legs(self, q, targets, tol=1e-9):
        """Leg joints so each foot body frame reaches its target; root and arms unchanged.

        Bounded least squares per leg (trust-region reflective), so joint stops
        hold. targets: {foot: (pos, mat)} for a foot on the floor, or
        (pos, mat, False) for a swinging foot, whose orientation is only a
        preference (it tilts where the ankle stops require). Position, and the
        orientation of feet on the floor, may miss by up to IK_ACCEPT where a
        stop binds; the largest miss is recorded.
        """
        m, d = self.m, self.d
        q = q.copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        worst = 0.
        for f, (p, R, *swing) in targets.items():
            idx, dof, b = self.leg_q[f], self.leg_d[f], m.body(f).id
            W = (SWING_ROT_WEIGHT if swing else ROT_WEIGHT) * np.eye(3)

            def rot_error(x):
                q[idx] = x
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                Re = R @ d.xmat[b].reshape(3, 3).T
                return p - d.xpos[b], .5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])

            lo, hi = self.lo[idx], self.hi[idx]
            x0 = np.clip(q[idx], lo + 1e-9, hi - 1e-9)
            smooth = SWING_SMOOTH if swing else 0.

            def residual(x):
                e, rot = rot_error(x)
                return np.r_[e, W @ rot, smooth * (x - x0)]

            def jacobian(x):
                rot_error(x)
                mujoco.mj_comPos(m, d)
                mujoco.mj_jacBody(m, d, jp, jr, b)
                return np.vstack([-jp[:, dof], -W @ jr[:, dof], smooth * np.eye(6)])

            sol = least_squares(residual, x0, jac=jacobian, bounds=(lo, hi), method='trf',
                xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=300)
            e, rot = rot_error(sol.x)
            worst = max(worst, np.abs(e).max(), 0. if swing else np.abs(rot).max())
            if swing:
                self.swing_tilt = max(self.swing_tilt, float(np.abs(rot).max()))
        if worst > IK_ACCEPT:
            raise ValueError(f'leg IK did not converge ({worst:.2e})')
        self.ik_miss = max(self.ik_miss, worst)
        return q

    def hands(self, q, targets, upright=.02, accept=IK_ACCEPT):
        """Arm joints so each knuckle centre (the mean of body.knuckle_points) reaches its world target; root and
        legs unchanged. targets: {side: point}. The forearm is kept near vertical (knuckles under the elbow) with
        weight `upright` m per unit of its axis' tilt and the shoulder yaw near zero, so the redundant fourth arm
        joint has a preference."""
        m, d = self.m, self.d
        q = q.copy()
        jp = np.zeros((3, m.nv))
        worst = 0.
        for s, p in targets.items():
            idx = self.arm_q[s]
            b = m.body(f'{s}_forearm').id
            k = self.knuckle[f'{s}_forearm']
            lo, hi = self.lo[idx], self.hi[idx]
            x0 = np.clip(q[idx], lo + 1e-9, hi - 1e-9)

            def residual(x):
                q[idx] = x
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                R = d.xmat[b].reshape(3, 3)
                return np.r_[100 * (p - (d.xpos[b] + R @ k)), upright * (R[:2, 2]), 1e-3 * (x - x0), .03 * x[2]]  # x[2]: shoulder yaw, kept near 0

            sol = least_squares(residual, x0, bounds=(lo, hi), method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            worst = max(worst, float(np.abs(residual(sol.x)[:3]).max()) / 100)
        if worst > accept:
            raise ValueError(f'arm IK did not reach the knuckle target ({worst:.2e})')
        self.ik_miss = max(self.ik_miss, worst)
        return q

    def knuckle_world(self, q, s):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        b = self.m.body(f'{s}_forearm').id
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.knuckle[f'{s}_forearm']

    def com(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1].copy()

    def foot_target(self, f, dxy=(0, 0), dz=0., yaw=0.):
        p, R = self.foot0[f]
        return yaw_mat(yaw) @ p + np.r_[dxy, dz], yaw_mat(yaw) @ R


def derivatives(m, qpos, fps):
    """Central-difference qvel (free-joint convention) and qacc; references start and end at rest."""
    dt = 1 / fps
    T = len(qpos)
    qvel = np.zeros((T, m.nv))
    for t in range(1, T - 1):
        mujoco.mj_differentiatePos(m, qvel[t], 2 * dt, qpos[t - 1], qpos[t + 1])
    qacc = np.gradient(qvel, dt, axis=0)
    return qvel, qacc


def finish(B, name, family, tempo, unit, qpos, stance, note, fps=FPS):
    """stance: [T, 2] feet flat on the floor, [T, 2, 4] sole corners on it, or None to infer from heights."""
    pad = int(.25 * fps)
    qpos = np.concatenate([np.repeat(qpos[:1], pad, 0), qpos, np.repeat(qpos[-1:], pad, 0)])
    qvel, qacc = derivatives(B.m, qpos, fps)
    corners = None
    if stance is not None:
        stance = np.asarray(stance, bool)
        stance = stance if stance.ndim == 3 else np.repeat(stance[:, :, None], 4, 2)
        corners = np.concatenate([np.repeat(stance[:1], pad, 0), stance, np.repeat(stance[-1:], pad, 0)])
    miss, tilt, B.ik_miss, B.swing_tilt = B.ik_miss, B.swing_tilt, 0., 0.
    return dict(name=name, family=family, tempo=tempo, tempo_unit=unit, note=note, fps=fps, ik_max_miss_m=float(miss),
        swing_foot_max_tilt_rad=float(tilt),
        qpos=qpos, qvel=qvel, qacc=qacc, corners=corners)


# ---- retained skill rollouts ---------------------------------------------------------------------------------------

def retained(B, name, speed):
    import motions  # parent module: published references and their time stretch
    source = next(r for r in motions.references() if r['name'] == name)
    if sha(body.tracking.ROOT / source['trace']) != source['sha256']:
        raise ValueError(f'{name}: retained trace changed')
    e = np.load(body.tracking.ROOT / source['trace'])
    qpos, qvel = motions.stretch(e['qpos'], e['qvel'], speed)
    replay = 'physics rollout' if speed == 1 else ('faster replay: same stride, quicker cadence' if 'walk' in name
        else 'time-compressed replay, feet planted')
    out = dict(name=name if speed == 1 else f'{name}-x{speed:g}', family=f'retained {name}', tempo=float(speed),
        tempo_unit='x recorded speed', fps=50, qpos=qpos, qvel=qvel, qacc=np.gradient(qvel, 1 / 50, axis=0), corners=None,
        note=f"{source['skill']} ({source['result_id']}); {replay}", ik_max_miss_m=0., swing_foot_max_tilt_rad=0.)
    out['source'] = dict(trace=source['trace'], sha256=source['sha256'])
    return out


# ---- authored moves with planted feet ------------------------------------------------------------------------------

def lean(depth):
    return np.minimum(LEAN_MAX, LEAN_PER_M * np.maximum(depth, 0.))


def planted(B, name, family, tempo, unit, duration, torso, arms, note, ramp=.4):
    """torso(t) -> (pos offset from stand, yaw, pitch, roll); arms(t) -> (right, left) in degrees.

    Blends in and out with a min-jerk weight over a ramp long enough to keep the
    blend under BLEND_SPEED; crouching adds the forward lean.
    """
    feet = {f: B.foot0[f] for f in FEET}
    rest = (ARM_REST, ARM_REST)
    reach = max(max(np.abs(np.radians(np.subtract(arms(t), rest))).max() for t in (0., duration)), 1e-9)
    ramp = max(ramp, 1.875 * reach / BLEND_SPEED)
    t_all = np.arange(0, duration + 2 * ramp, DT)
    qs = []
    for t in t_all:
        w = minjerk(t / ramp) * minjerk((duration + 2 * ramp - t) / ramp)
        tt = np.clip(t - ramp, 0, duration)
        dp, yaw, pitch, roll = torso(tt)
        dp = w * np.asarray(dp)
        q = np.zeros(B.m.nq)
        q[:3] = np.r_[0, 0, STAND_Z] + dp
        q[3:7] = quat(w * yaw, w * pitch + lean(-dp[2]), w * roll)
        r, l = arms(tt)
        B.set_arms(q, *(np.asarray(a) * w + np.asarray(b) * (1 - w) for a, b in zip((r, l), rest)))
        qs.append(B.legs(B.seeded(q, qs[-1] if qs else B.q_stand), feet))
    qpos = np.array(qs)
    return finish(B, name, family, tempo, unit, qpos, np.ones((len(qpos), 2), bool), note)


def osc(t, f, duration, phase=0., edge=.5):
    """sin(2 pi f t + phase) under an amplitude envelope that rises and falls over edge s, so a move starts and ends at rest."""
    return minjerk(t / edge) * minjerk((duration - t) / edge) * np.sin(2 * np.pi * f * t + phase)


def still(t):
    return (0, 0, 0), 0., 0., 0.


def chest_beat(B, f):
    base = np.array([50., 35., -50., 100.])
    amp = np.array([8., 0., 22., 18.])
    arms = lambda t: (base + amp * osc(t, f, 2.5), base + amp * osc(t, f, 2.5, np.pi))
    return planted(B, f'chest-beat-{f:g}hz', 'chest beat', f, 'beats/s per hand', 2.5, still, arms,
        'alternating fists toward the chest; shoulder yaw and elbow carry the beat')


def overhead_wave(B, f):
    def arms(t):
        a = np.array([140., 25. + 20. * osc(t, f, 2.5), 0., 30. + 20. * osc(t, f, 2.5, np.pi / 2)])
        return a, a
    return planted(B, f'overhead-wave-{f:g}hz', 'overhead wave', f, 'waves/s', 2.5, still, arms,
        'both arms overhead swaying side to side, elbows pumping a quarter cycle behind')


def arm_display(B, f):
    def arms(t):
        s = osc(t, f, 2.5)
        return np.array([40. + 55. * s, 15., 0., 50.]), np.array([40. - 55. * s, 15., 0., 50.])
    return planted(B, f'arm-display-{f:g}hz', 'arm display swing', f, 'swings/s', 2.5, still, arms,
        'big alternating forward-back arm swings (excited display), shoulder pitch -15 to 95 deg')


def knee_bounce(B, f):
    dip = lambda t: .5 * (1 + osc(t, f, 2.5, -np.pi / 2)) * minjerk(t / .5) * minjerk((2.5 - t) / .5)
    torso = lambda t: ((0, 0, -.025 * dip(t)), 0., 0., 0.)
    arms = lambda t: ((ARM_REST[0] + 20 * dip(t), 15., 0., 30.),) * 2
    return planted(B, f'knee-bounce-{f:g}hz', 'knee bounce', f, 'bounces/s', 2.5, torso, arms,
        'excited bouncing on bent knees, 25 mm torso travel with forward lean, arms lifting with each dip')


def side_sway(B, f):
    torso = lambda t: ((0, .025 * osc(t, f, 3.), -.01), 0., 0., np.radians(5) * osc(t, f, 3.))
    arms = lambda t: ((10., 20. + 10 * osc(t, f, 3.), 0., 20.), (10., 20. - 10 * osc(t, f, 3.), 0., 20.))
    return planted(B, f'side-sway-{f:g}hz', 'side sway', f, 'sways/s', 3., torso, arms,
        'rocking side to side over planted feet 10 mm below stand height, +-25 mm and +-5 deg roll')


def quick_crouch(B, T):
    def torso(t):
        s = minjerk(t / T) - minjerk((t - T - .3) / T)
        return (0, 0, -.035 * s), 0., 0., 0.
    arms = lambda t: ((ARM_REST[0] + 25 * (minjerk(t / T) - minjerk((t - T - .3) / T)), 15., 0., 30.),) * 2
    return planted(B, f'quick-crouch-{T:g}s', 'quick crouch', T, 's down (same up)', 2 * T + .6, torso, arms,
        'drop 35 mm (leaning 30 deg) in T, hold 0.3 s, rise in T', ramp=.3)


def contact_quintic(p0, v0, a0, p1, v1, a1, T0):
    """A quintic between the two states over a whole number of frames, lengthened from T0 until it never
    accelerates downward faster than gravity (feet on the floor cannot pull the body down)."""
    for T in T0 * np.linspace(1, 4, 61):
        n = max(2, int(round(T / DT)))
        c = quintic(p0, v0, a0, p1, v1, a1, n * DT)
        t = np.arange(n) * DT
        if np.polyval(np.polyder(c[::-1], 2), t).min() >= -body.G - 1e-6:
            return np.polyval(c[::-1], t)
    raise ValueError('no contact-feasible quintic')


def hops(B, h, count=3, crouch=.05, takeoff=.02, pitch=np.radians(25), sit_back=.01):
    """count vertical hops of about h m flight rise from a crouch depth below stand, taking off at takeoff depth.

    The torso leans forward and sits back (ankle stop) while first crouching and
    holds that through the hops, so the hops are purely vertical; the pose
    frozen in the air is exactly ballistic. Flight lasts a whole number of
    frames, which sets the takeoff speed.
    """
    zc, zto = STAND_Z - crouch, STAND_Z - takeoff
    n_flight = int(round(2 * np.sqrt(2 * body.G * h) / body.G / DT))
    v = body.G * n_flight * DT / 2
    span = zto - zc
    pieces = [(STAND_Z - crouch * minjerk(np.arange(0, .5, DT) / .5), True)]
    for i in range(count):
        pieces.append((contact_quintic(zc, 0, 0, zto, v, -body.G, 2 * span / v), True))
        t = np.arange(n_flight) * DT
        pieces.append((zto + v * t - body.G * t * t / 2, False))
        pieces.append((contact_quintic(zto, -v, -body.G, zc, 0, 0, 2.5 * span / v), True))
        pieces.append((np.full(int(.12 / DT), zc), True))
    pieces.append((zc + crouch * minjerk(np.arange(0, .5, DT) / .5), True))
    z = np.concatenate([p for p, _ in pieces])
    ground = np.concatenate([np.full(len(p), g) for p, g in pieces])
    arms = ((10., 15., 0., 25.),) * 2
    feet = {f: B.foot0[f] for f in FEET}
    qs = []
    for zz, g in zip(z, ground):
        if not g:
            q = qs[-1].copy()
            q[2] = zz
            qs.append(q)
            continue
        w = minjerk((STAND_Z - zz) / takeoff)
        q = np.zeros(B.m.nq)
        q[:3], q[3:7] = (-sit_back * w, 0, zz), quat(0, pitch * w)
        B.set_arms(q, *arms)
        qs.append(B.legs(B.seeded(q, qs[-1] if qs else B.q_stand), feet))
    qpos = np.array(qs)
    rise = v * v / (2 * body.G)
    return finish(B, f'hop-{h * 100:g}cm', 'hops', round(rise * 100, 2), 'cm flight rise', qpos, np.repeat(ground[:, None], 2, 1),
        f'{count} flat-footed hops from a {crouch * 1000:.0f} mm crouch (torso leaning {np.degrees(pitch):.0f} deg), takeoff '
        f'{takeoff * 1000:.0f} mm below stand at {v:.2f} m/s, {n_flight * DT * 1000:.0f} ms flight, soft landing; no toe push')


# ---- gaits ---------------------------------------------------------------------------------------------------------

def preview_gains(zc, N):
    """Kajita/Katayama ZMP preview control gains for the cart-table model at DT."""
    A = np.array([[1, DT, DT * DT / 2], [0, 1, DT], [0, 0, 1]])
    B = np.array([[DT ** 3 / 6], [DT * DT / 2], [DT]])
    C = np.array([[1, 0, -zc / body.G]])
    Ab = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    Bb = np.vstack([C @ B, B])
    Q = np.diag([1., 0, 0, 0])
    R = np.array([[1e-6]])
    P = solve_discrete_are(Ab, Bb, Q, R)
    S = np.linalg.inv(R + Bb.T @ P @ Bb)
    K = S @ Bb.T @ P @ Ab
    Ac = Ab - Bb @ K
    I = np.array([[1.], [0], [0], [0]])
    X = -Ac.T @ P @ I
    Gd = [-K[0, 0]]
    for _ in range(1, N):
        Gd.append((S @ Bb.T @ X).item())
        X = Ac.T @ X
    return A, B, C, K[0, 0], K[0, 1:], np.array(Gd)


def preview(zmp, zc, horizon=1.6):
    """COM path (T, 2) whose cart-table ZMP follows zmp (T, 2)."""
    N = int(horizon / DT)
    A, B, C, Gi, Gx, Gd = preview_gains(zc, N)
    ref = np.concatenate([zmp, np.repeat(zmp[-1:], N, 0)])
    out = np.zeros((len(zmp), 2))
    for axis in range(2):
        x = np.array([ref[0, axis], 0, 0])
        e = 0.
        for k in range(len(zmp)):
            e += (C @ x).item() - ref[k, axis]
            u = -Gi * e - Gx @ x - Gd @ ref[k + 1:k + 1 + N, axis]
            x = A @ x + B[:, 0] * u
            out[k, axis] = x[0]
    return out


def gait(B, name, family, tempo, unit, steps, step_time, note, ds_frac=.25, swing_h=.025, arm_swing=3., settle=.8,
         toe_off=TOE_OFF):
    """steps: [(foot, dxy, yaw)] each a target relative to that foot's neutral frame, in order.

    Each step: double support while the ZMP moves onto the support foot and the
    stepping foot's heel rises toe_off about its toe edge (toe corners only on the
    floor), then single support while it swings (min-jerk, sin^2 lift) and its
    pitch returns to flat for a flat touchdown.
    """
    pose = {f: (np.zeros(2), 0.) for f in FEET}
    inward = {f: np.array([0, -np.sign(B.foot0[f][0][1])]) for f in FEET}
    centre = lambda f: (B.sole0[f] - B.foot0[f][0])[:2] + inward[f] * ZMP_INSET
    corners = body.sole_corners(B.m)
    toe = {f: corners[f][:2].mean(0) for f in FEET}  # front edge (sole_corners lists the two +x corners first)
    flat, toes, air = np.ones(4, bool), np.array([1, 1, 0, 0], bool), np.zeros(4, bool)

    def place(f, dxy, dz=0., yaw=0., pitch=0.):
        """Foot origin and frame; pitch (toes down) pivots about the toe edge of the flat placement."""
        p, R = B.foot_target(f, np.add(dxy, yaw_mat(yaw)[:2, :2] @ inward[f] * GAIT_NARROW), dz, yaw)
        if pitch:
            pivot = p + R @ toe[f]
            c, s_ = np.cos(pitch), np.sin(pitch)
            R = yaw_mat(yaw) @ np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ yaw_mat(-yaw) @ R
            p = pivot - R @ toe[f]
        return p, R

    def sole(f):
        dxy, yaw = pose[f]
        p, _ = place(f, dxy, 0, yaw)
        return p[:2] + yaw_mat(yaw)[:2, :2] @ centre(f)

    T_ds, T_ss = ds_frac * step_time, (1 - ds_frac) * step_time
    zmp, frames = [], []  # frames: {foot: (pos, mat, corner mask, swinging)}

    def still(f):
        return (*place(f, *pose[f][:1], 0., pose[f][1]), flat, False)

    def hold(duration, z0, z1, lifting=None):
        n = int(round(duration / DT))
        for i in range(n):
            zmp.append(z0 + (z1 - z0) * minjerk((i + 1) / n))
            fr = {f: still(f) for f in FEET}
            if lifting:
                fr[lifting] = (*place(lifting, pose[lifting][0], 0., pose[lifting][1], toe_off * minjerk((i + 1) / n)), toes, False)
            frames.append(fr)

    mid = lambda: (sole(FEET[0]) + sole(FEET[1])) / 2
    hold(settle, mid(), mid())
    current = mid()
    for foot, dxy, yaw in steps:
        support = FEET[1 - FEET.index(foot)]
        hold(T_ds, current, sole(support), lifting=foot)
        current = sole(support)
        (dxy0, yaw0), goal = pose[foot], (np.asarray(dxy, float), float(yaw))
        p0, _ = place(foot, dxy0, 0., yaw0, toe_off)
        p1, _ = place(foot, goal[0], 0., goal[1])
        n = int(round(T_ss / DT))
        for i in range(n):
            s = minjerk((i + 1) / n)
            yaw_s = yaw0 + (goal[1] - yaw0) * s
            p = p0 + (p1 - p0) * s + np.r_[0, 0, swing_h * np.sin(np.pi * (i + 1) / n) ** 2]
            _, R = place(foot, goal[0], 0., yaw_s, toe_off * (1 - s))
            zmp.append(current)
            frames.append({support: still(support), foot: (p, R, air, True)})
        pose[foot] = goal
    hold(T_ds + .3, current, mid())
    hold(settle, mid(), mid())
    zmp = np.array(zmp)
    q0 = B.stand(GAIT_Z)
    q0[3:7] = quat(0, GAIT_LEAN)
    q0 = B.legs(q0, {f: place(f, (0, 0)) for f in FEET})
    zc = B.com(q0)[2]
    com = preview(zmp, zc)
    qs, contact = [], []
    q = q0.copy()
    for k, fr in enumerate(frames):
        targets = {f: (fr[f][0], fr[f][1], *((False,) if fr[f][3] else ())) for f in FEET}
        yaw = np.mean([np.arctan2(fr[f][1][1, 0], fr[f][1][0, 0]) for f in FEET])
        side = yaw_mat(-yaw) @ np.r_[com[k] - (fr[FEET[0]][0][:2] + fr[FEET[1]][0][:2]) / 2, 0]
        q[3:7] = quat(yaw, GAIT_LEAN, GAIT_ROLL * side[1])
        q[2] = GAIT_Z
        rel = {f: (yaw_mat(-yaw) @ (fr[f][0] - q[:3]))[0] for f in FEET}
        B.set_arms(q, *((ARM_REST[0] - np.degrees(arm_swing * rel[f]), *ARM_REST[1:]) for f in FEET))
        for _ in range(4):
            q = B.legs(q, targets)
            q[:2] += com[k] - B.com(q)[:2]
        qs.append(B.legs(q, targets))
        contact.append([fr[f][2] for f in FEET])
    qpos = np.array(qs)
    return finish(B, name, family, tempo, unit, qpos, np.array(contact),
        note + f'; step {step_time:.2f} s ({ds_frac:.0%} double support with {np.degrees(toe_off):.0f} deg heel lift), '
        f'swing {swing_h * 1000:.0f} mm, COM height {zc:.3f} m')


def turn(B, rate, total=np.radians(90)):
    per = np.radians(30)
    n = int(round(total / per))
    T = per / np.radians(rate)
    steps = []
    yaw = {f: 0. for f in FEET}
    for i in range(2 * n):
        f = FEET[i % 2]
        yaw[f] = per * (i // 2 + 1)
        steps.append((f, (0., 0.), yaw[f]))
    return gait(B, f'turn-{rate:g}dps', 'turn in place', rate, 'deg/s', steps, T, f'{np.degrees(total):.0f} deg in 30 deg step pairs, counter-clockwise')


def march(B, step_time, n=8):
    steps = [(FEET[i % 2], (0., 0.), 0.) for i in range(n)]
    return gait(B, f'march-{step_time:g}s', 'march in place', step_time, 's per step', steps, step_time,
        'stomping in place, 40 mm foot lift', swing_h=.04)


def catalogue(B):
    out = []
    for name, speeds in RETAINED.items():
        out += [lambda n=name, s=s: retained(B, n, s) for s in speeds]
    out += [lambda f=f: chest_beat(B, f) for f in (1, 2, 3)]
    out += [lambda f=f: overhead_wave(B, f) for f in (1, 2, 3)]
    out += [lambda f=f: arm_display(B, f) for f in (.5, 1, 2)]
    out += [lambda f=f: knee_bounce(B, f) for f in (1.5, 2.5, 3.5)]
    out += [lambda f=f: side_sway(B, f) for f in (.75, 1.5, 2.5)]
    out += [lambda T=T: quick_crouch(B, T) for T in (.6, .35, .2)]
    out += [lambda h=h: hops(B, h) for h in (.01, .02, .04)]
    out += [lambda r=r: turn(B, r) for r in (30, 60)]
    out += [lambda T=T: march(B, T) for T in (.5, .33)]
    return out


# ---- ape catalogue (agility step 4): in-between tempos and ape-character moves -------------------------------------
# The agility catalogue above stays as it is (its hashes are cited by the demand, allocation and tape records).
# Here: the tempos between the agility set's feasible and infeasible ones, so a selection can keep each family's
# fastest tempo the mixed servo set and the floor can carry, plus new moves with an ape character, all with
# planted feet.

def ape_crouch_bounce(B, f):
    """Low ape stance: 30 mm below stand, bouncing 12 mm at f, arms hanging forward and swinging with the bounce."""
    dip = lambda t: .5 * (1 + osc(t, f, 3., -np.pi / 2)) * minjerk(t / .5) * minjerk((3. - t) / .5)
    low = lambda t: minjerk(t / .6) * minjerk((3. - t) / .6)
    torso = lambda t: ((0, 0, -.03 * low(t) - .012 * dip(t)), 0., 0., 0.)
    arms = lambda t: ((25. + 15 * dip(t), 12., 0., 40. + 15 * dip(t)), (25. + 15 * dip(t), 12., 0., 40. + 15 * dip(t)))
    return planted(B, f'ape-crouch-bounce-{f:g}hz', 'ape crouch bounce', f, 'bounces/s', 3., torso, arms,
        'low ape stance 30 mm below stand, 12 mm bounces, long arms hanging forward and swinging with each dip')


def excited_beat_bounce(B, f):
    """Knee bounce and chest beat together: the excited display."""
    dip = lambda t: .5 * (1 + osc(t, f, 3., -np.pi / 2)) * minjerk(t / .5) * minjerk((3. - t) / .5)
    torso = lambda t: ((0, 0, -.02 * dip(t)), 0., 0., 0.)
    base, amp = np.array([50., 35., -50., 100.]), np.array([8., 0., 22., 18.])
    arms = lambda t: (base + amp * osc(t, f, 3.), base + amp * osc(t, f, 3., np.pi))
    return planted(B, f'excited-beat-bounce-{f:g}hz', 'excited beat bounce', f, 'beats/s', 3., torso, arms,
        'bouncing 20 mm on bent knees while beating alternating fists on the chest')


def side_rock_reach(B, f):
    """Rocking side to side, the arm on the rising side reaching overhead."""
    def torso(t):
        s = osc(t, f, 4.)
        return (0, .025 * s, -.012), 0., 0., np.radians(6) * s
    def arms(t):
        s = osc(t, f, 4.)
        up = lambda x: (40. + 100. * max(x, 0.), 20., 0., 30. + 20. * max(x, 0.))
        return np.array(up(-s)), np.array(up(s))
    return planted(B, f'side-rock-reach-{f:g}hz', 'side rock reach', f, 'rocks/s', 4., torso, arms,
        'rocking +-25 mm with +-6 deg roll, the arm opposite the lean reaching up to 140 deg')


def crouch_scan(B, f):
    """Drop 30 mm and hold, scanning the torso left and right with the arms out front."""
    low = lambda t: minjerk(t / .6) * minjerk((4. - t) / .6)
    torso = lambda t: ((0, 0, -.03 * low(t)), np.radians(10) * osc(t, f, 4., edge=.8), 0., 0.)
    arms = lambda t: ((45. * low(t) + ARM_REST[0], 15., 0., 35.),) * 2
    return planted(B, f'crouch-scan-{f:g}hz', 'crouch scan', f, 'scans/s', 4., torso, arms,
        'crouch 30 mm and hold, torso turning +-10 deg to look around, arms reaching forward')


def side_shuffle(B, step_time, width=.03, n=3):
    """Side shuffle: n sideways steps toward +y, the foot on that side stepping out and the other closing up, then n back."""
    lead, close = sorted(FEET, key=lambda f: -B.foot0[f][0][1])
    steps = []
    for i in range(1, n + 1):
        steps += [(lead, (0., width * i), 0.), (close, (0., width * i), 0.)]
    for i in range(n - 1, -1, -1):
        steps += [(close, (0., width * i), 0.), (lead, (0., width * i), 0.)]
    return gait(B, f'side-shuffle-{step_time:g}s', 'side shuffle', step_time, 's per step', steps, step_time,
        f'{n} sideways steps of {width * 1000:.0f} mm, {lead} stepping out and {close} closing up, then back',
        swing_h=.02)


# ---- knuckle support (long arms: APE0_ARM_SCALE) ------------------------------------------------------------------

# Knuckle stance on the ape body (upper arm 1.6x, forearm 1.27x, 30 mm fist, feet 1.3x), found by scanning lean, hip
# height, hip offset and knuckle placement (cove50 agility loop, arm options): hips 0.20 m up and 45 mm behind the
# ankles, torso leaning 68 deg, knuckles 30 mm ahead of the shoulders (arms angled forward, as an ape's); hips about
# 102 deg flexed (limit 110), ankles -25 (limit -35), centre of mass about 55% of the way from heels to knuckles.
# Knuckles under the shoulders put 90% of the weight toward the hands; 50 mm ahead leaves the arms no slack to rock.
KNUCKLE = dict(pitch=np.radians(68), hip_z=.20, hip_x=-.045, ahead=.03)
HAND_CONTACT = .004  # knuckle centre within this of the floor bears load


def hip_local(B):
    q = B.q_stand
    B.d.qpos[:] = q
    mujoco.mj_kinematics(B.m, B.d)
    hip = np.mean([B.d.xanchor[B.m.joint(f'{s}_hip_pitch').id] for s in SIDES], 0)
    return (hip - q[:3]) @ B.d.xmat[1].reshape(3, 3)


def torso_at(B, hip, pitch, roll=0.):
    """Root position and orientation placing the hip-pitch centre at `hip` with the given lean and roll."""
    quat_ = quat(pitch=pitch, roll=roll)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat_)
    return hip - R.reshape(3, 3) @ hip_local(B), quat_


def knuckle_frames(B, name, family, tempo, unit, hold, path, note, drop=.8, limbs=None, advance=0., hip_x=None):
    """Drop from stand onto the knuckles over `drop` s, run path(t) -> (hip offset xyz, extra lean, roll) for `hold`
    s, and rise again. Legs by IK to the feet, arms by IK to the knuckle targets.

    limbs(t) -> ({foot: (dx, dz)}, {side: (dx, dz)}) moves feet and knuckles from their stance placement during the
    hold (dz > 0: that limb swings, its contact is off); `advance` is the forward travel at the end of the hold, kept
    through the rise. Without limbs every hand and foot stays planted."""
    K = KNUCKLE
    feet_x = np.mean([B.foot0[f][0][0] for f in FEET])
    B.d.qpos[:] = B.q_stand
    mujoco.mj_kinematics(B.m, B.d)
    hip0 = np.mean([B.d.xanchor[B.m.joint(f'{s}_hip_pitch').id] for s in SIDES], 0)
    hip1 = np.array([feet_x + (K['hip_x'] if hip_x is None else hip_x), 0., K['hip_z']])
    start = {s: B.knuckle_world(B.q_stand, s) for s in SIDES}
    # Knuckle floor points: under the shoulders of the stance pose, `ahead` forward, at the hanging arm's width.
    q = np.zeros(B.m.nq)
    q[:3], q[3:7] = torso_at(B, hip1, K['pitch'])
    B.d.qpos[:] = q
    mujoco.mj_kinematics(B.m, B.d)
    floor = {s: np.array([B.d.xanchor[B.m.joint(f'{s}_shoulder_pitch').id][0] + K['ahead'], start[s][1], 0.]) for s in SIDES}
    reach = (55., 10., 0., 25.)  # arm seed for the drop
    frames, contact, planted = [], [], []
    n_drop, n_hold = int(round(drop * FPS)), int(round(hold * FPS))
    prev = B.q_stand
    shift = lambda x: np.array([x, 0., 0.])
    for k in range(2 * n_drop + n_hold + 1):
        holding = n_drop < k <= n_drop + n_hold
        if k <= n_drop:
            w, t, adv = minjerk(k / n_drop), 0., 0.
        elif holding:
            w, t, adv = 1., (k - n_drop) * DT, 0.
        else:
            w, t, adv = minjerk((2 * n_drop + n_hold - k) / n_drop), hold, advance
        dp, dpitch, roll = path(t) if holding else ((0., 0., 0.), 0., 0.)
        hip = (1 - w) * hip0 + w * hip1 + np.asarray(dp) + shift(adv)
        moves = limbs(t) if holding and limbs else ({}, {})
        feet, down = {}, []
        for f in FEET:
            dx, dz = moves[0].get(f, (adv, 0.))
            p, R = B.foot0[f]
            feet[f] = (p + np.array([dx, 0., dz]), R, *((False,) if dz > 0 else ()))
            down.append(dz <= 0)
        q = np.zeros(B.m.nq)
        q[:3], q[3:7] = torso_at(B, hip, w * K['pitch'] + dpitch, roll)
        for f in FEET:
            q[B.leg_q[f]] = prev[B.leg_q[f]]
        B.set_arms(q, *(np.asarray(ARM_REST) * (1 - w) + np.asarray(reach) * w,) * 2)
        if k:
            q[np.concatenate([B.arm_q[s] for s in SIDES])] = prev[np.concatenate([B.arm_q[s] for s in SIDES])]
        q = B.legs(q, feet)
        # The knuckle target follows the hand down: the floor point weighted in as the drop ends (w^3 keeps the hand
        # high early, so it does not sweep the floor).
        target, air = {}, False
        for s_ in SIDES:
            dx, dz = moves[1].get(s_, (adv, 0.))
            target[s_] = (1 - w ** 3) * (start[s_] + shift(adv)) + w ** 3 * (floor[s_] + np.array([dx, 0., dz]))
            air = air or dz > 0
        # 2 mm into a knuckle pad on the floor; a hand in the air only follows its path.
        q = B.hands(q, target, upright=.005, accept=.002 if w ** 3 > .999 and not air else .01)
        frames.append(q)
        contact.append([B.knuckle_world(q, s_)[2] < HAND_CONTACT for s_ in SIDES])
        planted.append(down)
        prev = q
    out = finish(B, name, family, tempo, unit, np.array(frames), np.array(planted), note)
    pad = int(.25 * FPS)
    contact = np.array(contact)
    out['hand_contacts'] = np.concatenate([np.repeat(contact[:1], pad, 0), contact, np.repeat(contact[-1:], pad, 0)])
    return out


def knuckle_walk(B, cycle, stride=.03, n=3, lift=.015, foot_lift=.012, sway=.012, hip_x=-.055):
    """Diagonal-sequence walk on feet and knuckles, as chimpanzees knuckle-walk: left hand, right foot, right hand,
    left foot, each swinging for a quarter of the cycle while the other three bear the load. The hips advance
    `stride` per cycle at constant speed and sway `sway` m toward the support triangle (toward the side opposite
    the swinging limb). Starts and ends standing. The hips sit 10 mm further back than the knuckle stance, so a
    planted foot's ankle keeps margin to its -35 deg stop as the hips pass over it. Feet lift `foot_lift` (12 mm): in
    this crouch 15 mm is reachable only by flipping the hip roll, and higher hips leave the knuckles short of the
    floor. 8 mm lifts were too small for the swing and contact terms to see (the trained walk dragged its feet)."""
    order = (('hand', 'L'), ('foot', 'R'), ('hand', 'R'), ('foot', 'L'))
    hold = n * cycle + .5  # n cycles, then 0.5 s on all fours before rising

    def limbs(t):
        feet, hands = {}, {}
        for i, (kind, side) in enumerate(order):
            # completed swings plus progress through the current one
            phase = (t / cycle) - i / 4
            done = np.clip(np.floor(phase) + 1, 0, n) if phase >= 0 else 0
            frac = phase - np.floor(phase)
            swinging = 0 <= phase < n and frac < .25
            s = minjerk(frac / .25) if swinging else 0.
            dx = stride * ((done - 1) + s) if swinging else stride * done
            dz = (foot_lift if kind == 'foot' else lift) * np.sin(np.pi * frac / .25) ** 2 if swinging else 0.
            (feet if kind == 'foot' else hands)[f'{side}_foot' if kind == 'foot' else side] = (dx, dz)
        return feet, hands

    def path(t):
        v = stride / cycle
        x = v * np.clip(t, 0, n * cycle)
        envelope = minjerk(t / cycle) * minjerk((n * cycle - t) / cycle) if t < n * cycle else 0.
        return (x, sway * envelope * np.cos(2 * np.pi * t / cycle), 0.), 0., 0.

    return knuckle_frames(B, f'knuckle-walk-{cycle:g}s', 'knuckle walk', cycle, 's per cycle', hold, path,
        f'diagonal-sequence knuckle walk, {n} cycles of {stride * 1000:.0f} mm, hands lift {lift * 1000:.0f} mm and feet {foot_lift * 1000:.0f} mm, '
        f'hips swaying {sway * 1000:.0f} mm toward the support triangle', limbs=limbs, advance=n * stride, hip_x=hip_x)


def knuckle_drop(B, T):
    return knuckle_frames(B, f'knuckle-drop-{T:g}s', 'knuckle drop', T, 's per transition', 1.0, lambda t: ((0, 0, 0), 0., 0.),
        'drop from stand onto the knuckles (torso 68 deg, hips 0.20 m), hold 1 s, rise', drop=T)


def knuckle_rock(B, f):
    path = lambda t: ((.01 * osc(t, f, 3.), 0., -.004 * osc(t, f, 3.)), 0., 0.)
    return knuckle_frames(B, f'knuckle-rock-{f:g}hz', 'knuckle rock', f, 'rocks/s', 3., path,
        'on the knuckles, rocking the hips +-10 mm fore and aft, weight passing between hands and feet')


def knuckle_sway(B, f):
    path = lambda t: ((0., .01 * osc(t, f, 3.), 0.), 0., np.radians(3) * osc(t, f, 3.))
    return knuckle_frames(B, f'knuckle-sway-{f:g}hz', 'knuckle sway', f, 'sways/s', 3., path,
        'on the knuckles, swaying the hips +-10 mm side to side with +-3 deg roll')


def knuckle_bounce(B, f):
    dip = lambda t: .5 * (1 + osc(t, f, 3., -np.pi / 2)) * minjerk(t / .5) * minjerk((3. - t) / .5)
    path = lambda t: ((0., 0., -.01 * dip(t)), 0., 0.)
    return knuckle_frames(B, f'knuckle-bounce-{f:g}hz', 'knuckle bounce', f, 'bounces/s', 3., path,
        'on the knuckles, bouncing the hips 10 mm at f, the excited ape on all fours')


def knuckle_catalogue(B):
    out = [lambda T=T: knuckle_drop(B, T) for T in (.8, .6, .45)]
    out += [lambda f=f: knuckle_rock(B, f) for f in (1, 1.5, 2)]
    out += [lambda f=f: knuckle_sway(B, f) for f in (1, 1.5)]
    out += [lambda f=f: knuckle_bounce(B, f) for f in (1.5, 2, 2.5)]
    out += [lambda T=T: knuckle_walk(B, T) for T in (2., 1.6, 1.2, 1., .8)]
    return out


def ape_catalogue(B):
    out = [lambda s=s: retained(B, 'forward-walk', s) for s in (1.1,)]
    out += [lambda n=n, s=s: retained(B, n, s) for n in ('deep-squat', 'deep-rise', 'bilateral-arms', 'alternating-arms') for s in (3,)]
    out += [lambda f=f: chest_beat(B, f) for f in (2.5,)]
    out += [lambda f=f: overhead_wave(B, f) for f in (2.5,)]
    out += [lambda f=f: arm_display(B, f) for f in (1.5,)]
    out += [lambda f=f: knee_bounce(B, f) for f in (2, 3)]
    out += [lambda f=f: side_sway(B, f) for f in (2,)]
    out += [lambda T=T: quick_crouch(B, T) for T in (.27,)]
    out += [lambda h=h: hops(B, h) for h in (.015,)]
    out += [lambda T=T: march(B, T) for T in (.4,)]
    out += [lambda f=f: ape_crouch_bounce(B, f) for f in (1.5, 2, 2.5)]
    out += [lambda f=f: excited_beat_bounce(B, f) for f in (1.5, 2, 2.5)]
    out += [lambda f=f: side_rock_reach(B, f) for f in (.5, .75, 1)]
    out += [lambda f=f: crouch_scan(B, f) for f in (.5,)]
    out += [lambda T=T: side_shuffle(B, T) for T in (.8, .6, .5, .4, .33)]
    return out


# ---- validation catalogue: motions the target tape never contains ---------------------------------------------------
# The score is agility across all motions, including motions the controller has not trained on (AGENTS.md). Three
# kinds, all within the trained range of speed and size: tempos between (or just below) each trained family's
# tempos, mirror images (the trained turn only goes counter-clockwise, the shuffle only starts to +y), and arm moves
# layered on leg moves, which the tape only ever shows apart. Build with the long-arm settings, as the target tape.

def _arm_qadr(B):
    return [B.m.jnt_qposadr[B.m.joint(f'{s}_{a}').id] for s in SIDES for a in ARM]


def mirrored(B, mo, name):
    """Mirror image across the body's sagittal plane: left and right joints swap (roll and yaw axes are mirrored
    between sides, pitch axes are shared), neck yaw and the root's y, roll and yaw change sign."""
    m, q = B.m, mo['qpos']
    out = q.copy()
    out[:, 1] = -q[:, 1]
    out[:, 3:7] = q[:, 3:7] * np.array([1., -1., 1., -1.])
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        n, a = m.joint(j).name, m.jnt_qposadr[j]
        if n[:2] in ('R_', 'L_'):
            out[:, a] = q[:, m.jnt_qposadr[m.joint(('L_' if n[0] == 'R' else 'R_') + n[2:]).id]]
        elif n == 'neck_yaw':
            out[:, a] = -q[:, a]
    qvel, qacc = derivatives(m, out, mo['fps'])
    res = dict(mo, name=name, family=f"{mo['family']} (mirrored)", note=f"mirror image of {mo['name']}: {mo['note']}",
        qpos=out, qvel=qvel, qacc=qacc, corners=None if mo.get('corners') is None else mo['corners'][:, ::-1])
    if 'hand_contacts' in mo:
        res['hand_contacts'] = mo['hand_contacts'][:, ::-1]
    return res


def overlay(B, legs, arms, name):
    """`legs` with the shoulder and elbow angles of `arms`, both centred in time; the shorter one holds its first and
    last frame (both start and end at rest). Feet and torso come from `legs` unchanged."""
    fps = legs['fps']
    t_arms = np.arange(len(arms['qpos'])) / arms['fps']
    n_arms = int(round(t_arms[-1] * fps)) + 1
    adr = _arm_qadr(B)
    qa = np.stack([np.interp(np.arange(n_arms) / fps, t_arms, arms['qpos'][:, a]) for a in adr], 1)
    n = max(len(legs['qpos']), n_arms)
    hold = lambda x, k: np.concatenate([np.repeat(x[:1], (k - len(x)) // 2, 0), x, np.repeat(x[-1:], k - len(x) - (k - len(x)) // 2, 0)])
    q = hold(legs['qpos'], n)
    q[:, adr] = hold(qa, n)
    qvel, qacc = derivatives(B.m, q, fps)
    corners = None if legs.get('corners') is None else hold(legs['corners'], n)
    return dict(legs, name=name, family=f"{legs['family']} with {arms['family']}", tempo=legs['tempo'],
        note=f"{legs['name']} legs and torso with the arms of {arms['name']}", qpos=q, qvel=qvel, qacc=qacc, corners=corners)


def validation_catalogue(B):
    out = [lambda f=f: chest_beat(B, f) for f in (1.5,)]
    out += [lambda f=f: overhead_wave(B, f) for f in (1.5,)]
    out += [lambda f=f: arm_display(B, f) for f in (.75,)]
    out += [lambda f=f: knee_bounce(B, f) for f in (1.75, 2.25)]
    out += [lambda f=f: side_sway(B, f) for f in (1.,)]
    out += [lambda T=T: quick_crouch(B, T) for T in (.45,)]
    out += [lambda r=r: turn(B, r) for r in (45,)]
    out += [lambda T=T: march(B, T) for T in (.6,)]
    out += [lambda T=T: side_shuffle(B, T) for T in (.7, .45)]
    out += [lambda f=f: ape_crouch_bounce(B, f) for f in (1.75,)]
    out += [lambda f=f: excited_beat_bounce(B, f) for f in (1.75,)]
    out += [lambda f=f: side_rock_reach(B, f) for f in (.4,)]
    out += [lambda f=f: crouch_scan(B, f) for f in (.4,)]
    out += [lambda n=n, s=s: retained(B, n, s) for n, s in (('forward-walk', 1.05), ('deep-squat', 2.5), ('deep-rise', 1.5),
        ('bilateral-arms', 1.5), ('alternating-arms', 1.5))]
    out += [lambda T=T: knuckle_drop(B, T) for T in (.7,)]
    out += [lambda f=f: knuckle_rock(B, f) for f in (1.25,)]
    out += [lambda f=f: knuckle_sway(B, f) for f in (1.25,)]
    out += [lambda f=f: knuckle_bounce(B, f) for f in (1.75,)]
    out += [lambda T=T: knuckle_walk(B, T) for T in (1.4,)]
    out += [lambda: mirrored(B, turn(B, 30), 'turn-30dps-clockwise'),
            lambda: mirrored(B, side_shuffle(B, .6), 'side-shuffle-0.6s-mirrored'),
            lambda: mirrored(B, side_rock_reach(B, .5), 'side-rock-reach-0.5hz-mirrored'),
            lambda: mirrored(B, knuckle_walk(B, 1.2), 'knuckle-walk-1.2s-mirrored')]
    out += [lambda: overlay(B, march(B, .5), overhead_wave(B, 1), 'march-0.5s-with-overhead-wave-1hz'),
            lambda: overlay(B, side_shuffle(B, .6), chest_beat(B, 1), 'side-shuffle-0.6s-with-chest-beat-1hz'),
            lambda: overlay(B, turn(B, 30), arm_display(B, .5), 'turn-30dps-with-arm-display-0.5hz'),
            lambda: overlay(B, retained(B, 'forward-walk', 1), overhead_wave(B, 1), 'forward-walk-with-overhead-wave-1hz'),
            lambda: overlay(B, knee_bounce(B, 1.5), arm_display(B, 1), 'knee-bounce-1.5hz-with-arm-display-1hz'),
            lambda: overlay(B, side_sway(B, .75), chest_beat(B, 2), 'side-sway-0.75hz-with-chest-beat-2hz')]
    return out


# ---- training variants: more variety for the training tape, never the held-out validation motions -----------------
# Mirror images of every target motion but the four the validation catalogue holds out, and arm moves layered on leg
# moves in pairings other than its six. Derived from a folder of built target motions (motions.json and NPZ files).
HELD_OUT_MIRRORS = ('turn-30dps', 'side-shuffle-0.6s', 'side-rock-reach-0.5hz', 'knuckle-walk-1.2s')
TRAINING_COMBOS = (('march-0.5s', 'arm-display-0.5hz'), ('side-shuffle-0.8s', 'overhead-wave-1hz'),
    ('turn-60dps', 'chest-beat-1hz'), ('forward-walk', 'arm-display-1hz'), ('walk-stop', 'chest-beat-1hz'),
    ('knee-bounce-2hz', 'overhead-wave-2hz'), ('ape-crouch-bounce-1.5hz', 'arm-display-0.5hz'),
    ('side-sway-1.5hz', 'overhead-wave-1hz'), ('quick-crouch-0.6s', 'arm-display-1hz'), ('side-shuffle-0.5s', 'chest-beat-1hz'))


def load(folder, row):
    z = np.load(folder / row['motion'])
    mo = {k: v for k, v in row.items() if k not in ('frames', 'duration_s', 'motion', 'sha256', 'torso_z_m', 'peak_joint_speed_rad_s')}
    mo.update(fps=int(z['fps'][0]), qpos=z['qpos'], qvel=z['qvel'], qacc=z['qacc'],
        corners=z['corners'] if 'corners' in z.files else None, **({'hand_contacts': z['hand_contacts']} if 'hand_contacts' in z.files else {}))
    return mo


def derived(B, folder):
    rows = {r['name']: r for r in json.loads((folder / 'motions.json').read_text())['motions']}
    out = [lambda n=n: mirrored(B, load(folder, rows[n]), f'{n}-mirrored') for n in rows if n not in HELD_OUT_MIRRORS]
    out += [lambda l=l, a=a: overlay(B, load(folder, rows[l]), load(folder, rows[a]), f'{l}-with-{a}') for l, a in TRAINING_COMBOS]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--derive', type=Path, help='a folder of built target motions: write training variants of them '
        '(mirror images and arm-leg combinations, never the validation catalogue\'s) instead of a catalogue')
    parser.add_argument('--catalogue', choices=('agility', 'ape', 'knuckle', 'validation'), default='agility',
        help='ape: step-4 tempos and ape moves; knuckle: moves on feet and knuckles (needs long arms, APE0_ARM_SCALE); '
             'validation: never-trained tempos, mirror images and arm-leg combinations (long-arm settings)')
    args = parser.parse_args()
    B = Body()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    makers = derived(B, args.derive) if args.derive else dict(agility=catalogue, ape=ape_catalogue, knuckle=knuckle_catalogue,
        validation=validation_catalogue)[args.catalogue](B)
    for make in makers:
        motion = make()
        if args.only and motion['name'] not in args.only and motion['family'] not in args.only:
            continue
        path = args.output / f"{motion['name']}.npz"
        corners = motion.pop('corners')
        meta = {k: motion.pop(k) for k in list(motion) if not isinstance(motion[k], np.ndarray)}
        np.savez(path, fps=np.array([meta['fps']]), **motion, **({} if corners is None else dict(corners=corners)))
        if 'hand_contacts' in motion:
            meta['hand_contact_share'] = round(float(motion['hand_contacts'].mean()), 3)
        q = motion['qpos']
        rows.append(dict(**meta, frames=len(q), duration_s=round(len(q) / meta['fps'], 3), motion=path.name, sha256=sha(path),
            torso_z_m=[round(float(q[:, 2].min()), 4), round(float(q[:, 2].max()), 4)],
            peak_joint_speed_rad_s=round(float(np.abs(motion['qvel'][:, 6:]).max()), 2)))
        print(f"{meta['name']:28s} {rows[-1]['duration_s']:6.2f} s  peak joint speed {rows[-1]['peak_joint_speed_rad_s']:5.2f} rad/s")
    (args.output / 'motions.json').write_text(json.dumps(dict(schema='cove50.agility.motions.v1', fps_generated=FPS,
        joint_names=body.JOINTS, motions=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
