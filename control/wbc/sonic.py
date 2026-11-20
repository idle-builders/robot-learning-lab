"""GEAR-SONIC as the starting whole-body controller for Cove-50 R3.

NVIDIA GEAR-SONIC (GR00T-WholeBodyControl default release, 2026-02-19, NVIDIA
Open Model License) is a Unitree G1 motion tracker trained on large retargeted
human-motion sets. Its g1-mode encoder (4.2 M parameters) turns 10 future
reference frames 0.1 s apart (joint positions, joint velocities, anchor
orientation) into a 64-D FSQ token. A 10.2 M-parameter SiLU MLP decoder turns
token + 10 frames of proprioception into 29 joint-position actions at 50 Hz.

This module rebuilds that network in PyTorch from the released ONNX graphs
(protobuf initializers; no pickle), checks it on G1 itself, maps G1 joints onto
R3 joints and supplies the R3 actor, observations and action for fine-tuning:

    .venv-control/bin/python control/wbc/sonic.py fetch --dir .cache/sonic-base
    .venv-control/bin/python .../sonic.py check-onnx --dir .cache/sonic-base
    .venv-control/bin/python .../sonic.py map
    infra/lab storage run sonic-g1-check -- .venv-control/bin/python .../sonic.py g1-check --dir .cache/sonic-base --out '{evidence}'
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import urllib.request
import zlib

import mujoco
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = 'https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/'
FILES = {
    'model_encoder.onnx': ('013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3', 50100513),
    'model_decoder.onnx': ('c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed', 40900688),
    'sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl':
        ('005aaba3906fa6b99a8b4e89e9d01845d90c5699abf0b5072cc07b099e894f2b', 349988),
}
SAMPLE = 'sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl'

# G1 29-DOF in MuJoCo order; the policy uses Isaac Lab order
# (gear_sonic_deploy policy_parameters.hpp isaaclab_to_mujoco).
G1_MUJOCO = tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
        ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')) + \
    ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + \
    tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
        ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
ISAACLAB_TO_MUJOCO = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
G1_ISAAC = tuple(n for _, n in sorted(zip(ISAACLAB_TO_MUJOCO, G1_MUJOCO)))
DEFAULT = dict(zip(G1_MUJOCO, (-.312, 0, 0, .669, -.363, 0, -.312, 0, 0, .669, -.363, 0, 0, 0, 0,
    .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0)))
# SONIC/BeyondMimic G1 actuators: (armature kg m^2, effort N m); kp = a w^2,
# kd = 2 zeta a w with w = 10 Hz, zeta = 2; doubled motors double the armature.
_MOTOR = dict(m7520_22=(.025101925, 139.), m7520_14=(.010177520, 88.), m5020=(.003609725, 25.), m4010=(.00425, 5.),
    m5020x2=(2 * .003609725, 50.))
_KIND = dict(hip_pitch='m7520_22', hip_roll='m7520_22', hip_yaw='m7520_14', knee='m7520_22', ankle_pitch='m5020x2',
    ankle_roll='m5020x2', waist_yaw='m7520_14', waist_roll='m5020x2', waist_pitch='m5020x2', shoulder_pitch='m5020',
    shoulder_roll='m5020', shoulder_yaw='m5020', elbow='m5020', wrist_roll='m5020', wrist_pitch='m4010', wrist_yaw='m4010')
_W = 10 * 2 * np.pi


def motor(name):
    kind = _KIND[name.replace('left_', '').replace('right_', '').replace('_joint', '')]
    armature, effort = _MOTOR[kind]
    kp, kd = armature * _W ** 2, 2 * 2. * armature * _W
    return dict(armature=armature, effort=effort, kp=kp, kd=kd, action_scale=.25 * effort / kp)


# R3 -> G1 joint map, q_G1 = sign * q_R3 + offset. R3 "R_" links sit at +y
# with the robot facing +x (frames.json: CAD side names), i.e. on the
# anatomical left, so they map to G1 left_*. Signs come from world joint axes
# at the neutral pose; G1's neutral elbow is ~90 deg bent and its straight
# elbow is 1.278 rad, while R3's neutral elbow is straight (`map` re-derives).
ELBOW_OFFSET = 1.278
MAP = (
    ('R_hip_pitch', 'left_hip_pitch_joint', 1, 0.), ('R_hip_roll', 'left_hip_roll_joint', -1, 0.),
    ('R_hip_yaw', 'left_hip_yaw_joint', 1, 0.), ('R_knee', 'left_knee_joint', 1, 0.),
    ('R_ankle_pitch', 'left_ankle_pitch_joint', 1, 0.), ('R_ankle_roll', 'left_ankle_roll_joint', -1, 0.),
    ('L_hip_pitch', 'right_hip_pitch_joint', 1, 0.), ('L_hip_roll', 'right_hip_roll_joint', 1, 0.),
    ('L_hip_yaw', 'right_hip_yaw_joint', -1, 0.), ('L_knee', 'right_knee_joint', 1, 0.),
    ('L_ankle_pitch', 'right_ankle_pitch_joint', 1, 0.), ('L_ankle_roll', 'right_ankle_roll_joint', 1, 0.),
    ('R_shoulder_pitch', 'left_shoulder_pitch_joint', -1, 0.), ('R_shoulder_roll', 'left_shoulder_roll_joint', 1, 0.),
    ('R_shoulder_yaw', 'left_shoulder_yaw_joint', 1, 0.), ('R_elbow', 'left_elbow_joint', -1, ELBOW_OFFSET),
    ('L_shoulder_pitch', 'right_shoulder_pitch_joint', -1, 0.), ('L_shoulder_roll', 'right_shoulder_roll_joint', -1, 0.),
    ('L_shoulder_yaw', 'right_shoulder_yaw_joint', -1, 0.), ('L_elbow', 'right_elbow_joint', -1, ELBOW_OFFSET),
)
UNMAPPED_R3 = ('neck_yaw', 'neck_pitch')  # no G1 counterpart: new zero-initialized outputs
REFERENCE_DIM, PROPRIO_DIM = 640, 930


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(directory):
    for name, (digest, size) in FILES.items():
        path = Path(directory) / name
        if not path.exists() or sha(path) != digest:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SOURCE + name, path)
        if (sha(path), path.stat().st_size) != (digest, size):
            raise ValueError(f'{name} differs from the pinned release')
    return {k: v[0] for k, v in FILES.items()}


def _mlp(dims):
    layers = [m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.SiLU())]
    return nn.Sequential(*layers[:-1])


class SonicG1(nn.Module):
    """g1-mode encoder + FSQ token + g1_dyn decoder of the default SONIC release."""

    def __init__(self):
        super().__init__()
        self.encoder = _mlp((REFERENCE_DIM, 2048, 1024, 512, 512, 64))
        self.decoder = _mlp((64 + PROPRIO_DIM, 2048, 2048, 1024, 1024, 512, 512, 29))
        for name in ('fsq_shift', 'fsq_half', 'fsq_offset', 'fsq_width'):
            self.register_buffer(name, torch.zeros(32))

    def token(self, reference):
        z = torch.tanh(self.encoder(reference).view(-1, 2, 32) + self.fsq_shift) * self.fsq_half - self.fsq_offset
        z = z + (torch.round(z) - z).detach()  # straight-through, as exported
        return (z / self.fsq_width).reshape(-1, 64)

    def forward(self, reference, proprio):
        return self.decoder(torch.cat((self.token(reference), proprio), -1))


def load(directory, pretrained=True):
    """SonicG1 with the released weights, read from the ONNX initializers.
    pretrained=False keeps random layers but the release FSQ levels (its
    quantizer constants are structure, not learned weights)."""
    import onnx
    from onnx import numpy_helper
    net = SonicG1()
    graph = onnx.load(str(Path(directory) / 'model_encoder.onnx')).graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for i, layer in enumerate(m for m in net.encoder if isinstance(m, nn.Linear) and pretrained):
        layer.weight.data.copy_(torch.from_numpy(init[f'module.encoders.g1.module.{2 * i}.weight']))
        layer.bias.data.copy_(torch.from_numpy(init[f'module.encoders.g1.module.{2 * i}.bias']))
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in graph.node if n.op_type == 'Constant'}
    const.update(init)
    for node in graph.node:  # '/quantizer/' (no suffix) is the g1 encoder's FSQ
        key = dict(Add='fsq_shift', Mul='fsq_half', Sub='fsq_offset', Div='fsq_width').get(node.op_type)
        if key and node.output[0] == f'/quantizer/{node.op_type}_output_0':
            value = next(const[i] for i in node.input if i in const and const[i].shape == (32,))
            getattr(net, key).copy_(torch.from_numpy(value))
    if not pretrained:
        return net
    graph = onnx.load(str(Path(directory) / 'model_decoder.onnx')).graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    weights = [init[n.input[1]] for n in graph.node if n.op_type == 'MatMul']
    for i, layer in enumerate(m for m in net.decoder if isinstance(m, nn.Linear)):
        layer.weight.data.copy_(torch.from_numpy(weights[i].T))
        layer.bias.data.copy_(torch.from_numpy(init[f'module.decoders.g1_dyn.module.{2 * i}.bias']))
    return net


def reference_input(pos, vel, ori6):
    """Encoder input [B,640] from future frames [B,10,29], [B,10,29], [B,10,6].

    As in SONIC training (command_multi_future): all positions then all
    velocities, reshaped to 10 rows of 58, each row joined with one 6-D
    anchor-orientation frame.
    """
    b = pos.shape[0]
    command = torch.cat((pos.reshape(b, -1), vel.reshape(b, -1)), -1).view(b, 10, 58)
    return torch.cat((command, ori6), -1).reshape(b, REFERENCE_DIM)


def ori6(matrix):
    """First two rotation-matrix columns, row-major (SONIC motion_anchor_ori_b)."""
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def check_onnx(directory, seed=0):
    """Torch rebuild vs the ONNX graphs on random inputs (onnx reference evaluator)."""
    from onnx.reference import ReferenceEvaluator
    net = load(directory).double()
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(1, 1762)).astype(np.float32)
    obs[0, :4] = 0  # encoder mode g1, then three padding zeros
    token = ReferenceEvaluator(str(Path(directory) / 'model_encoder.onnx')).run(None, {'obs_dict': obs})[0]
    # The graph drops obs[0] (mode), then reads command [3:583] and the
    # multi-frame anchor orientation [600:660] of the remainder.
    command, anchor = obs[:, 4:584].reshape(1, 10, 58), obs[:, 601:661].reshape(1, 10, 6)
    with torch.no_grad():
        mine = net.token(torch.from_numpy(np.concatenate((command, anchor), -1).reshape(1, 640)).double())
    prop = rng.normal(size=(1, 994)).astype(np.float32)
    action = ReferenceEvaluator(str(Path(directory) / 'model_decoder.onnx')).run(None, {'obs_dict': prop})[0]
    with torch.no_grad():
        p = torch.from_numpy(prop).double()
        mine_action = net.decoder(p).numpy()
    return dict(token_max_abs_diff=float(np.abs(token - mine.numpy()).max()), token_levels=int(np.unique(np.round(token * 16)).size),
        action_max_abs_diff=float(np.abs(action - mine_action).max()), action_rms=float(np.sqrt((action ** 2).mean())))


class _ArrayWrapper:
    """Stand-in for joblib's NumpyArrayWrapper; array bytes follow it in the stream."""

    def __setstate__(self, state):
        self.__dict__.update(state)


class _MotionUnpickler(pickle._Unpickler):
    """Restricted reader for the joblib motion file: only numpy arrays and dtypes."""

    ALLOWED = {('joblib.numpy_pickle', 'NumpyArrayWrapper'): _ArrayWrapper, ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'refused global {module}.{name}')
        return self.ALLOWED[module, name]

    def load_build(self):
        super().load_build()
        if isinstance(self.stack[-1], _ArrayWrapper):
            w = self.stack.pop()
            if getattr(w, 'numpy_array_alignment_bytes', None) is not None:
                pad = self.read(1)[0]
                self.read(pad)
            count = int(np.prod(w.shape))
            array = np.frombuffer(self.read(count * w.dtype.itemsize), dtype=w.dtype, count=count).copy()
            self.stack.append(array.reshape(w.shape[::-1]).T if w.order == 'F' else array.reshape(w.shape))

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.BUILD[0]] = load_build


def read_motion(path):
    import io
    raw = zlib.decompress(Path(path).read_bytes())
    data = _MotionUnpickler(io.BytesIO(raw)).load()
    return next(iter(data.values())) if len(data) == 1 else data


def g1_model():
    """mjlab's G1 MJCF with SONIC's PD actuators, armature and 200 Hz physics, plus a floor."""
    import mjlab
    spec = mujoco.MjSpec.from_file(str(Path(mjlab.__file__).parent / 'asset_zoo/robots/unitree_g1/xmls/g1.xml'))
    spec.option.timestep, spec.option.integrator = .005, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], friction=[1, .005, .0001])
    for name in G1_MUJOCO:
        m = motor(name)
        spec.joint(name).armature = m['armature']
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[m['kp']] + [0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -m['kp'], -m['kd']] + [0] * 7,
            forcelimited=True, forcerange=[-m['effort'], m['effort']])
    return spec.compile()


def _quat_matrix(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def g1_check(directory, out, seconds=None):
    """Sim-to-sim of the released weights on G1 against the SONIC sample walk."""
    motion = read_motion(Path(directory) / SAMPLE)
    fps = float(motion['fps'])
    dof, root, rot = (np.asarray(motion[k], dtype=np.float64) for k in ('dof', 'root_trans_offset', 'root_rot'))
    # Resample to 50 Hz. root_rot is scipy xyzw in this format; MuJoCo wants wxyz.
    t = np.arange(0, (len(dof) - 1) / fps, .02)
    if seconds:
        t = t[t < seconds]
    src = np.arange(len(dof)) / fps
    lerp = lambda x: np.stack([np.interp(t, src, x[:, i]) for i in range(x.shape[1])], 1)
    dof, root = lerp(dof), lerp(root)
    quat = lerp(rot[:, [3, 0, 1, 2]])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dof_vel = np.gradient(dof, .02, axis=0)
    isaac = np.array([G1_MUJOCO.index(n) for n in G1_ISAAC])
    default = np.array([DEFAULT[n] for n in G1_ISAAC])
    scale = np.array([motor(n)['action_scale'] for n in G1_ISAAC])
    net = load(directory).eval()
    model = g1_model()
    report = dict(schema='cove50.wbc.sonic-g1-check.v1', motion=SAMPLE, motion_fps=fps, control_hz=50,
        physics_dt_s=model.opt.timestep, seconds=float(len(t) * .02), sha256=fetch(directory), rows=[])
    for mode in ('sonic', 'playback'):
        data = mujoco.MjData(model)
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[0], quat[0], dof[0]
        data.qvel[6:] = dof_vel[0]
        mujoco.mj_forward(model, data)
        history, action = None, np.zeros(29)
        log = dict(pelvis_z=[], tilt_rad=[], joint_error_rad=[], root_xy_error_m=[], action_abs=[])
        fall = None
        for step in range(len(t) - 1):
            R = _quat_matrix(data.qpos[3:7])
            frame = np.concatenate((data.qvel[3:6], data.qpos[7:][isaac] - default, data.qvel[6:][isaac], action,
                R.T @ np.array([0, 0, -1.])))
            history = np.repeat(frame[None], 10, 0) if history is None else np.concatenate((history[1:], frame[None]))
            parts = (slice(0, 3), slice(3, 32), slice(32, 61), slice(61, 90), slice(90, 93))
            proprio = np.concatenate([history[:, p].reshape(-1) for p in parts])
            future = np.minimum(step + 5 * np.arange(10), len(t) - 1)
            ori = np.stack([R.T @ _quat_matrix(quat[f]) for f in future])
            ref = reference_input(torch.tensor(dof[future][:, isaac])[None], torch.tensor(dof_vel[future][:, isaac])[None],
                ori6(torch.tensor(ori))[None]).float()
            if mode == 'sonic':
                with torch.no_grad():
                    action = net(ref, torch.tensor(proprio, dtype=torch.float32)[None])[0].double().numpy()
                target = default + scale * action
            else:
                target = dof[step][isaac]
            data.ctrl[isaac] = target
            for _ in range(4):
                mujoco.mj_step(model, data)
            tilt = float(np.arccos(np.clip(_quat_matrix(data.qpos[3:7])[2, 2], -1, 1)))  # pelvis z axis vs world z
            log['pelvis_z'].append(float(data.qpos[2]))
            log['tilt_rad'].append(tilt)
            log['joint_error_rad'].append(float(np.abs(data.qpos[7:] - dof[step + 1]).mean()))
            log['root_xy_error_m'].append(float(np.linalg.norm(data.qpos[:2] - root[step + 1, :2])))
            log['action_abs'].append(float(np.abs(action).mean()))
            if fall is None and (data.qpos[2] < .45 or tilt > 1.):
                fall = (step + 1) * .02
        trace = {k: np.array(v) for k, v in log.items()}
        np.savez_compressed(Path(out) / f'g1-{mode}.npz', **trace)
        upright = slice(0, int(fall / .02) if fall else None)
        report['rows'].append(dict(mode=mode, first_fall_s=fall, steps=len(t) - 1,
            joint_error_rad_mean_while_upright=float(trace['joint_error_rad'][upright].mean()),
            root_xy_error_m_final=float(trace['root_xy_error_m'][-1]), pelvis_z_min_m=float(trace['pelvis_z'].min()),
            tilt_rad_max=float(trace['tilt_rad'].max()), reference_root_travel_m=float(np.linalg.norm(root[-1, :2] - root[0, :2]))))
        print(json.dumps(report['rows'][-1]), flush=True)
    (Path(out) / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def derive_map():
    """Re-derive MAP signs from world joint axes and the elbow offset from geometry."""
    import mjlab
    spec = mujoco.MjSpec.from_file(str(ROOT / 'artifacts/robot/model/timestep/cove50-rigid.xml'))
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    r3 = spec.compile()
    g1 = mujoco.MjModel.from_xml_path(str(Path(mjlab.__file__).parent / 'asset_zoo/robots/unitree_g1/xmls/g1.xml'))
    rows, axes = [], []
    for model in (r3, g1):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        axes.append({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): (data.xaxis[j].copy(), data.xanchor[j].copy())
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE})
    for r3_name, g1_name, sign, offset in MAP:
        a3, p3 = axes[0][r3_name]
        ag, pg = axes[1][g1_name]
        derived = int(np.sign(a3 @ ag))
        jr = mujoco.mj_name2id(r3, mujoco.mjtObj.mjOBJ_JOINT, r3_name)
        lo, hi = r3.jnt_range[jr] if r3.jnt_limited[jr] else (-np.inf, np.inf)
        g1_range = sorted((sign * lo + offset, sign * hi + offset))
        rows.append(dict(r3=r3_name, g1=g1_name, sign=sign, derived_sign=derived, axis_cos=round(float(a3 @ ag), 3),
            same_side=bool(np.sign(p3[1]) == np.sign(pg[1])), offset=offset,
            r3_range_in_g1=[round(float(v), 3) for v in g1_range], g1_default=DEFAULT[g1_name],
            g1_default_in_r3=round(float(sign * (DEFAULT[g1_name] - offset)), 3)))
    bad = [r for r in rows if r['sign'] != r['derived_sign'] or not r['same_side']]
    if bad:
        raise ValueError(f'MAP disagrees with the models: {bad}')
    return rows


# ---- R3 fine-tuning task: actor, observations, action -------------------------------------------

def _tables(env):
    """Index/sign/offset tensors for this env, cached on it.

    Motion and action rows follow the entity's MJCF joint order; the R3
    feedback packet follows the actuator contract (legs grouped by hip,
    then knee/ankle), so measured values get their own index.
    """
    cached = getattr(env, '_sonic_tables', None)
    if cached is None:
        import tracking
        names = list(env.scene['robot'].joint_names)
        packet = list(tracking.runtime().get_runtime(env).names)
        r3 = torch.tensor([names.index(r) for r, *_ in MAP], device=env.device)
        measured = torch.tensor([packet.index(r) for r, *_ in MAP], device=env.device)
        g1 = torch.tensor([G1_ISAAC.index(g) for _, g, *_ in MAP], device=env.device)
        sign = torch.tensor([float(s) for *_, s, _ in MAP], device=env.device)
        offset = torch.tensor([o for *_, o in MAP], device=env.device)
        default = torch.tensor([DEFAULT[g] for g in G1_ISAAC], device=env.device)
        cached = env._sonic_tables = dict(r3=r3, measured=measured, g1=g1, sign=sign, offset=offset, default=default)
    return cached


def _scatter(env, values):
    """[N,20] mapped values -> [N,29] G1 Isaac-order vector, zeros where R3 has no joint."""
    out = torch.zeros(values.shape[:-1] + (29,), device=values.device)
    out[..., _tables(env)['g1']] = values
    return out


def obs_reference(env):
    """Encoder input: 10 R3 reference frames 0.1 s apart, in G1 joint language."""
    from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul
    tab, cmd = _tables(env), env.command_manager.get_term('motion')
    m = cmd.motion
    idx = (cmd.time_steps[:, None] + 5 * torch.arange(10, device=env.device)).clamp(max=m.time_step_total - 1)
    pos = _scatter(env, tab['sign'] * m.joint_pos[idx][..., tab['r3']] + tab['offset'])
    vel = _scatter(env, tab['sign'] * m.joint_vel[idx][..., tab['r3']])
    ref = m.body_quat_w[idx, cmd.motion_anchor_body_index]
    robot = quat_inv(cmd.robot_anchor_quat_w)[:, None].expand_as(ref)
    return reference_input(pos, vel, ori6(matrix_from_quat(quat_mul(robot, ref))))


def obs_joint_pos(env):
    from tracking import measured
    tab = _tables(env)
    q = tab['sign'] * measured(env, 0, 22)[:, tab['measured']] + tab['offset']
    return _scatter(env, q - tab['default'][tab['g1']])


def obs_joint_vel(env):
    from tracking import measured
    tab = _tables(env)
    return _scatter(env, tab['sign'] * measured(env, 22, 44)[:, tab['measured']])


def obs_last_action(env):
    tab = _tables(env)
    return _scatter(env, env.action_manager.action[:, tab['r3']])


def action_terms():
    """Per-joint scale/offset so that R3 target = sign * (G1 default + G1 scale * a - offset)."""
    scale, offset = {}, {}
    for r3, g1, sign, o in MAP:
        scale[r3] = sign * motor(g1)['action_scale']
        offset[r3] = sign * (DEFAULT[g1] - o)
    for name in UNMAPPED_R3:
        scale[name], offset[name] = .5, 0.
    return scale, offset


def env_cfg(motion_file, *, play=False, num_envs=1, reward_widths='g1', **plant):
    """The R3 tracking task with SONIC's observation layout and G1-unit actions."""
    import tracking
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    cfg = tracking.env_cfg(motion_file, play=play, num_envs=num_envs, reward_widths=reward_widths, **plant)
    scale, offset = action_terms()
    cfg.actions = {'joint_pos': tracking.R3PositionActionCfg(entity_name='robot', actuator_names=('.*',),
        scale=scale, offset=offset, use_default_offset=False)}
    hist = dict(history_length=10, flatten_history_dim=True)
    terms = dict(
        reference=ObservationTermCfg(func=obs_reference),
        base_ang_vel=ObservationTermCfg(func=tracking.measured, params=dict(start=44, stop=47), **hist),
        joint_pos=ObservationTermCfg(func=obs_joint_pos, **hist),
        joint_vel=ObservationTermCfg(func=obs_joint_vel, **hist),
        actions=ObservationTermCfg(func=obs_last_action, **hist),
        gravity=ObservationTermCfg(func=tracking.measured, params=dict(start=47, stop=50), **hist))
    cfg.observations['actor'] = ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=False)
    return cfg


# SONIC release PPO: actor 2e-5 and critic 1e-3 learning rates, initial
# action noise 0.05, entropy 0.01. The R3 critic is new and learns alone for
# the first CRITIC_WARMUP iterations so its advantages do not steer the
# pretrained actor at random.
ACTOR_LR, CRITIC_LR, INIT_STD, CRITIC_WARMUP = 2e-5, 1e-3, .05, 10
# 'release' is that recipe as run in sonic-finetune-r3: the entropy bonus
# tripled the noise and each round moved the policy far past the PPO clip.
# 'trust-region' fixes the noise at INIT_STD, drops the entropy bonus, freezes
# the reference encoder and adapts the decoder rate to DESIRED_KL per
# mini-batch from TRUST_ACTOR_LR; the critic keeps CRITIC_LR. On R3
# observations one Adam step at 1e-7 moves the warm decoder by KL 4e-4 at this
# noise (random layers: 1e-8); with the encoder trainable, FSQ code flips push
# single samples to KL 1.
RECIPES, DESIRED_KL, TRUST_ACTOR_LR = ('release', 'trust-region'), .01, 1e-7


def ppo_cfg(recipe='release', **overrides):
    import tracking
    cfg = tracking.ppo_cfg(experiment_name='cove50_sonic', **overrides)
    cfg.algorithm.entropy_coef = .01 if recipe == 'release' else 0.
    cfg.algorithm.schedule, cfg.algorithm.learning_rate, cfg.algorithm.desired_kl = 'fixed', CRITIC_LR, DESIRED_KL
    return cfg


class TrustRegionAdam(torch.optim.Adam):
    """Each mini-batch rsl-rl's adaptive schedule multiplies alg.learning_rate by
    1.5, 1/1.5 or 1 from that batch's KL, clamped to [1e-5, 1e-2], and writes it
    into every group. The actor rate takes the same factor unclamped, then
    alg.learning_rate is re-centred so the clamp never binds; the critic keeps
    CRITIC_LR."""

    CENTRE = 1e-3

    def __init__(self, alg, groups):
        super().__init__(groups)
        self.alg, self.actor_lr = alg, 0.
        alg.learning_rate = self.CENTRE

    def step(self, closure=None):
        self.actor_lr *= self.alg.learning_rate / self.CENTRE
        self.alg.learning_rate = self.CENTRE
        actor, critic = self.param_groups
        actor['lr'], critic['lr'] = self.actor_lr, CRITIC_LR
        return super().step(closure)


def split_optimizer(alg, recipe='release'):
    """Separate actor/critic rates (rsl-rl builds one shared Adam group); the
    actor rate is zero until release_actor() ends the critic warmup."""
    groups = [dict(params=list(alg.actor.parameters()), lr=0.), dict(params=list(alg.critic.parameters()), lr=CRITIC_LR)]
    if recipe == 'release':
        alg.optimizer = torch.optim.Adam(groups)
    else:
        alg.actor.distribution.std_param.requires_grad_(False)
        alg.actor.mlp.net.encoder.requires_grad_(False)
        alg.optimizer = TrustRegionAdam(alg, groups)


def release_actor(alg, recipe='release'):
    if recipe == 'release':
        alg.optimizer.param_groups[0]['lr'] = ACTOR_LR
    else:
        alg.optimizer.actor_lr, alg.schedule = TRUST_ACTOR_LR, 'adaptive'


def rows_for(joint_names):
    """Actor output row j -> G1 Isaac decoder row (or -1 for a new R3-only output)."""
    lookup = {r: G1_ISAAC.index(g) for r, g, *_ in MAP}
    return [lookup.get(n, -1) for n in joint_names]


class SonicNet(nn.Module):
    """SONIC encoder + decoder with the decoder head re-indexed to R3 actuators."""

    def __init__(self, rows, weights=None, pretrained=True):
        super().__init__()
        net = load(weights, pretrained) if weights else SonicG1()
        last = net.decoder[-1]
        head = nn.Linear(last.in_features, len(rows))
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        for j, g in enumerate(rows):
            if g >= 0:
                head.weight.data[j] = last.weight.data[g]
                head.bias.data[j] = last.bias.data[g]
        net.decoder[-1] = head
        self.net = net

    def forward(self, x):
        return self.net(x[:, :REFERENCE_DIM], x[:, REFERENCE_DIM:])


def model_class():
    """rsl-rl actor class (built lazily so the G1 check needs no rsl-rl)."""
    from rsl_rl.models import MLPModel
    from rsl_rl.modules.distribution import GaussianDistribution

    class SonicModel(MLPModel):
        def __init__(self, obs, obs_groups, obs_set, output_dim, rows, weights=None, pretrained=True, init_std=.05, **_):
            nn.Module.__init__(self)
            self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
            if self.obs_dim != REFERENCE_DIM + PROPRIO_DIM:
                raise ValueError(f'SONIC actor expects {REFERENCE_DIM + PROPRIO_DIM} inputs, got {self.obs_dim}')
            self.obs_normalization, self.obs_normalizer = False, nn.Identity()  # SONIC actor runs unnormalized
            self.distribution = GaussianDistribution(output_dim, init_std=init_std, std_type='scalar')
            self.mlp = SonicNet(rows, weights, pretrained)

    return SonicModel


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('action', choices=('fetch', 'check-onnx', 'g1-check', 'map'))
    parser.add_argument('--dir', type=Path, default=ROOT / '.cache/sonic-base')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    if args.action == 'fetch':
        result = fetch(args.dir)
    elif args.action == 'check-onnx':
        result = check_onnx(args.dir)
    elif args.action == 'map':
        result = derive_map()
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        result = g1_check(args.dir, args.out, args.seconds)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
