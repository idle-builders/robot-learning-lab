"""Shared identities and SI frame convention for the selected CAD transfer."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('COVE50_MODEL_DIR', ROOT / 'artifacts/robot/model/build'))
DATA = ROOT / 'data/model-transfer-r1-20261026'
CAD = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'
VERSION = 'r1-20261026'
RESULT = 'cove50-learning-model-r1-20261026'
# Proper rotation, never a reflection. CAD forward is -Y. Preserve source R/L
# names even though source R is at positive REP y; document this discrepancy.
R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
ROOT_ORIGIN_CAD_M = np.array([0., 0., .300])

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

def ref(p):
    p = Path(p)
    return dict(path=rel(p), sha256=sha(p), bytes=p.stat().st_size)

def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

def vec(v):
    return ' '.join(f'{float(x):.12g}' for x in np.ravel(v))

def origins(manifest):
    return {'torso': R @ ROOT_ORIGIN_CAD_M,
            **{j['child']: R @ (np.array(j['origin_world_mm']) / 1000)
               for j in manifest['joints']}}

def frame_contract():
    return dict(units='m, kg, s, rad, N, N m',
        transform='p_model = R_model_CAD @ p_CAD_m; all link axes are aligned at neutral',
        R_model_CAD=R.tolist(), determinant=float(np.linalg.det(R)),
        base_link='torso; CAD origin [0,0,300] mm; model x=-CAD Y, y=CAD X, z=CAD Z',
        source_side_warning='CAD calls positive X R and negative X L while toe/camera forward is -Y. A reflection would violate right-handed physics. Retain R/L as source identifiers, not REP anatomical side assertions. Neutral R links therefore lie at positive model y.',
        optical='z forward, x image right, y image down; physical pixel roll uncalibrated',
        quaternion='MuJoCo wxyz; ROS xyzw adapters explicitly convert',
        pose_equation='p_W = T_W_C(t_capture) @ p_C; never use inference completion pose',
        global_frames='map -> odom -> base_footprint -> torso are estimator outputs, not fixed ground-truth URDF joints; odom continuous, map corrections may jump',
        selected_CAD=ref(CAD / 'assembly-parts.json'))
