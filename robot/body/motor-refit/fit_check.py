"""Shim: the body helpers (Body, load_mesh, src, to_mf, Prepped, measure, is_fastener, pose_set, ...) live in
robot/body/checks/bodylib.py. This module only re-exports them for scripts that still `import fit_check as F` (the
hip-shell kit, ape-proportions, fastening). It has no check logic; the checks are `robot/body/checks/run.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks.bodylib import *  # noqa: E402,F401,F403
from checks.bodylib import (FASTENER_NAME, FASTENER_ROLES, Body, Prepped, apply, is_fastener, load_mesh, measure,  # noqa: E402,F401
                            pose_set, rot_about, samples, src, to_mf)
