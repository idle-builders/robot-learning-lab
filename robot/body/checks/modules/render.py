"""render: the owner six-side case drawing of the manifest (owner_case_views.py), run in a subprocess. Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register


def check_render(manifest, out_dir, timeout):
    t0 = time.time()
    code = ("import sys; from pathlib import Path; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import owner_case_views as V; V.OUT = Path(%r); V.main()"
            % (str(ROOT / "robot/body/hip-shell"), str(ROOT / "robot/body/hip-shell/sections"), str(out_dir)))
    mp = Path(manifest).resolve()
    env = dict(os.environ, HIP_MANIFEST=str(mp.relative_to(ROOT) if mp.is_relative_to(ROOT) else mp), MPLBACKEND="Agg")
    try:
        r = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=timeout)
        res = dict(png=str((Path(out_dir) / "case-all-sides.png").relative_to(ROOT)), returncode=r.returncode,
                   error=r.stderr[-400:] if r.returncode else None)
    except subprocess.TimeoutExpired:
        res = dict(error=f"owner_case_views exceeded the {timeout} s limit")
    tick("render (six sides)", t0)
    return res


@register("render", lambda ctx: bool(ctx.scope))
def run(ctx):
    """The owner six-side case drawing of the whole manifest (not scoped to parts)."""
    r = check_render(ctx.manifest, ctx.out_dir, ctx.render_timeout)
    return dict(per_part={}, worst_cases=[], bad_events=dict(render_errors=1 if r.get("error") else 0),
                evidence=[r["png"]] if r.get("png") and not r.get("returncode") else [], notes=r.get("error") or "owner_case_views.py")
