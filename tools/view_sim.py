"""Interactive local physics viewer. On macOS invoke with .venv/bin/mjpython."""

import argparse
from pathlib import Path
import time
import mujoco
import mujoco.viewer

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
args = p.parse_args()
m = mujoco.MjModel.from_xml_path(str(args.model))
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.lookat[:] = [0, 0, 0.2]
    viewer.cam.distance = 1
    while viewer.is_running():
        started = time.monotonic()
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(max(0, m.opt.timestep - (time.monotonic() - started)))
