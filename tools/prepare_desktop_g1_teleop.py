"""Patch only the input seam in pinned upstream Isaac Lab record_demos.py."""

import sys
from pathlib import Path


source = Path(sys.argv[1]).read_text()
replacements = {
    "teleop_interface = setup_teleop_device(teleoperation_callbacks, use_isaac_teleop)":
        "from desktop_g1_teleop import DesktopDevice\n    teleop_interface = DesktopDevice(teleoperation_callbacks)",
    "env.reset()\n        teleop_interface.reset()":
        "env.reset()\n        teleop_interface.bind(env)",
    "obv = env.step(actions)":
        "obv = env.step(actions)\n                    teleop_interface.publish(obv[0])",
    "current_recorded_demo_count = 0\n    success_step_count = 0":
        "from isaaclab.managers import ManagerTermBase\n"
        "    if success_term is not None and isinstance(success_term.func, type) and issubclass(success_term.func, ManagerTermBase):\n"
        "        success_term.func = success_term.func(success_term, env)\n"
        "    current_recorded_demo_count = 0\n    success_step_count = 0",
    "camera_feed_session.refresh()\n                    should_reset_recording_instance = False":
        "camera_feed_session.refresh()\n"
        "                    if success_term is not None and hasattr(success_term.func, 'reset'):\n"
        "                        success_term.func.reset()\n"
        "                    should_reset_recording_instance = False",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise SystemExit(f"Expected one upstream insertion point, found {source.count(old)}: {old}")
    source = source.replace(old, new)
Path(sys.argv[2]).write_text(source)
