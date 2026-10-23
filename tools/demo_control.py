"""Example: drive the same RobotIO interface an eventual hardware adapter will expose.

Run: .venv/bin/python tools/demo_control.py artifacts/validated-wheeled
"""

from pathlib import Path
import argparse
import json
import math
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.io import SimRobot, JointCommand

p = argparse.ArgumentParser()
p.add_argument("run_directory")
args = p.parse_args()
root = Path(args.run_directory)
robot = make_robot(
    RobotConfig.model_validate_json((root / "resolved-config.json").read_text())
)
library = root / (
    "supervisor.dylib" if __import__("sys").platform == "darwin" else "supervisor.so"
)
io = SimRobot(robot, root / "supported.xml", library.resolve())
state = io.reset()
neck = state.names.index("neck_yaw_joint")
for sequence in range(100):
    targets = [0.0] * len(state.names)
    targets[neck] = 0.2 * math.sin(state.time_s * math.pi)
    assert io.command(JointCommand(sequence, tuple(targets)))
    state = io.advance(
        robot.config.control.physics_hz // robot.config.control.command_hz
    )
print(
    json.dumps(
        {
            "time_s": state.time_s,
            "joints": len(state.names),
            "supervisor_status": state.supervisor_status,
        }
    )
)
