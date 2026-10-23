"""Save aligned metric depth for a previously rendered Habitat camera trace."""

import argparse
import json
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis

from render_replica_ape_route import make_simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    width, height = trace["camera_resolution_px"]
    eye = trace["camera_height_above_nav_m"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with make_simulator(args.dataset_root, args.scene, width, height, eye) as sim:
        agent = sim.initialize_agent(0)
        for frame in trace["frames"]:
            state = habitat_sim.AgentState()
            state.position = np.asarray(frame["position_m"])
            state.rotation = quat_from_angle_axis(
                frame["yaw_rad"], np.array([0.0, 1.0, 0.0])
            )
            agent.set_state(state)
            depth = np.asarray(sim.get_sensor_observations()["depth"])
            valid = np.isfinite(depth) & (depth > 0)
            fraction = float(valid.mean())
            expected = frame["valid_depth_fraction"]
            if abs(fraction - expected) > 0.005:
                raise RuntimeError(
                    f"frame {frame['frame']}: depth coverage changed {expected:.3f} to {fraction:.3f}"
                )
            filename = "depth_%04d.png" % frame["frame"]
            depth_mm = np.where(valid, np.clip(np.rint(depth * 1000), 0, 65535), 0)
            Image.fromarray(depth_mm.astype(np.uint16)).save(args.output / filename)
            rows.append({"frame": frame["frame"], "depth_mm_png": filename,
                         "valid_depth_fraction": fraction})
    result = {
        "schema": "ape.metric-depth-replay.v1",
        "source_trace": str(args.trace),
        "source_scene": args.scene,
        "encoding": "uint16_millimetres_zero_invalid",
        "camera_resolution_px": [width, height],
        "frames": rows,
    }
    (args.output / "depth_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(len(rows), args.output)


if __name__ == "__main__":
    main()
