"""Render a measured-height appearance probe in an official Replica real scan.

This is a camera/navmesh screen, not a closed owner task or body qualification.
Run with Habitat-Sim and the official Replica v1 data on a compute machine.
"""

import argparse
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis


def route_points(points, step=0.08):
    coordinates = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    offsets = np.r_[0.0, np.cumsum(segment_lengths)]
    total = float(offsets[-1])
    distances = np.linspace(0, total, max(2, math.ceil(total / step) + 1))
    result = np.column_stack(
        [np.interp(distances, offsets, coordinates[:, axis]) for axis in range(3)]
    )
    return result, total


def make_simulator(dataset_root, scene_id, width, height, eye_height):
    backend = habitat_sim.SimulatorConfiguration()
    dataset_config = dataset_root / "replica.scene_dataset_config.json"
    if dataset_config.is_file():
        backend.scene_dataset_config_file = str(dataset_config)
    backend.scene_id = scene_id
    backend.enable_physics = False

    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = 0.55
    agent.radius = 0.17
    specs = []
    for uuid, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [height, width]
        spec.position = [0.0, eye_height, 0.0]
        spec.hfov = 90.0
        specs.append(spec)
    agent.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def route_heading(positions, index):
    tangent = positions[min(index + 1, len(positions) - 1)] - positions[
        max(index - 1, 0)
    ]
    return math.atan2(-float(tangent[0]), -float(tangent[2]))


def candidate_visibility(sim, points):
    """Reject navmesh corridors that are visually outside the captured surface."""
    positions, _ = route_points(points)
    agent = sim.initialize_agent(0)
    fractions = []
    for index in np.linspace(0, len(positions) - 1, 7).astype(int):
        state = habitat_sim.AgentState()
        state.position = positions[index]
        state.rotation = quat_from_angle_axis(
            route_heading(positions, index), np.array([0.0, 1.0, 0.0])
        )
        agent.set_state(state)
        depth = np.asarray(sim.get_sensor_observations()["depth"])
        fractions.append(float((np.isfinite(depth) & (depth > 0)).mean()))
    return float(np.mean(fractions)), float(np.min(fractions))


def select_route(sim, random_seed, minimum=2.0, maximum=5.0):
    pathfinder = sim.pathfinder
    pathfinder.seed(random_seed)
    best = None
    for _ in range(120):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        if abs(float(start[1] - goal[1])) > 0.2:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if not pathfinder.find_path(path):
            continue
        length = float(path.geodesic_distance)
        if minimum <= length <= maximum:
            mean_visibility, minimum_visibility = candidate_visibility(sim, path.points)
            score = mean_visibility + 0.35 * minimum_visibility - 0.06 * abs(length - 3.0)
            if best is None or score > best[3]:
                best = (path.points, length, mean_visibility, score)
            if minimum_visibility > 0.97 and mean_visibility > 0.99:
                break
    if best is None:
        raise RuntimeError("no connected 2-5 m navigation route was found")
    return best


def render(dataset_root, scene_id, output, seed, eye_height, recompute_navmesh,
           source_url, source_release, scene_label, save_depth):
    output.mkdir(parents=True, exist_ok=True)
    with make_simulator(dataset_root, scene_id, 640, 480, eye_height) as sim:
        if recompute_navmesh:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = 0.55
            settings.agent_radius = 0.17
            settings.cell_height = 0.05
            settings.cell_size = 0.03
            if not sim.recompute_navmesh(sim.pathfinder, settings):
                raise RuntimeError("could not compute a provisional navmesh")
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("Replica navigation mesh is unavailable")
        path_points, geodesic, preview_visibility, _ = select_route(sim, seed)
        positions, polyline = route_points(path_points)
        agent = sim.initialize_agent(0)
        frames = []
        for index, position in enumerate(positions):
            yaw = route_heading(positions, index)
            state = habitat_sim.AgentState()
            state.position = position
            state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(state)
            observation = sim.get_sensor_observations()
            rgb = np.asarray(observation["rgb"])[..., :3]
            depth = np.asarray(observation["depth"])
            image_name = "frame_%04d.png" % index
            Image.fromarray(rgb).save(output / image_name)
            valid = depth[np.isfinite(depth) & (depth > 0)]
            frame = {
                    "frame": index,
                    "image": image_name,
                    "position_m": position.tolist(),
                    "yaw_rad": yaw,
                    "valid_depth_fraction": float(valid.size / depth.size),
                    "median_depth_m": float(np.median(valid)) if valid.size else None,
                    "dark_rgb_fraction": float((rgb.mean(axis=-1) < 10).mean()),
                }
            if save_depth:
                depth_name = "depth_%04d.png" % index
                depth_mm = np.where(np.isfinite(depth) & (depth > 0),
                                    np.clip(np.rint(depth * 1000), 0, 65535), 0)
                Image.fromarray(depth_mm.astype(np.uint16)).save(output / depth_name)
                frame["depth_mm_png"] = depth_name
            frames.append(frame)
        record = {
            "schema": "ape.replica-appearance-route.v1",
            "source": source_url,
            "source_release": source_release,
            "scene_id": scene_label or scene_id,
            "camera_height_above_nav_m": eye_height,
            "camera_resolution_px": [640, 480],
            "camera_hfov_deg": 90.0,
            "route_seed": seed,
            "proxy_radius_m": 0.17,
            "proxy_height_m": 0.55,
            "source_navmesh_geodesic_m": geodesic,
            "rendered_path_polyline_m": polyline,
            "candidate_preview_mean_valid_depth_fraction": preview_visibility,
            "depth_encoding": "uint16_millimetres_zero_invalid" if save_depth else "statistics_only",
            "caveat": "A navmesh camera probe, not a physical ape or owner-task run.",
            "frames": frames,
        }
        (output / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"scene_id": scene_label or scene_id, "frames": len(frames), "metres": polyline}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eye-height", type=float, default=0.42)
    parser.add_argument("--recompute-navmesh", action="store_true")
    parser.add_argument("--source-url", default="https://github.com/facebookresearch/Replica-Dataset")
    parser.add_argument("--source-release", default="v1.0")
    parser.add_argument("--scene-label")
    parser.add_argument("--save-depth", action="store_true")
    args = parser.parse_args()
    render(args.dataset_root, args.scene, args.output, args.seed, args.eye_height,
           args.recompute_navmesh, args.source_url, args.source_release,
           args.scene_label, args.save_depth)


if __name__ == "__main__":
    main()
