"""Check whether a stopped Replica approach has an observed-free bypass.

This is a route information probe, not an executed or body-qualified path.
Only source-mesh two-height rays become actor map evidence; the saved target
position is the actor's RGB-D estimate, not the evaluator's placement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realistic_people"))
from fan_view_route import FanViewRoute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-mesh", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    scenario = json.loads((args.run / "scenario.json").read_text())
    walk = json.loads((args.run / "walk/trace.json").read_text())
    stopped = walk["actions"][-1]
    assert stopped["decision"] == "hold"
    pose = np.asarray(stopped["actor_xz_m"], float)
    yaw = float(stopped["actor_yaw_deg"])
    target = np.asarray(stopped["observed_target_xz_m"], float)
    base_y = float(scenario["start_nav_y_m"])

    mesh = trimesh.load(args.converted_mesh, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected one converted source mesh")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    angles = np.arange(-90., 91., 5.)
    route = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1),
                         standoff_min_m=.55, standoff_max_m=.65,
                         max_route_m=3., max_ray_m=3.)
    sweeps = []
    plans = []
    for delta in (0., -45., 45., -90., 90., 180.):
        heading = yaw + delta
        rays = np.asarray([[pose[0], base_y + height, pose[1],
                            math.sin(math.radians(heading + angle)), 0.,
                            math.cos(math.radians(heading + angle))]
                           for height in (.1, .5) for angle in angles], np.float32)
        ranges = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(2, -1)
        sweeps.append({"turn_from_stopped_deg": delta,
                       "nearest_low_m": float(np.min(ranges[0])),
                       "nearest_mid_m": float(np.min(ranges[1])),
                       "finite_beams": int(np.isfinite(ranges).sum())})
        decision = route.plan(ranges, angles, pose, heading, target, 0.)
        plans.append({"after_sweep_deg": delta,
                      "intent": decision["intent"],
                      "reason": decision.get("reason"),
                      "reachable_cells": decision.get("reachable_cells"),
                      "goal_xy_m": decision.get("goal_xy_m"),
                      "next_waypoint_xy_m": decision.get("next_waypoint_xy_m"),
                      "yaw_delta_deg": decision.get("yaw_delta_deg"),
                      "path_cells": decision.get("path_cells")})
    result = {"schema": "ape.replica-route-recovery-probe.v1",
              "run": str(args.run), "pose_xz_m": pose.tolist(),
              "observed_target_xz_m": target.tolist(),
              "assumed_base_y_m": base_y,
              "method": "Stationary six-heading two-height ideal source range sweep; no motion executed",
              "route_clearance_m": .17,
              "sweeps": sweeps, "plans": plans}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"plans": plans}))


if __name__ == "__main__":
    main()
