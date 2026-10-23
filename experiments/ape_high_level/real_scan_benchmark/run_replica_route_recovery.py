"""Try the unchanged observed-free planner after an item route stops.

This starts from a saved actor stop, uses the saved RGB-D item location, and
executes bounded turns/steps against fresh range and floor observations. It is
a development probe; the 17 cm disk and ideal pose are not a physical ape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedFloorObject
from replica_capture_scene import ReplicaCaptureScene
from independent_range_monitor import IndependentRangeMonitor
import run_real_capture_teach_find as task
from active_scan import Vision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "realistic_people"))
from fan_view_route import FanViewRoute  # noqa: E402

ANGLES = np.arange(-90., 91., 5.)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("renderer", "source-scene", "converted-mesh", "run", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    scenario = json.loads((args.run / "scenario.json").read_text())
    old_walk = json.loads((args.run / "walk/trace.json").read_text())
    stopped = old_walk["actions"][-1]
    assert stopped["decision"] == "hold" and stopped["reason"] == "forward_range_blocked"
    pose = np.asarray(stopped["actor_xz_m"], float)
    yaw = float(stopped["actor_yaw_deg"])
    target = np.asarray(stopped["observed_target_xz_m"], float)
    true_target = np.asarray(scenario["find_target_xz_m"], float)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    route = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1),
                         standoff_min_m=.55, standoff_max_m=.65,
                         max_route_m=3., max_ray_m=3.)
    records = []
    result = "action_cap"
    final_view = None
    final_health = None
    with ReplicaCaptureScene(converted_mesh=args.converted_mesh,
            source_scene=args.source_scene, renderer=args.renderer,
            render_output=out / "rgb_requests",
            floor_y=float(scenario["start_nav_y_m"])) as scene:
        scene.set_props({scenario["target"]: (TexturedFloorObject(
            props / f"{scenario['target']}.obj", scale=.012), true_target, 0.),
            scenario["distractor"]: (TexturedFloorObject(
            props / f"{scenario['distractor']}.obj", scale=.012),
            scenario["find_distractor_xz_m"], 0.)})
        monitor = IndependentRangeMonitor(args.converted_mesh, pose,
            float(scene.floor_height(*pose)), yaw)
        started = monitor.mark()
        with imageio.get_writer(str(out / "first_person.mp4"), fps=5,
                codec="libx264", quality=7, macro_block_size=8) as writer:
            for frame in range(81):
                rgb, depth, _ids, ranges = task.camera(scene, pose, yaw, None)
                writer.append_data(rgb)
                if frame in (0, 80) or frame % 10 == 0:
                    Image.fromarray(rgb).save(out / f"rgb_{frame:03d}.png")
                    Image.fromarray(np.where(np.isfinite(depth),
                        np.clip(np.rint(depth * 1000.), 0, 65535), 0
                        ).astype(np.uint16)).save(out / f"depth_{frame:03d}.png")
                guard = monitor.health(since_wall_s=started)
                if not guard["authorized"]:
                    decision = {"intent": "hold", "reason": guard["reason"]}
                    result = "source_unhealthy"
                else:
                    decision = route.plan(ranges, ANGLES, pose, yaw, target,
                                          time_s=frame * .1)
                row = {"frame": frame, "pose_xz_m": pose.tolist(),
                    "heading_deg": yaw, "target_distance_m": float(
                        np.linalg.norm(pose - target)),
                    "evaluator_target_gap_m": float(np.linalg.norm(
                        pose - true_target)),
                    "nearest_range_m": float(np.min(ranges)),
                    "source_guard": guard, "route": decision}
                intent = decision["intent"]
                if intent == "hold_viewpoint":
                    result = "observed_target_standoff"
                elif intent in ("hold_reobserve", "hold"):
                    result = decision.get("reason", "hold")
                elif intent == "turn_to_waypoint":
                    yaw += float(decision["yaw_delta_deg"])
                elif intent == "step_to_waypoint":
                    amount = float(decision["forward_step_m"])
                    candidate = pose + amount * np.asarray([
                        math.sin(math.radians(yaw)),
                        math.cos(math.radians(yaw))])
                    eye = scene.eye_at(*candidate)
                    candidate_ranges = (scene.range_fan(eye, yaw, owner_xz=None,
                        angles_deg=ANGLES) if eye is not None else None)
                    accepted = bool(eye is not None and
                        scene.floor_supported(*candidate) and
                        np.min(candidate_ranges) >= .17)
                    row["motion_accepted"] = accepted
                    if accepted:
                        pose = candidate
                    else:
                        route.observe_rejected_step(pose, yaw, amount,
                            time_s=frame * .1)
                        result = "simulator_rejected_motion"
                records.append(row)
                if intent in ("hold_viewpoint", "hold_reobserve", "hold") or \
                        result == "simulator_rejected_motion":
                    break
                monitor.update_pose(pose, float(scene.floor_height(*pose)), yaw)
        health = monitor.health(since_wall_s=started)
        if result == "observed_target_standoff" and health["authorized"]:
            teaching = json.loads((args.run / "teach/trace.json").read_text())
            teach_rgb = Image.open(args.run / "teach/teach_rgb.png").convert("RGB")
            vision = Vision("generic-proposal", device="cuda")
            exemplar = vision.feature([teach_rgb.crop(tuple(
                teaching["selected_box"]))])[0]
            colour = task.colour_histogram(np.asarray(teach_rgb),
                                            teaching["selected_box"])
            final_started = monitor.mark()
            final_view = task.run_search(scene, vision, pose, yaw,
                pose - np.asarray(scenario["start_xz_m"], float), yaw,
                exemplar, float(teaching["acceptance_similarity"]), True,
                out / "final_view", start_time_s=3.,
                search_xz=tuple(true_target),
                search_distractor_xz=scenario["find_distractor_xz_m"],
                owner_start_xz=tuple(scenario["start_xz_m"]),
                target_name=scenario["target"],
                distractor_name=scenario["distractor"],
                camera_width=scene.WIDTH, camera_height=scene.HEIGHT,
                identity_mode="multi_cue", colour_exemplar=colour,
                colour_acceptance=float(teaching["colour_acceptance_similarity"]))
            final_health = monitor.health(since_wall_s=final_started)
        packets = monitor.close()
    packet_dir = out / "independent_range"
    packet_dir.mkdir(exist_ok=True)
    (packet_dir / "packets.json").write_text(json.dumps({
        "schema": "ape.independent-two-height-range.v1",
        "period_s": .1,
        "source_geometry": "static converted Replica room mesh; props and person omitted",
        "fault": "none", "packets": packets}, indent=2) + "\n")
    trace = {"schema": "ape.replica-route-recovery.v1",
        "source_run": str(args.run), "observed_target_xz_m": target.tolist(),
        "route_clearance_m": .17,
        "result": result, "final_pose_xz_m": pose.tolist(),
        "evaluator_final_target_gap_m": float(np.linalg.norm(pose - true_target)),
        "source_health": health, "range_packets": len(packets),
        "final_visual_action": final_view["action"] if final_view else None,
        "final_source_health": final_health,
        "effective_final_action": (final_view["action"] if final_view and
            final_health and final_health["authorized"] else None),
        "actions": records,
        "limits": "Development-exposed room; static ideal range source, ideal command pose, ideal per-command range/floor motion screen, 17cm kinematic disk, no selected ape body."}
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps({k: v for k, v in trace.items() if k not in ("actions", "source_health")}))


if __name__ == "__main__":
    main()
