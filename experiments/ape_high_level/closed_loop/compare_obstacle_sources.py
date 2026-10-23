"""Make a paired no-map/depth/range obstacle-decision clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


CONFIGS = (("RGB-D, no obstacle map", "chest_rgbd"),
           ("RGB-D depth obstacle points", "chest_rgbd_depth_obstacles"),
           ("RGB-D + 72 range beams", "chest_rgbd_range"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--active", action="store_true",
                        help="Compare fixed chest depth with a swept head-depth view")
    parser.add_argument("--recovery", action="store_true",
                        help="Compare range/head depth with and without view recovery")
    parser.add_argument("--continuous", action="store_true",
                        help="Compare range and head-depth decisions with a continuously moving prop")
    parser.add_argument("--monitor", action="store_true",
                        help="Compare range recovery with and without the motion monitor")
    parser.add_argument("--predictive", action="store_true",
                        help="Compare range monitor with footprint and motion-aware routing")
    parser.add_argument("--furnished", action="store_true",
                        help="Compare swept head depth and two range routes in a furnished room")
    parser.add_argument("--dual-monitor", action="store_true",
                        help="Compare temporal range tracking with the independent raw-scan alarm")
    parser.add_argument("--midstep", action="store_true",
                        help="Compare dual monitoring with a 10 Hz guard during translation")
    parser.add_argument("--move-wait", action="store_true",
                        help="Compare translation-only and move-and-wait 10 Hz guards")
    parser.add_argument("--reacquire", action="store_true",
                        help="Compare move/wait guard with watched target reacquisition")
    parser.add_argument("--active-view", action="store_true",
                        help="Compare watched reacquisition with range-cleared viewpoint search")
    parser.add_argument("--delayed-view", action="store_true",
                        help="Compare watched reacquisition with delayed viewpoint search")
    parser.add_argument("--head-fallback", action="store_true",
                        help="Compare chest-only recovery with guarded lower-head RGB-D fallback")
    parser.add_argument("--view-route", action="store_true",
                        help="Compare delayed sideways search with target-viewpoint routing")
    parser.add_argument("--goal-map", action="store_true",
                        help="Compare delayed sideways search with a persistent viewing-goal map")
    parser.add_argument("--shadow-baseline-run", type=Path,
                        help="Compare the same goal-map config in this run against a prior run without head-depth shadow")
    parser.add_argument("--marking-baseline-run", type=Path,
                        help="Compare a head-depth shadow goal map against fresh upper-depth obstacle marking")
    parser.add_argument("--frontier-baseline-run", type=Path,
                        help="Compare the same depth-marked goal map with and without confirmed-target frontier viewpoints")
    parser.add_argument("--true-frontier-baseline-run", type=Path,
                        help="Compare confirmed-target viewpoints with and without a true free/unknown frontier boundary")
    parser.add_argument("--view-ray-baseline-run", type=Path,
                        help="Compare the same guarded goal map with and without live 3-D crop-ray viewpoint goals")
    parser.add_argument("--local-progress-baseline-run", type=Path,
                        help="Compare 8 cm crop-ray goals with one-step 4 cm progressive view goals")
    parser.add_argument("--two-height-baseline-run", type=Path,
                        help="Compare 4 cm progressive views with two-height bounded detour views")
    args = parser.parse_args()
    if sum(bool(root) for root in (args.shadow_baseline_run,
                                   args.marking_baseline_run,
                                   args.frontier_baseline_run,
                                   args.true_frontier_baseline_run,
                                   args.view_ray_baseline_run,
                                   args.local_progress_baseline_run,
                                   args.two_height_baseline_run)) > 1:
        parser.error("Choose only one paired baseline run")
    paired_baseline_run = (args.shadow_baseline_run or args.marking_baseline_run
                           or args.frontier_baseline_run or
                           args.true_frontier_baseline_run or
                           args.view_ray_baseline_run or
                           args.local_progress_baseline_run or
                           args.two_height_baseline_run)
    if args.two_height_baseline_run:
        configs = (("4 cm progressive view baseline", "chest_rgbd_range_guard_goal_map"),
                   ("Two-height bounded detour view", "chest_rgbd_range_guard_goal_map"))
    elif args.local_progress_baseline_run:
        configs = (("8 cm crop-ray view goals", "chest_rgbd_range_guard_goal_map"),
                   ("4 cm progressive view goals", "chest_rgbd_range_guard_goal_map"))
    elif args.view_ray_baseline_run:
        configs = (("Confirmed-target frontier baseline", "chest_rgbd_range_guard_goal_map"),
                   ("3-D crop-ray short view goals", "chest_rgbd_range_guard_goal_map"))
    elif args.true_frontier_baseline_run:
        configs = (("Confirmed-target reachable cells", "chest_rgbd_range_guard_goal_map"),
                   ("True free/unknown frontier viewpoints", "chest_rgbd_range_guard_goal_map"))
    elif args.frontier_baseline_run:
        configs = (("Depth-marked map, no confirmed frontier", "chest_rgbd_range_guard_goal_map"),
                   ("Depth-marked map + confirmed frontier", "chest_rgbd_range_guard_goal_map"))
    elif args.marking_baseline_run:
        configs = (("Same timed head looks, map in shadow", "chest_rgbd_range_guard_goal_map"),
                   ("Fresh upper-depth marks in goal map", "chest_rgbd_range_guard_goal_map"))
    elif args.shadow_baseline_run:
        configs = (("Goal map, no head-depth pauses", "chest_rgbd_range_guard_goal_map"),
                   ("Goal map + head-depth shadow", "chest_rgbd_range_guard_goal_map"))
    elif args.goal_map:
        configs = (("Delayed sideways search", "chest_rgbd_range_guard_delayed_view"),
                   ("Persistent viewing goal", "chest_rgbd_range_guard_goal_map"))
    elif args.view_route:
        configs = (("Delayed sideways search", "chest_rgbd_range_guard_delayed_view"),
                   ("Range-planned target viewpoint", "chest_rgbd_range_guard_view_route"))
    elif args.head_fallback:
        configs = (("Chest RGB-D + range", "chest_rgbd_range_guard_delayed_view"),
                   ("Chest + lower head RGB-D + range", "chest_head_rgbd_range_guard"))
    elif args.delayed_view:
        configs = (("Fresh-view resume", "chest_rgbd_range_guard_reacquire"),
                   ("Wait, then safe viewpoint + resume",
                    "chest_rgbd_range_guard_delayed_view"))
    elif args.active_view:
        configs = (("Fresh-view resume", "chest_rgbd_range_guard_reacquire"),
                   ("Fresh-view + safe viewpoint", "chest_rgbd_range_guard_active_view"))
    elif args.reacquire:
        configs = (("10 Hz guard, no task resume", "chest_rgbd_range_move_wait_guard"),
                   ("10 Hz guard + fresh-view resume", "chest_rgbd_range_guard_reacquire"))
    elif args.move_wait:
        configs = (("Range: 10 Hz moves only", "chest_rgbd_range_midstep_guard"),
                   ("Range: 10 Hz moves + waits", "chest_rgbd_range_move_wait_guard"))
    elif args.midstep:
        configs = (("Range: dual alarm between actions", "chest_rgbd_range_dual_monitor"),
                   ("Range: dual alarm + 10 Hz action guard", "chest_rgbd_range_midstep_guard"))
    elif args.dual_monitor:
        configs = (("Range: temporal motion monitor", "chest_rgbd_range_monitored"),
                   ("Range: temporal + raw-scan monitor", "chest_rgbd_range_dual_monitor"))
    elif args.furnished:
        configs = (("Swept head depth + view recovery", "chest_rgbd_head_sweep_recovery"),
                   ("Range + reactive motion monitor", "chest_rgbd_range_monitored"),
                   ("Range + footprint route", "chest_rgbd_range_predictive"))
    elif args.predictive:
        configs = (("Range: reactive motion monitor", "chest_rgbd_range_monitored"),
                   ("Range: footprint + motion route", "chest_rgbd_range_predictive"))
    elif args.monitor:
        configs = (("Range: route recovery only", "chest_rgbd_range_recovery"),
                   ("Range: motion monitor + escape", "chest_rgbd_range_monitored"))
    elif args.continuous:
        configs = (("Range + view recovery", "chest_rgbd_range_recovery"),
                   ("Head depth: lost-view stop", "chest_rgbd_head_sweep_depth"),
                   ("Head depth + view recovery", "chest_rgbd_head_sweep_recovery"))
    elif args.recovery:
        configs = (("Range: stop on lost view", "chest_rgbd_range"),
                   ("Range: bounded view recovery", "chest_rgbd_range_recovery"),
                   ("Head depth: stop on lost view", "chest_rgbd_head_sweep_depth"),
                   ("Head depth: bounded view recovery", "chest_rgbd_head_sweep_recovery"))
    elif args.active:
        configs = (("Fixed chest depth", "chest_rgbd_depth_obstacles"),
                   ("Active head depth", "chest_rgbd_head_sweep_depth"),
                   ("72 range beams", "chest_rgbd_range"))
    else:
        configs = CONFIGS
    folders = ([(root / f"{args.scenario}-seed{args.seed}-0-{config}-none")
                for root, (_, config) in zip((paired_baseline_run, args.run), configs)]
               if paired_baseline_run else
               [args.run / f"{args.scenario}-seed{args.seed}-0-{config}-none"
                for _, config in configs])
    traces = [json.loads((folder / "trace.json").read_text()) for folder in folders]
    readers = [imageio.get_reader(folder / "episode.mp4") for folder in folders]
    try:
        counts = [reader.count_frames() for reader in readers]
        label = ("two-height-detour" if args.two_height_baseline_run else
                 "local-progress-view" if args.local_progress_baseline_run else
                 "view-ray-goals" if args.view_ray_baseline_run else
                 "true-frontier" if args.true_frontier_baseline_run else
                 "confirmed-frontier" if args.frontier_baseline_run else
                 "head-depth-marking" if args.marking_baseline_run else
                 "head-depth-shadow" if args.shadow_baseline_run else
                 "goal-map" if args.goal_map else
                 "view-route" if args.view_route else
                 "head-fallback" if args.head_fallback else
                 "delayed-view" if args.delayed_view else
                 "active-view" if args.active_view else
                 "reacquire" if args.reacquire else
                 "move-wait" if args.move_wait else "midstep" if args.midstep else
                 "dual-monitor" if args.dual_monitor else
                 "furnished" if args.furnished else "predictive" if args.predictive else
                 "monitor" if args.monitor else
                 "continuous" if args.continuous else "recovery" if args.recovery
                 else "active" if args.active else "obstacles")
        prefix = ("tri-grid" if args.furnished else "pair" if paired_baseline_run or args.goal_map or args.view_route or args.head_fallback or args.delayed_view or args.active_view or args.reacquire or args.move_wait or args.midstep or args.dual_monitor or args.monitor or
                  args.predictive else "quad" if args.recovery else "triple")
        output = args.run / f"{prefix}-{label}-{args.scenario}-seed{args.seed}.mp4"
        with imageio.get_writer(output, fps=4, codec="libx264", quality=7) as writer:
            for index in range(max(counts)):
                parts = [reader.get_data(min(index, count - 1))
                         for reader, count in zip(readers, counts)]
                if args.furnished:
                    blank = np.zeros_like(parts[0])
                    pixels = np.concatenate((np.concatenate(parts[:2], axis=1),
                                             np.concatenate((parts[2], blank), axis=1)), axis=0)
                elif args.recovery:
                    pixels = np.concatenate((np.concatenate(parts[:2], axis=1),
                                             np.concatenate(parts[2:], axis=1)), axis=0)
                else:
                    pixels = np.concatenate(parts, axis=1)
                image = Image.fromarray(pixels)
                draw = ImageDraw.Draw(image)
                for side, ((title, _), trace) in enumerate(zip(configs, traces)):
                    grid = args.recovery or args.furnished
                    x = (side % 2 if grid else side) * parts[side].shape[1]
                    y = (side // 2 if grid else 0) * parts[side].shape[0]
                    draw.rectangle((x, y, x + parts[side].shape[1], y + 52), fill=(0, 0, 0))
                    draw.text((x + 8, y + 6), title, fill=(255, 255, 255))
                    draw.text((x + 8, y + 27),
                              f"Clearance {trace['minimum_robot_obstacle_clearance_m']*100:.1f} cm | "
                              f"{trace['outcome']}", fill=(230, 230, 230))
                if args.furnished:
                    x, y = parts[0].shape[1], parts[0].shape[0]
                    draw.text((x + 28, y + 28), "Actor: RGB-D / depth or range / estimated pose",
                              fill=(245, 245, 245))
                    draw.text((x + 28, y + 58), "Map and external camera: evaluator review only",
                              fill=(245, 245, 245))
                writer.append_data(np.asarray(image))
    finally:
        for reader in readers:
            reader.close()
    output.with_suffix(".json").write_text(json.dumps({
        "inputs": [str(folder / "episode.mp4") for folder in folders],
        "traces": [str(folder / "trace.json") for folder in folders],
        "map_note": "Map and true object positions are evaluator overlays only.",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
