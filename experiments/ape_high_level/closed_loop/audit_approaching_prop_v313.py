"""V313: approaching low-prop counterexample and uncertainty-aware action gate.

Four new geometry-valid Direct Frame screens bring the prop toward a lateral
step after its current frame. Compare current occupancy, nominal two-frame
motion, and lower lateral-velocity bounds across all ten timed scenes.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_moving_cluster_prediction_v311 import (
    depth_cloud, forecast_gap, groups, MATCH_M, MIN_CLUSTER_POINTS,
)
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import (
    CASES as EARLIER_CASES, HeldoutMovingRoom, screen_case,
)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v313"
V308 = ROOT / "artifacts/ape-closed-loop-cove50-v308/moving-depth-action-audit.json"
V311 = ROOT / "artifacts/ape-closed-loop-cove50-v311/moving-cluster-prediction-audit.json"
APPROACH_CASES = tuple((seed, "present", .30, 7.4, 8., 10., 12.)
                       for seed in (160, 162, 164, 166))
CASES = EARLIER_CASES + APPROACH_CASES
SPEED_UNCERTAINTY_M_S = (0., .05, .10, .15, .20)
ACTOR_MARGIN_M = .065
EVALUATOR_MARGIN_M = .05


def video_frame(side: Image.Image, external: Image.Image,
                label: str) -> np.ndarray:
    panel = Image.fromarray(np.concatenate((np.asarray(side),
                                            np.asarray(external)), axis=1))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(panel)


def main() -> None:
    earlier_truth = {r["case"][0]: r for r in json.loads(V308.read_text())["rows"]}
    earlier_actor = {r["case"][0]: r for r in json.loads(V311.read_text())["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = [screen_case(case) for case in APPROACH_CASES]
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        branch = OUT / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=branch)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H,
                             agreement_m=.04)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            previous_points, _ = depth_cloud(room, mask, seed, 7.2, 0., 0., 12)
            current_points, actor_qpos = depth_cloud(
                room, mask, seed, 7.6, .0464, .08, 22)
            previous, current = groups(previous_points), groups(current_points)
            if sum(map(len, current)) != len(current_points):
                raise AssertionError("Point cluster loss")
            speeds = []
            tracks = []
            for cluster in current:
                center = cluster[:, :2].mean(axis=0)
                options = [(float(np.linalg.norm(
                    center - old[:, :2].mean(axis=0))), old)
                    for old in previous
                    if len(cluster) >= MIN_CLUSTER_POINTS and
                    len(old) >= MIN_CLUSTER_POINTS]
                distance, match = (min(options, key=lambda item: item[0])
                                   if options else (float("inf"), None))
                matched = match is not None and distance <= MATCH_M
                speed = ((center - match[:, :2].mean(axis=0)) / .4
                         if matched else np.zeros(2))
                speeds.append(speed)
                tracks.append({"points": len(cluster),
                               "matched_previous": matched,
                               "estimated_speed_xy_m_s": speed.round(5).tolist()})
            np.savez_compressed(branch / "actor-low-clusters.npz",
                                previous_points_xyz_m=previous_points,
                                current_points_xyz_m=current_points,
                                actor_estimated_pose_xyyaw=np.array((0., .08, 0.)),
                                previous_time_s=7.2, current_time_s=7.6)
            actor_qpos[room.head_addr] = 0.
            pose = np.array((0., .08, 0.))
            static = forecast_gap(robot, actor_qpos, pose, current, speeds,
                                  dynamic=False)
            variants = {}
            for uncertainty in SPEED_UNCERTAINTY_M_S:
                lower_speeds = []
                for track, velocity in zip(tracks, speeds):
                    lower = velocity.copy()
                    if track["matched_previous"]:
                        lower[1] -= uncertainty
                    lower_speeds.append(lower)
                lower_gap = forecast_gap(robot, actor_qpos, pose,
                                         current, lower_speeds, dynamic=True)
                nominal_gap = forecast_gap(robot, actor_qpos, pose,
                                           current, speeds, dynamic=True)
                robust_gap = min(lower_gap, nominal_gap)
                variants[str(uncertainty)] = {
                    "actor_min_gap_m": round(robust_gap, 5),
                    "actor_pass": robust_gap >= ACTOR_MARGIN_M}
            if seed in earlier_actor:
                if abs(static - earlier_actor[seed]["actor_static_gap_m"]) > 1e-5:
                    raise AssertionError("V311 static comparator changed")
                if abs(variants["0.0"]["actor_min_gap_m"] - earlier_actor[
                        seed]["actor_predicted_gap_m"]) > 1e-5:
                    raise AssertionError("V311 nominal comparator changed")
            # Independent evaluator: moving full-body geom clearance. The
            # actor point/velocity decision above is complete before this.
            scene = scene_geoms(room)
            bodies = body_geoms(room, scene)
            path = []
            video = None
            if seed in (160, 162, 164, 166):
                room.time_s = 7.6
                room.actual[:] = (0., .0464, 0.)
                room.head = 1.2
                room._update_moving_obstacle()
                room._forward()
                side, _ = room.capture("lower_rgb", depth=False)
                video = imageio.get_writer(branch / "approaching-step.mp4",
                                           fps=10, codec="libx264", quality=7)
                start_image = video_frame(side, room.capture_external(),
                    f"seed {seed} | t7.6 side look | static={static:.3f}m "
                    f"nominal={variants['0.0']['actor_min_gap_m']:.3f}m")
                for _ in range(5):
                    video.append_data(start_image)
            try:
                for fraction in np.linspace(0., 1., 21):
                    room.time_s = 7.6 + .4 * float(fraction)
                    room._update_moving_obstacle()
                    room.actual[:] = (0., .0464 + .94 * .08 * fraction, 0.)
                    room.head = 0.
                    room._forward()
                    gap, robot_geom, scene_geom = full_body_clearance(
                        room, scene, bodies)
                    path.append({"time_s": round(room.time_s, 4),
                                 "evaluator_model_clearance_m": round(float(gap), 5),
                                 "limiting_robot_geom": robot_geom,
                                 "limiting_scene_geom": scene_geom})
                    if video is not None:
                        video.append_data(video_frame(
                            side, room.capture_external(),
                            f"seed {seed} | t{room.time_s:.2f} | "
                            f"full-body gap {gap:.3f}m"))
            finally:
                if video is not None:
                    video.close()
            minimum = min(path, key=lambda entry:
                          entry["evaluator_model_clearance_m"])
            if seed in earlier_truth and abs(minimum[
                    "evaluator_model_clearance_m"] - earlier_truth[seed][
                        "evaluator_min_dynamic_clearance_m"]) > 1e-5:
                raise AssertionError("Earlier full-body label changed")
            trace = branch / "action-trace.json"
            trace.write_text(json.dumps({"case": case, "actor_tracks": tracks,
                                         "actor_static_gap_m": round(static, 5),
                                         "actor_variants": variants,
                                         "evaluator_path": path}, indent=2) + "\n")
            rows.append({"case": case,
                         "actor_cluster_path": str((branch / "actor-low-clusters.npz").relative_to(ROOT)),
                         "actor_cluster_sha256": sha(branch / "actor-low-clusters.npz"),
                         "actor_tracks": tracks,
                         "actor_static_gap_m": round(static, 5),
                         "actor_static_pass": static >= ACTOR_MARGIN_M,
                         "actor_variants": variants,
                         "evaluator_minimum": minimum,
                         "evaluator_safe_5cm": minimum[
                             "evaluator_model_clearance_m"] >= EVALUATOR_MARGIN_M,
                         "trace_path": str(trace.relative_to(ROOT)),
                         "trace_sha256": sha(trace),
                         "video_path": (str((branch / "approaching-step.mp4").relative_to(ROOT))
                                        if video is not None else None),
                         "video_sha256": (sha(branch / "approaching-step.mp4")
                                          if video is not None else None)})
            print(seed, mode, "safe", rows[-1]["evaluator_safe_5cm"],
                  "static", round(static, 4),
                  "uncertainty", {key: value["actor_pass"]
                                  for key, value in variants.items()}, flush=True)
        finally:
            mask.close()
            room.close()
    def score(key: str) -> dict:
        decisions = [row["actor_static_pass"] if key == "static" else
                     row["actor_variants"][key]["actor_pass"] for row in rows]
        return {"actions": len(rows),
                "unsafe_actions": sum(not row["evaluator_safe_5cm"] for row in rows),
                "false_safe": sum(decision and not row["evaluator_safe_5cm"]
                                  for decision, row in zip(decisions, rows)),
                "false_block": sum(not decision and row["evaluator_safe_5cm"]
                                   for decision, row in zip(decisions, rows)),
                "passed": sum(decisions)}
    scores = {"static": score("static"), **{
        str(uncertainty): score(str(uncertainty))
        for uncertainty in SPEED_UNCERTAINTY_M_S}}
    report = {"scope": "V313 extends V311's six Direct Frame timed scenes with four new geometry-valid approaching-prop screens at seeds160/162/164/166. All use paired actor-only +1.2rad lower 7–50cm depth t7.2/7.6, 1cm Gaussian noise, 15% holes, 4cm self mask and fixed 4cm ground cut. Compare a static current cloud, nominal associated-cluster velocity, and a lower lateral-velocity bound 0–20cm/s below nominal during the 0.4s 8cm side step. The last bounds are sensitivity parameters, not calibrated covariance. Actor 6.5cm point gap; independent evaluator advances selected full-body solids and scripted prop every20ms and labels 5cm safe. New scenes have a currently clear-looking step while the prop approaches; timing/motion was declared before actor runs and the geometry screen is evaluator-only. Side-looking head actuation is omitted; native gait/stop, observed-free coverage and calibrated uncertainty are still required. Videos show the evaluator path and actor side view, never evidence passed to actor.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "v308_sha256": sha(V308), "v311_sha256": sha(V311),
              "new_case_preflight": preflight,
              "rows": rows, "scores": scores}
    (OUT / "approaching-prop-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
