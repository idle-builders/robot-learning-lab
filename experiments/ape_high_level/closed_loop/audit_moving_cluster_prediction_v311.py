"""V311: actor-only side-depth cluster tracking for one moving step.

This diagnostic predicts a projected low obstacle across a 0.4s action from
two side-looking frames. The frames are V307's simultaneous-look upper bound:
head actuation, exposure and scan delay are not yet part of the actor timing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_localization_v297 import depth_variant
from audit_lateral_clutter_depth_v285 import project
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import CASES, HeldoutMovingRoom
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v311"
V308 = ROOT / "artifacts/ape-closed-loop-cove50-v308/moving-depth-action-audit.json"
V310 = ROOT / "artifacts/ape-closed-loop-cove50-v310/ground-depth-audit.json"
OBSERVATIONS = ((7.2, 0., 0., 12), (7.6, .0464, .08, 22))
HEIGHT_MIN_M = .04
CONNECT_M = .04
MIN_CLUSTER_POINTS = 8
MATCH_M = .16
MARGINS_M = (.05, .065, .10)


def groups(points: np.ndarray) -> list[np.ndarray]:
    if not len(points):
        return []
    adjacent = np.linalg.norm(points[:, None, :2] -
                              points[None, :, :2], axis=2) < CONNECT_M
    seen: set[int] = set()
    clusters: list[np.ndarray] = []
    for index in range(len(points)):
        if index in seen:
            continue
        pending = [index]
        seen.add(index)
        members = []
        while pending:
            member = pending.pop()
            members.append(member)
            for other in np.where(adjacent[member])[0]:
                other = int(other)
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        clusters.append(points[members])
    return sorted(clusters, key=len, reverse=True)


def depth_cloud(room, mask: SelfDepthMask, seed: int, time_s: float,
                actual_y: float, estimated_y: float, variant_index: int):
    room.time_s = time_s
    room.actual[:] = (0., actual_y, 0.)
    room.estimated[:] = (0., estimated_y, 0.)
    room.head = 1.2
    room._update_moving_obstacle()
    room._forward()
    _, ideal = room.capture("lower_rgb", depth=True)
    if ideal is None:
        raise AssertionError("Rendered depth unavailable")
    measured = depth_variant(ideal, "short", seed, variant_index)
    rng = np.random.default_rng(seed * 1009 +
                                (0 if time_s == 7.2 else 41) + 12)
    measured += rng.normal(0., .01, measured.shape).astype(measured.dtype)
    measured[rng.random(measured.shape) < .15] = np.nan
    camera_pose = room.camera_pose_estimate("lower_rgb")
    qpos = room.pose_data.qpos.copy()
    filtered, _ = mask.apply(measured, qpos, max_depth_m=2.5)
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    points = project(filtered, camera_pose, fovy)
    points = points[points[:, 2] >= HEIGHT_MIN_M]
    return points, qpos


def forecast_gap(robot: RobotGeometry, qpos: np.ndarray, pose: np.ndarray,
                 clusters: list[np.ndarray], velocity: list[np.ndarray],
                 *, dynamic: bool) -> float:
    gap = float("inf")
    for fraction in np.linspace(0., 1., 21):
        future = pose.copy()
        future[1] += .08 * fraction
        robot.set_state(robot.pose_qpos(qpos, future))
        for points, speed in zip(clusters, velocity):
            cloud = points.copy()
            if dynamic:
                cloud[:, :2] += speed * (.4 * fraction)
            gap = min(gap, float(np.min(RobotGeometry.solid_distances(
                cloud, robot.primitives()))))
    return gap


def main() -> None:
    truth = {r["case"][0]: r for r in json.loads(V308.read_text())["rows"]}
    baseline = {r["case"][0]: r for r in json.loads(V310.read_text())["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=OUT)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H,
                             agreement_m=.04)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            frames = [depth_cloud(room, mask, seed, *observation)
                      for observation in OBSERVATIONS]
            previous = groups(frames[0][0])
            current = groups(frames[1][0])
            if sum(map(len, current)) != len(frames[1][0]):
                raise AssertionError("Every small return must remain in the gate")
            velocities = []
            track = []
            for cluster in current:
                center = cluster[:, :2].mean(axis=0)
                options = [(float(np.linalg.norm(center - old[:, :2].mean(axis=0))),
                            old) for old in previous
                           if len(cluster) >= MIN_CLUSTER_POINTS and
                           len(old) >= MIN_CLUSTER_POINTS]
                distance, matched = min(options, key=lambda pair: pair[0]) if options else (float("inf"), None)
                speed = ((center - matched[:, :2].mean(axis=0)) / .4
                         if matched is not None and distance <= MATCH_M
                         else np.zeros(2))
                velocities.append(speed)
                track.append({"points": len(cluster),
                              "centroid_xy_m": center.round(5).tolist(),
                              "matched_previous": matched is not None and distance <= MATCH_M,
                              "match_displacement_m": None if not np.isfinite(distance)
                              else round(distance, 5),
                              "estimated_velocity_xy_m_s": speed.round(5).tolist()})
            qpos = frames[1][1].copy()
            qpos[room.head_addr] = 0.
            pose = np.array((0., .08, 0.))
            static_gap = forecast_gap(robot, qpos, pose, current, velocities,
                                      dynamic=False)
            moving_gap = forecast_gap(robot, qpos, pose, current, velocities,
                                      dynamic=True)
            if abs(round(static_gap, 5) - baseline[seed]["variants"]["0.04"]["actor_gap_m"]) > 1e-5:
                raise AssertionError("Static frozen ground-filter baseline changed")
            rows.append({"case": case,
                         "evaluator_safe_5cm": truth[seed]["evaluator_safe_5cm"],
                         "evaluator_min_dynamic_clearance_m": truth[seed][
                             "evaluator_min_dynamic_clearance_m"],
                         "previous_cluster_sizes": [len(c) for c in previous],
                         "current_tracks": track,
                         "actor_static_gap_m": round(static_gap, 5),
                         "actor_predicted_gap_m": round(moving_gap, 5),
                         "actor_pass": {
                             str(margin): {"static": static_gap >= margin,
                                           "predicted": moving_gap >= margin}
                             for margin in MARGINS_M}})
            print(seed, mode, "safe", truth[seed]["evaluator_safe_5cm"],
                  "track", track, "gaps", round(static_gap, 4),
                  round(moving_gap, 4), flush=True)
        finally:
            mask.close()
            room.close()
    scores = {}
    for margin in MARGINS_M:
        for variant in ("static", "predicted"):
            key = f"{margin}:{variant}"
            scores[key] = {
                "actions": len(rows),
                "unsafe_actions": sum(not r["evaluator_safe_5cm"] for r in rows),
                "false_safe": sum(r["actor_pass"][str(margin)][variant] and
                                  not r["evaluator_safe_5cm"] for r in rows),
                "false_block": sum(not r["actor_pass"][str(margin)][variant] and
                                   r["evaluator_safe_5cm"] for r in rows),
                "passed": sum(r["actor_pass"][str(margin)][variant]
                              for r in rows)}
    report = {"scope": "V311 re-renders V307's same six selected Direct Frame side-looking short-depth frames at 7.2/7.6s with deterministic 1cm Gaussian noise, 15% holes, actor estimated pose, 4cm self agreement and fixed 4cm floor-height cut. Actor clusters 4cm-neighbor projected returns, associates a cluster centroid within16cm and predicts constant XY motion through a 0.4s 8cm side step. The comparator keeps clusters fixed. Both use actor body solids and margins 5/6.5/10cm; evaluator-only V308 full-body moving prop labels score the decisions. These captures assume the head already points +1.2rad at both timestamps and exclude look actuation, stop travel, calibrated floor/pose uncertainty, unobserved-space proof and physical depth module. A motion estimate from two 0.4s-apart frames cannot establish obstacle dynamics or safety.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "v308_sha256": sha(V308), "v310_sha256": sha(V310),
              "observations": OBSERVATIONS,
              "rows": rows, "scores": scores}
    (OUT / "moving-cluster-prediction-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
