"""Continue V246p after a safe first viewpoint fails to reveal the item.

The actor takes a fresh bounded head sweep, fuses medium-depth and near-range
free/occupied/unknown evidence, and may take one more 8 cm forward step.
It then searches again and still requires a fresh owner-confirmed view.
This is historical kinematic motion, not a physical skill qualification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from depth_voxel import VOXEL_M
from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live
from run_medium_depth_recovery_v246p import CASES, MediumWindowMap, profile_depth


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v250p"
HEAD_YAWS = (0., -.6, .4)
FINAL_YAWS = (.4, .5)


def after_first_view(room, observe, branch: Path, video, robot,
                     filtered_depth, seed: int) -> dict:
    voxel_map = MediumWindowMap()
    head_inputs = []
    for index, yaw in enumerate(HEAD_YAWS):
        room.look(yaw, "lower_rgb")
        rgb, ideal = room.capture("lower_rgb", depth=True)
        if ideal is None:
            raise AssertionError("Recovery head depth unavailable")
        measured = live.noisy_depth(ideal, "short", seed, 600 + index)
        camera_pose = room.camera_pose_estimate("lower_rgb")
        qpos = room.pose_data.qpos.copy()
        filtered = filtered_depth(room, "lower_rgb", measured, qpos)
        mapped, quality = reject_isolated_depth_spikes(filtered)
        fovy = float(room.model.cam_fovy[
            room.model.camera("lower_rgb").id])
        map_info = voxel_map.observe(
            mapped, camera_pose, room.time_s, fovy_deg=fovy,
            stride=4, max_depth_m=1., self_masked=True)
        path = branch / f"second-head-{index:02d}.npz"
        np.savez_compressed(path, rgb=np.asarray(rgb),
                            measured_depth_m=measured,
                            filtered_depth_m=filtered,
                            map_depth_m=mapped,
                            camera_origin_estimated_m=camera_pose[0],
                            camera_rotation_estimated=camera_pose[1],
                            actor_qpos=qpos,
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        video.append_data(live.panel(
            room, rgb, f"second viewpoint look yaw{yaw:+.1f} "
                       f"t{room.time_s:.1f}"))
        head_inputs.append({"yaw_rad": yaw,
                            "capture_time_s": round(room.time_s, 4),
                            "actor_frame_path": str(path.relative_to(ROOT)),
                            "actor_frame_sha256": sha(path),
                            "map_info": map_info,
                            "depth_quality": quality})

    time_s = room.time_s
    pose = room.estimated.copy()
    qpos = room.pose_data.qpos.copy()
    packets, packet_counts = live.add_near_packets(
        room, pose, seed, time_s, voxel_map)
    packet_path = branch / "second-near-range-packets.json"
    packet_path.write_text(json.dumps({
        "capture_time_s": time_s,
        "estimated_pose_xyyaw": pose.tolist(),
        "mounts_body_m_and_yaw_rad": live.NEAR_MOUNTS,
        "ray_packets": packets}, indent=2) + "\n")
    future = pose.copy()
    future[0] += live.STEP_M
    shell = robot.swept_exterior_coverage(
        voxel_map, qpos, pose, future, time_s,
        margin_m=live.UPPER_SHELL_MARGIN_M)
    occupied = np.asarray([[(x + .5) * VOXEL_M,
                            (y + .5) * VOXEL_M,
                            (z + .5) * VOXEL_M]
                           for (x, y, z), stamp in voxel_map.occupied.items()
                           if stamp >= time_s - .4 and
                           stamp >= voxel_map.free.get((x, y, z), -math.inf)
                           and (z + .5) * VOXEL_M >= .04],
                          dtype=float).reshape(-1, 3)
    upper_gap = live.forward_point_gap(robot, qpos, pose, occupied)
    shin, shin_counts = live.actor_shin_points(
        room, robot, f"v250-second-{seed}-{time_s:.2f}")
    shin_gap = live.forward_point_gap(robot, qpos, pose, shin)
    shin_age = max(0., time_s - float(shin_counts.get(
        "capture_time_s", time_s)))
    shin_path = branch / "second-shin.npz"
    np.savez_compressed(shin_path,
                        estimated_environment_points_xyz_m=shin,
                        actor_qpos=qpos,
                        estimated_pose_xyyaw=pose,
                        capture_time_s=time_s)
    oldest_head_age = time_s - head_inputs[0]["capture_time_s"]
    gate = {"status": "free" if (
        shell["status"] == "free" and oldest_head_age <= .8 + 1e-6 and
        shin_age <= .4 and len(shin) >= 8 and
        min(shin_gap, upper_gap) >= live.POINT_MOTION_MARGIN_M and
        not room.motion_interrupted) else "blocked",
        "shell_plus_5cm": shell,
        "oldest_head_age_s": round(oldest_head_age, 4),
        "fresh_shin_age_s": round(shin_age, 4),
        "shin_points": len(shin),
        "actor_shin_swept_gap_m": round(shin_gap, 5),
        "actor_upper_occupied_swept_gap_m": round(upper_gap, 5),
        "point_motion_margin_m": live.POINT_MOTION_MARGIN_M,
        "fresh_occupied_voxel_points": len(occupied),
        "packet_counts": packet_counts,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": sha(packet_path),
        "shin_frame_path": str(shin_path.relative_to(ROOT)),
        "shin_frame_sha256": sha(shin_path)}
    audit = {"head_inputs": head_inputs,
             "actor_gate": gate,
             "actor_motion_authorized": gate["status"] == "free",
             "candidate_step_body_m": [live.STEP_M, 0.],
             "post_step_search_yaws_rad": FINAL_YAWS}
    rgb, _ = room.capture("lower_rgb", depth=False)
    video.append_data(live.panel(
        room, rgb, f"second route {gate['status']} | "
                   f"unknown {shell['unknown']} | "
                   f"shin {shin_gap:.3f}m | upper {upper_gap:.3f}m"))
    if gate["status"] != "free":
        return {"audit": audit, "found_event": None}
    room.move(live.STEP_M, 0.)
    found = None
    for yaw in FINAL_YAWS:
        room.look(yaw, "lower_rgb")
        view = observe("lower_rgb", f"second forward yaw{yaw:+.1f}")
        if view["accepted_box"] is not None:
            found = view
            break
    audit["post_step_found_actor_box"] = (
        None if found is None else found["accepted_box"])
    return {"audit": audit, "found_event": found}


def main() -> None:
    original_cases = live.CASES
    original_depth = live.noisy_depth
    original_map = live.DepthVoxelMap
    try:
        live.CASES = CASES
        live.noisy_depth = profile_depth("medium_30cm_1m", original_depth)
        live.DepthVoxelMap = MediumWindowMap
        live.main(output_root=OUT, source=SOURCE,
                  map_depth_filter=reject_isolated_depth_spikes,
                  experiment_label="V250p second certified information view",
                  after_first_view=after_first_view)
    finally:
        live.CASES = original_cases
        live.noisy_depth = original_depth
        live.DepthVoxelMap = original_map
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["scope_override"] = (
        "V246p medium head depth and owner task with V321's first certified "
        "forward view. If no item is accepted there, an actor-only fresh "
        "head/range/shin free/occupied/unknown gate may authorize exactly "
        "one more 8cm forward step and a bounded +0.4/+0.5rad relook. "
        "Both movements are kinematic historical-body surrogates; item "
        "truth only scores the unchanged owner-confirmation outcome.")
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.second-viewpoint.v1",
        "source_sha256": sha(SOURCE),
        "runner_report_path": str(path.relative_to(ROOT)),
        "runner_report_sha256": sha(path),
        "cases": CASES,
        "rows": [{"case": row["case"], "status": row["status"],
                  "minimum_clearance_m": row[
                      "evaluator_path_min_clearance_m"],
                  "first_gate": row["recovery_gate"]["status"]}
                 for row in report["rows"]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
