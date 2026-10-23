"""Independently audit the paired second/third-view owner episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from direct_structural_rooms_v270 import selected_visual
from run import ROOT, model_with_room


BASE = ROOT / "artifacts/ape-articulated-depth-route-v250p"
REPEAT = ROOT / "artifacts/ape-articulated-depth-route-v252p"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_capture(left: Path, right: Path) -> None:
    with np.load(left) as a, np.load(right) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def check_capture(entry: dict, model: mujoco.MjModel,
                  data: mujoco.MjData) -> int:
    path = ROOT / entry["actor_frame_path"]
    assert digest(path) == entry["actor_frame_sha256"]
    with np.load(path) as packet:
        depth = packet["measured_depth_m"]
        finite = np.isfinite(depth)
        assert finite.any()
        assert np.all((depth[finite] >= .30) & (depth[finite] <= 1.00))
        mapped = packet["map_depth_m"]
        assert not np.any(np.isfinite(mapped) & ~np.isfinite(
            packet["filtered_depth_m"]))
        data.qpos[:] = packet["actor_qpos"]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        cid = model.camera("lower_rgb").id
        np.testing.assert_allclose(data.cam_xpos[cid],
                                   packet["camera_origin_estimated_m"],
                                   atol=1e-5)
        np.testing.assert_allclose(data.cam_xmat[cid].reshape(3, 3),
                                   packet["camera_rotation_estimated"],
                                   atol=1e-5)
        assert np.isclose(float(packet["capture_time_s"]),
                          entry["capture_time_s"], atol=1e-4)
        return int(finite.sum())


def check_gate(stage: dict) -> tuple[str, float]:
    gate = stage["actor_gate"]
    assert digest(ROOT / gate["packet_path"]) == gate["packet_sha256"]
    assert digest(ROOT / gate["shin_frame_path"]) == gate[
        "shin_frame_sha256"]
    packet = json.loads((ROOT / gate["packet_path"]).read_text())
    assert len(packet["ray_packets"]) == 4
    with np.load(ROOT / gate["shin_frame_path"]) as shin:
        assert len(shin["estimated_environment_points_xyz_m"]) == gate[
            "shin_points"]
        np.testing.assert_allclose(
            shin["estimated_pose_xyyaw"], packet["estimated_pose_xyyaw"])
        assert np.isclose(float(shin["capture_time_s"]),
                          packet["capture_time_s"])
    shell = gate["shell_plus_5cm"]
    assert shell["free"] + shell["occupied"] + shell["unknown"] == shell[
        "samples"]
    assert np.isclose(shell["margin_m"], .05)
    gap = min(gate["actor_shin_swept_gap_m"],
              gate["actor_upper_occupied_swept_gap_m"])
    should_be_free = (shell["status"] == "free" and
                      gate["oldest_head_age_s"] <= .800001 and
                      gate["fresh_shin_age_s"] <= .4 and
                      gate["shin_points"] >= 8 and
                      gap >= gate["point_motion_margin_m"])
    assert (gate["status"] == "free") == should_be_free
    assert stage["actor_motion_authorized"] == should_be_free
    return gate["status"], gap


def video_frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        count = 0
        for frame in reader:
            assert frame.ndim == 3 and frame.shape[2] == 3
            count += 1
        return count


def main() -> None:
    base = json.loads((BASE / "blocked-forward-viewpoint-audit.json").read_text())
    repeated_path = REPEAT / "blocked-forward-viewpoint-audit.json"
    repeated = json.loads(repeated_path.read_text())
    wrapper = json.loads((REPEAT / "report.json").read_text())
    assert wrapper["runner_report_sha256"] == digest(repeated_path)
    assert wrapper["source_sha256"] == digest(ROOT /
        "experiments/ape_high_level/closed_loop/run_slip_aware_third_view_v252p.py")
    assert len(base["rows"]) == len(repeated["rows"]) == 4
    model = model_with_room(visual=selected_visual())
    data = mujoco.MjData(model)
    rows = []
    paired_packets = 0
    fresh_packets = 0
    for old, new in zip(base["rows"], repeated["rows"], strict=True):
        assert old["case"] == new["case"]
        assert digest(ROOT / old["trace_path"]) == old["trace_sha256"]
        assert digest(ROOT / new["trace_path"]) == new["trace_sha256"]
        assert digest(ROOT / new["video_path"]) == new["video_sha256"]
        old_trace = json.loads((ROOT / old["trace_path"]).read_text())
        trace = json.loads((ROOT / new["trace_path"]).read_text())
        old_recovery = old_trace["recovery_inputs"][0]
        recovery = trace["recovery_inputs"][0]
        assert len(recovery["continuation"]["stages"]) == 2
        old_stage = old_recovery["continuation"]
        stages = recovery["continuation"]["stages"]
        for a, b in zip(old_trace["side_probe_inputs"] +
                        old_recovery["head_inputs"] + old_stage["head_inputs"],
                        trace["side_probe_inputs"] +
                        recovery["head_inputs"] + stages[0]["head_inputs"],
                        strict=True):
            same_capture(ROOT / a["actor_frame_path"],
                         ROOT / b["actor_frame_path"])
            paired_packets += 1
        assert old_trace["slips_evaluator_only"] == trace[
            "slips_evaluator_only"][:len(old_trace["slips_evaluator_only"])]
        stage_statuses = []
        stage_gaps = []
        for stage in stages:
            assert len(stage["head_inputs"]) == 3
            for entry in stage["head_inputs"]:
                check_capture(entry, model, data)
                fresh_packets += 1
            status, gap = check_gate(stage)
            stage_statuses.append(status)
            stage_gaps.append(gap)
        assert old_stage["actor_gate"]["status"] == stage_statuses[0]
        assert new["base_moves"] - old["base_moves"] == int(
            stage_statuses[1] == "free")
        resolution = next(event for event in trace["controller_events"]
                          if event["action"] == "route_help_resolution")
        assert len(resolution["actor_gates"]) == (
            2 + int(stage_statuses[1] == "free"))
        assert all(gate["status"] == "free" for gate in resolution[
            "actor_gates"])
        assert np.isclose(trace["evaluator_path_min_clearance"][0],
                          new["evaluator_path_min_clearance_m"], atol=1e-5)
        assert not new["evaluator_motion_interrupted"]
        assert new["evaluator_path_min_clearance_m"] > 0.
        if new["status"] == "owner_confirmed_forward_visual_only":
            assert new["owner_questions"] == 1
            accepted = [event for event in trace["controller_events"]
                        if event["action"] == "view" and
                        event["accepted_box"] is not None]
            assert accepted[-2]["evaluator_correct_item"]
            assert accepted[-1]["label"] == "post-owner forward fresh"
            assert accepted[-1]["accepted_box"] is not None
        else:
            assert new["status"] == "forward_viewpoint_advanced_no_item"
            assert new["owner_questions"] == 0
        frames = video_frames(ROOT / new["video_path"])
        assert frames > 10
        rows.append({"seed": new["case"][0], "status": new["status"],
                     "base_moves": new["base_moves"],
                     "gate_statuses": ["free", *stage_statuses],
                     "third_gate_point_gap_m": round(stage_gaps[1], 5),
                     "minimum_modeled_clearance_m": new[
                         "evaluator_path_min_clearance_m"],
                     "video_frames": frames})
    audit = {"schema": "robot-learning-lab.repeated-viewpoint-audit.v1",
             "base_report_sha256": digest(BASE /
                 "blocked-forward-viewpoint-audit.json"),
             "repeated_report_sha256": digest(repeated_path),
             "paired_captures_compared": paired_packets,
             "head_depth_captures_checked": fresh_packets,
             "videos_decoded": len(rows), "all_checks_pass": True,
             "rows": rows}
    (REPEAT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
