"""Independent packet, pose, route-status and media audit for V246p."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from direct_structural_rooms_v270 import selected_visual
from run import ROOT, model_with_room


OUT = ROOT / "artifacts/ape-articulated-depth-route-v246p"
SOURCE = ROOT / "experiments/ape_high_level/closed_loop/"
ROOT_REPORT = OUT / "report.json"
CASES = (170, 172, 174, 176)
PROFILES = ("medium_30cm_1m", "head_depth_dropout")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_count(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        count = 0
        for frame in reader:
            assert frame.ndim == 3 and frame.shape[2] == 3
            count += 1
        return count


def check_packet(path: Path, expected_hash: str, profile: str,
                 model: mujoco.MjModel, data: mujoco.MjData) -> int:
    assert digest(path) == expected_hash
    with np.load(path) as packet:
        depth = packet["measured_depth_m"]
        finite = np.isfinite(depth)
        if profile == "head_depth_dropout":
            assert not finite.any()
        else:
            assert finite.any()
            assert np.all((depth[finite] >= .30) & (depth[finite] <= 1.00))
        filtered = packet["filtered_depth_m"]
        assert not np.any(np.isfinite(filtered) & ~finite)
        data.qpos[:] = packet["actor_qpos"]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        # Side and recovery captures in V321 use lower_rgb.
        cid = model.camera("lower_rgb").id
        np.testing.assert_allclose(data.cam_xpos[cid],
                                   packet["camera_origin_estimated_m"],
                                   atol=1e-5)
        np.testing.assert_allclose(
            data.cam_xmat[cid].reshape(3, 3),
            packet["camera_rotation_estimated"], atol=1e-5)
        if "map_depth_m" in packet:
            mapped = packet["map_depth_m"]
            assert not np.any(np.isfinite(mapped) & ~np.isfinite(filtered))
        return int(finite.sum())


def main() -> None:
    root = json.loads(ROOT_REPORT.read_text())
    assert root["schema"] == "robot-learning-lab.medium-depth-recovery.v1"
    assert root["source_sha256"] == digest(
        SOURCE / "run_medium_depth_recovery_v246p.py")
    assert tuple(tuple(c) for c in root["cases"]) == tuple(tuple(c) for c in
        __import__("run_medium_depth_recovery_v246p").CASES)
    assert tuple(p["name"] for p in root["profiles"]) == PROFILES
    model = model_with_room(visual=selected_visual())
    data = mujoco.MjData(model)
    rows = []
    packet_count = 0
    video_count = 0
    for profile in root["profiles"]:
        name = profile["name"]
        report_path = ROOT / profile["report_path"]
        assert digest(report_path) == profile["report_sha256"]
        report = json.loads(report_path.read_text())
        assert report["virtual_head_depth_profile"] == name
        assert len(report["rows"]) == len(CASES)
        assert {r["case"][0] for r in report["rows"]} == set(CASES)
        for row in report["rows"]:
            seed = row["case"][0]
            trace_path = ROOT / row["trace_path"]
            video_path = ROOT / row["video_path"]
            assert digest(trace_path) == row["trace_sha256"]
            assert digest(video_path) == row["video_sha256"]
            trace = json.loads(trace_path.read_text())
            assert trace["case"] == row["case"]
            assert len(trace["recovery_inputs"]) == 1
            recovery = trace["recovery_inputs"][0]
            gate = recovery["actor_gate"]
            assert gate == row["recovery_gate"]
            assert np.isclose(trace["evaluator_path_min_clearance"][0],
                              row["evaluator_path_min_clearance_m"], atol=1e-5)
            moving = bool(gate["status"] == "free")
            resolution = next(e for e in trace["controller_events"]
                              if e["action"] == "route_help_resolution")
            assert bool(resolution["actor_motion_authorized"]) == moving
            packets = trace["side_probe_inputs"] + recovery["head_inputs"]
            valid_pixels = []
            for entry in packets:
                valid_pixels.append(check_packet(
                    ROOT / entry["actor_frame_path"],
                    entry["actor_frame_sha256"], name, model, data))
                packet_count += 1
            assert digest(ROOT / gate["packet_path"]) == gate["packet_sha256"]
            assert digest(ROOT / gate["shin_frame_path"]) == gate[
                "shin_frame_sha256"]
            frames = frame_count(video_path)
            assert frames > len(packets)
            video_count += 1
            rows.append({"seed": seed, "profile": name,
                         "status": row["status"],
                         "gate": gate["status"],
                         "owner_questions": row["owner_questions"],
                         "modeled_min_gap_m": row[
                             "evaluator_path_min_clearance_m"],
                         "recovery_unknown_cells": gate[
                             "shell_plus_5cm"]["unknown"],
                         "valid_head_depth_pixels": valid_pixels,
                         "video_frames": frames})
    audit = {"schema": "robot-learning-lab.medium-depth-recovery-audit.v1",
             "source_report_sha256": digest(ROOT_REPORT),
             "packets_verified": packet_count,
             "videos_decoded": video_count,
             "all_checks_pass": True,
             "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows),
                      "packets": packet_count, "videos": video_count,
                      "outcomes": [(r["seed"], r["profile"], r["status"],
                                    r["modeled_min_gap_m"]) for r in rows]},
                     indent=2))


if __name__ == "__main__":
    main()
