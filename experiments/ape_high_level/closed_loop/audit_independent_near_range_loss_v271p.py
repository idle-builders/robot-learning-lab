"""Pair V268p healthy and V270p missing-near-range full owner runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_repeated_viewpoints_v253p import same_capture
from run import ROOT


HEALTHY = ROOT / "artifacts/ape-articulated-depth-route-v268p"
LOSS = ROOT / "artifacts/ape-articulated-depth-route-v270p"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for frame in reader if frame.ndim == 3 and
                   frame.shape[2] == 3)


def main() -> None:
    healthy_path = HEALTHY / "blocked-forward-viewpoint-audit.json"
    loss_path = LOSS / "blocked-forward-viewpoint-audit.json"
    healthy = json.loads(healthy_path.read_text())
    loss = json.loads(loss_path.read_text())
    wrapper = json.loads((LOSS / "report.json").read_text())
    assert wrapper["runner_report_sha256"] == digest(loss_path)
    assert json.loads((LOSS / "predeclared-challenge.json").read_text())[
        "source_sha256"] == digest(
            LOSS / "run_independent_near_range_loss_v270p.py")
    assert digest(HEALTHY / "geometry-preflight.json") == digest(
        LOSS / "geometry-preflight.json")
    a = json.loads((HEALTHY / "predeclared-challenge.json").read_text())
    b = json.loads((LOSS / "predeclared-challenge.json").read_text())
    a.pop("source_sha256")
    b.pop("source_sha256")
    assert a.pop("sensor_profile") != b.pop("sensor_profile")
    assert a == b
    assert len(healthy["rows"]) == len(loss["rows"]) == 6
    side_pairs = 0
    recovery_pairs = 0
    wrong_success_claims = 0
    rows = []
    for good, bad in zip(healthy["rows"], loss["rows"], strict=True):
        assert good["case"] == bad["case"]
        for row in (good, bad):
            assert digest(ROOT / row["trace_path"]) == row["trace_sha256"]
            assert digest(ROOT / row["video_path"]) == row["video_sha256"]
        prior = json.loads((ROOT / good["trace_path"]).read_text())
        changed = json.loads((ROOT / bad["trace_path"]).read_text())
        assert changed["slips_evaluator_only"] == prior[
            "slips_evaluator_only"][:len(changed["slips_evaluator_only"])]
        for one, two in zip(prior["side_probe_inputs"],
                            changed["side_probe_inputs"], strict=True):
            assert digest(ROOT / two["actor_frame_path"]) == two[
                "actor_frame_sha256"]
            same_capture(ROOT / one["actor_frame_path"],
                         ROOT / two["actor_frame_path"])
            side_pairs += 1
        if good["recovery_gate"] is not None:
            assert bad["recovery_gate"] is not None
            left = prior["recovery_inputs"][0]
            right = changed["recovery_inputs"][0]
            for one, two in zip(left["head_inputs"], right["head_inputs"],
                                strict=True):
                same_capture(ROOT / one["actor_frame_path"],
                             ROOT / two["actor_frame_path"])
                recovery_pairs += 1
            gate_a, gate_b = left["actor_gate"], right["actor_gate"]
            rays_a = json.loads((ROOT / gate_a["packet_path"]).read_text())
            rays_b = json.loads((ROOT / gate_b["packet_path"]).read_text())
            assert rays_a["capture_time_s"] == rays_b["capture_time_s"]
            assert rays_a["estimated_pose_xyyaw"] == rays_b[
                "estimated_pose_xyyaw"]
            assert all(len(rays_a["ray_packets"][name]) > 0 for name in
                       rays_a["ray_packets"])
            assert all(len(rays_b["ray_packets"][name]) == 0 for name in
                       rays_b["ray_packets"])
            same_capture(ROOT / gate_a["shin_frame_path"],
                         ROOT / gate_b["shin_frame_path"])
        else:
            assert bad["recovery_gate"] is None
        count = frames(ROOT / bad["video_path"])
        assert count > 5
        assert bad["evaluator_path_min_clearance_m"] >= .05
        assert not bad["evaluator_motion_interrupted"]
        if (bad["case"][1] != "present" and bad["status"] in
                ("owner_confirmed_visual_only",
                 "owner_confirmed_forward_visual_only",
                 "owner_assisted_visual_reacquisition")):
            wrong_success_claims += 1
        rows.append({"seed": good["case"][0], "mode": good["case"][1],
                     "healthy_status": good["status"],
                     "loss_status": bad["status"],
                     "healthy_gate": (None if good["recovery_gate"] is None
                                      else good["recovery_gate"]["status"]),
                     "loss_gate": (None if bad["recovery_gate"] is None
                                   else bad["recovery_gate"]["status"]),
                     "healthy_moves": good["base_moves"],
                     "loss_moves": bad["base_moves"],
                     "loss_minimum_clearance_m": bad[
                         "evaluator_path_min_clearance_m"],
                     "loss_video_frames": count})
    audit = {"schema": "robot-learning-lab.independent-near-range-loss-audit.v1",
             "healthy_report_sha256": digest(healthy_path),
             "loss_report_sha256": digest(loss_path),
             "paired_side_captures": side_pairs,
             "paired_recovery_head_captures": recovery_pairs,
             "videos_decoded": len(rows),
             "wrong_success_claims": wrong_success_claims,
             "all_integrity_checks_pass": True,
             "rows": rows}
    (LOSS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
