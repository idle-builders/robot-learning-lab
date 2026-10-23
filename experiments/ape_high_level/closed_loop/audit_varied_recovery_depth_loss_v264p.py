"""Verify that V263p removes only recovery-map head depth in V261p rooms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_repeated_viewpoints_v253p import same_capture
from run import ROOT


HEALTHY = ROOT / "artifacts/ape-articulated-depth-route-v261p"
LOSS = ROOT / "artifacts/ape-articulated-depth-route-v263p"


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
    assert digest(HEALTHY / "geometry-preflight.json") == digest(
        LOSS / "geometry-preflight.json")
    a = json.loads((HEALTHY / "predeclared-challenge.json").read_text())
    b = json.loads((LOSS / "predeclared-challenge.json").read_text())
    a.pop("source_sha256")
    b.pop("source_sha256")
    assert a.pop("sensor_profile") != b.pop("sensor_profile")
    assert a == b
    assert len(healthy["rows"]) == len(loss["rows"]) == 4
    side_pairs = 0
    recovery_pairs = 0
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
        seed = good["case"][0]
        if good["recovery_gate"] is not None:
            assert bad["recovery_gate"] is not None
            left = prior["recovery_inputs"][0]
            right = changed["recovery_inputs"][0]
            assert len(left["head_inputs"]) == len(right["head_inputs"]) == 3
            for one, two in zip(left["head_inputs"], right["head_inputs"],
                                strict=True):
                assert digest(ROOT / two["actor_frame_path"]) == two[
                    "actor_frame_sha256"]
                with np.load(ROOT / one["actor_frame_path"]) as source, \
                     np.load(ROOT / two["actor_frame_path"]) as missing:
                    for key in ("rgb", "camera_origin_estimated_m",
                                "camera_rotation_estimated", "actor_qpos",
                                "estimated_pose_xyyaw", "capture_time_s"):
                        np.testing.assert_array_equal(source[key], missing[key])
                    assert np.isfinite(source["measured_depth_m"]).any()
                    assert not np.isfinite(missing["measured_depth_m"]).any()
                    assert not np.isfinite(missing["map_depth_m"]).any()
                recovery_pairs += 1
            gate_a, gate_b = left["actor_gate"], right["actor_gate"]
            assert json.loads((ROOT / gate_a["packet_path"]).read_text()) == \
                json.loads((ROOT / gate_b["packet_path"]).read_text())
            same_capture(ROOT / gate_a["shin_frame_path"],
                         ROOT / gate_b["shin_frame_path"])
        else:
            assert bad["recovery_gate"] is None
            assert prior["slips_evaluator_only"] == changed[
                "slips_evaluator_only"]
        count = frames(ROOT / bad["video_path"])
        assert count > 5
        assert bad["evaluator_path_min_clearance_m"] >= .05
        assert not bad["evaluator_motion_interrupted"]
        rows.append({"seed": seed,
                     "healthy_status": good["status"],
                     "loss_status": bad["status"],
                     "healthy_gate": (None if good["recovery_gate"] is None
                                      else good["recovery_gate"]["status"]),
                     "loss_gate": (None if bad["recovery_gate"] is None
                                   else bad["recovery_gate"]["status"]),
                     "loss_minimum_clearance_m": bad[
                         "evaluator_path_min_clearance_m"],
                     "loss_video_frames": count})
    audit = {"schema": "robot-learning-lab.varied-recovery-depth-loss-audit.v1",
             "healthy_report_sha256": digest(healthy_path),
             "loss_report_sha256": digest(loss_path),
             "paired_side_captures": side_pairs,
             "paired_recovery_captures": recovery_pairs,
             "videos_decoded": len(rows),
             "all_checks_pass": True, "rows": rows}
    (LOSS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
