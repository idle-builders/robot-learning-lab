"""Verify that V248p changes only recovery head depth on the paired owner run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT


MEDIUM = ROOT / "artifacts/ape-articulated-depth-route-v246p/medium_30cm_1m"
LOSS = ROOT / "artifacts/ape-articulated-depth-route-v248p"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(path: Path) -> int:
    with imageio.get_reader(path) as reader:
        return sum(1 for _ in reader)


def assert_equal_packets(a: Path, b: Path, keys: tuple[str, ...]) -> None:
    with np.load(a) as left, np.load(b) as right:
        for key in keys:
            np.testing.assert_array_equal(left[key], right[key])


def main() -> None:
    base = json.loads((MEDIUM / "blocked-forward-viewpoint-audit.json").read_text())
    changed_path = LOSS / "blocked-forward-viewpoint-audit.json"
    changed = json.loads(changed_path.read_text())
    wrapper = json.loads((LOSS / "report.json").read_text())
    assert wrapper["runner_report_sha256"] == digest(changed_path)
    assert changed["virtual_head_depth_profile"] == "recovery_only_depth_dropout"
    assert len(base["rows"]) == len(changed["rows"]) == 4
    paired = []
    checked_side = 0
    checked_recovery = 0
    for base_row, loss_row in zip(base["rows"], changed["rows"], strict=True):
        seed = base_row["case"][0]
        assert base_row["case"] == loss_row["case"]
        assert digest(ROOT / base_row["trace_path"]) == base_row["trace_sha256"]
        assert digest(ROOT / loss_row["trace_path"]) == loss_row["trace_sha256"]
        assert digest(ROOT / loss_row["video_path"]) == loss_row["video_sha256"]
        base_trace = json.loads((ROOT / base_row["trace_path"]).read_text())
        loss_trace = json.loads((ROOT / loss_row["trace_path"]).read_text())
        assert len(base_trace["side_probe_inputs"]) == len(
            loss_trace["side_probe_inputs"])
        for x, y in zip(base_trace["side_probe_inputs"],
                        loss_trace["side_probe_inputs"], strict=True):
            assert x["capture_time_s"] == y["capture_time_s"]
            assert x["allowed"] == y["allowed"]
            assert_equal_packets(
                ROOT / x["actor_frame_path"], ROOT / y["actor_frame_path"],
                ("rgb", "measured_depth_m", "filtered_depth_m",
                 "camera_origin_estimated_m", "camera_rotation_estimated",
                 "actor_qpos", "estimated_pose_xyyaw", "capture_time_s"))
            checked_side += 1
        assert loss_trace["slips_evaluator_only"] == base_trace[
            "slips_evaluator_only"][:len(loss_trace["slips_evaluator_only"])]
        base_recovery = base_trace["recovery_inputs"][0]
        loss_recovery = loss_trace["recovery_inputs"][0]
        assert len(base_recovery["head_inputs"]) == len(
            loss_recovery["head_inputs"]) == 3
        for x, y in zip(base_recovery["head_inputs"],
                        loss_recovery["head_inputs"], strict=True):
            assert x["capture_time_s"] == y["capture_time_s"]
            with np.load(ROOT / x["actor_frame_path"]) as healthy, \
                 np.load(ROOT / y["actor_frame_path"]) as missing:
                for key in ("rgb", "camera_origin_estimated_m",
                            "camera_rotation_estimated", "actor_qpos",
                            "estimated_pose_xyyaw", "capture_time_s"):
                    np.testing.assert_array_equal(healthy[key], missing[key])
                assert np.isfinite(healthy["measured_depth_m"]).any()
                assert not np.isfinite(missing["measured_depth_m"]).any()
                assert not np.isfinite(missing["filtered_depth_m"]).any()
                assert not np.isfinite(missing["map_depth_m"]).any()
            checked_recovery += 1
        healthy_gate = base_recovery["actor_gate"]
        loss_gate = loss_recovery["actor_gate"]
        near_a = json.loads((ROOT / healthy_gate["packet_path"]).read_text())
        near_b = json.loads((ROOT / loss_gate["packet_path"]).read_text())
        assert near_a == near_b
        with np.load(ROOT / healthy_gate["shin_frame_path"]) as shin_a, \
             np.load(ROOT / loss_gate["shin_frame_path"]) as shin_b:
            for key in shin_a.files:
                np.testing.assert_array_equal(shin_a[key], shin_b[key])
        count = frames(ROOT / loss_row["video_path"])
        assert count > 3
        paired.append({"seed": seed,
                       "medium_gate": healthy_gate["status"],
                       "loss_gate": loss_gate["status"],
                       "medium_status": base_row["status"],
                       "loss_status": loss_row["status"],
                       "medium_unknown": healthy_gate[
                           "shell_plus_5cm"]["unknown"],
                       "loss_unknown": loss_gate[
                           "shell_plus_5cm"]["unknown"],
                       "loss_min_gap_m": loss_row[
                           "evaluator_path_min_clearance_m"],
                       "loss_video_frames": count})
    audit = {"schema": "robot-learning-lab.recovery-only-depth-loss-audit.v1",
             "medium_report_sha256": digest(
                 MEDIUM / "blocked-forward-viewpoint-audit.json"),
             "loss_report_sha256": digest(changed_path),
             "paired_side_packets": checked_side,
             "paired_recovery_captures": checked_recovery,
             "all_checks_pass": True,
             "rows": paired}
    (LOSS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
