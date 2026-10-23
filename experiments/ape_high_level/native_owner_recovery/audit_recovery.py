"""Audit the retained owner-recovery report and decodable video evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "artifacts/ape-native-owner-recovery/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.report_root.resolve()
    report = json.loads((out / "report.json").read_text())
    checks = []
    assert report["schema"] == "robot-learning-lab.native-owner-recovery.v1"
    assert {row["seed"] for row in report["rows"]} == {85, 86}
    assert len(report["rows"]) == 8
    for name, digest in report["source_sha256"].items():
        source = {"owner_actor": ROOT / "experiments/ape_high_level/closed_loop/run.py",
                  "recovery": ROOT / "experiments/ape_high_level/native_owner_recovery/run_recovery.py",
                  "native_adapter": ROOT / "experiments/ape_high_level/native_owner_bridge/replay_owner_packets.py",
                  "native_qpos": ROOT / "artifacts/ape-cove50-transfer-20261008/varied150-forward-seed144.npz",
                  "native_odometry": ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1/varied150-forward-seed144-trace.json"}[name]
        assert sha(source) == digest, name
        checks.append(f"{name} source hash matches")
    for row in report["rows"]:
        assert sha(ROOT / row["owner_trace"]) == row["owner_trace_sha256"]
        if row["condition"] == "absent":
            assert row["actor_memory_available"] is False
            assert row["post_stop_actions"] == 0
            assert row["actor_response"] == "ask_owner_to_show_again"
            assert sha(ROOT / row["owner_video"]) == row["owner_video_sha256"]
            checks.append(f"seed {row['seed']} absent abstains without a look")
            continue
        assert sha(ROOT / row["video"]) == row["video_sha256"]
        video = cv2.VideoCapture(str(ROOT / row["video"]))
        assert video.isOpened(), row["video"]
        decoded = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            assert frame.shape == (520, 1600, 3)
            decoded += 1
        video.release()
        assert decoded == row["video_frames"] and decoded > 10
        actions = row["actions"]
        assert actions[0]["action"] == "chest_wait"
        assert row["head_actions"] <= 1
        if actions[0]["observation"]["candidate"] is not None:
            assert row["head_actions"] == 0
        for action in actions:
            if "observation" not in action:
                continue
            observation = action["observation"]
            for key in ("rgb", "depth"):
                assert sha(ROOT / observation[key]) == observation[f"{key}_sha256"]
            assert observation["capture_time_s"] >= row["stop_time_s"]
            assert len(action["range_samples"]) >= 2
            assert not action["interrupted"]
        assert row["path_min_clearance_evaluator_only_m"] > 0
        if row["actor_response"] == "offer_candidate_for_owner_confirmation":
            assert row["owner_reply_scripted_after_offer"] in ("yes", "no_show_again")
        else:
            assert row["owner_reply_scripted_after_offer"] is None
        assert row["owner_confirmed_reacquisition"] == (row["owner_reply_scripted_after_offer"] == "yes")
        if row["condition"] == "present":
            assert row["owner_reply_scripted_after_offer"] == "yes"
        if row["condition"] == "moved_target":
            assert row["owner_reply_scripted_after_offer"] != "yes"
        if row["condition"] == "lookalike_replacement":
            assert row["owner_reply_scripted_after_offer"] != "yes"
        checks.append(f"seed {row['seed']} {row['condition']}: {decoded} video frames, "
                      f"{row['head_actions']} head looks, {row['actor_response']}")
    live = [row for row in report["rows"] if row["condition"] != "absent"]
    metrics = {
        "present_owner_confirmed": sum(row["owner_confirmed_reacquisition"]
                                       for row in live if row["condition"] == "present"),
        "false_candidate_offers": sum(row["false_candidate_offer_evaluator_only"]
                                      for row in live),
        "head_looks_after_chest_candidate": sum(
            row["head_actions"] for row in live
            if row["actions"][0]["observation"]["candidate"] is not None),
        "head_looks_without_final_candidate": sum(
            row["head_actions"] for row in live
            if row["actor_response"] == "ask_owner_to_show_again"),
        "head_looks_total": sum(row["head_actions"] for row in live),
        "absent_post_stop_actions": sum(row["post_stop_actions"] for row in report["rows"]
                                        if row["condition"] == "absent"),
    }
    audit = {"schema": "robot-learning-lab.native-owner-recovery-audit.v1",
             "report": str((out / "report.json").relative_to(ROOT)),
             "report_sha256": sha(out / "report.json"),
             "checks": checks, "metrics": metrics, "passed": True}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS", len(checks), "checks")


if __name__ == "__main__":
    main()
